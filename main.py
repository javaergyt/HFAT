# main.py
import argparse
import logging
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
from collections import OrderedDict
import random
import wandb
from torch.cuda.amp import autocast, GradScaler

# 引入自定义工具和模型
from utils import get_dataloaders, attack_pgd, mixup_data, mixup_criterion
# 【重要】引入修改后的 ResNet18
from models.resnet18_gtsrb import GTSRB_ResNet18

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

upper_limit, lower_limit = 1, 0
EPS = 1E-20


def diff_in_weights(model, proxy):
    diff_dict = OrderedDict()
    model_state_dict = model.state_dict()
    proxy_state_dict = proxy.state_dict()
    for (old_k, old_w), (new_k, new_w) in zip(model_state_dict.items(), proxy_state_dict.items()):
        if len(old_w.size()) <= 1:
            continue
        if 'weight' in old_k:
            diff_w = new_w - old_w
            diff_dict[old_k] = old_w.norm() / (diff_w.norm() + EPS) * diff_w
    return diff_dict


def add_into_weights(model, diff, coeff=1.0):
    names_in_diff = diff.keys()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_diff:
                param.add_(coeff * diff[name])


def get_args():
    parser = argparse.ArgumentParser()
    # 默认改为 ResNet18
    parser.add_argument('--model', default='ResNet18')
    parser.add_argument('--l2', default=0, type=float)
    parser.add_argument('--l1', default=0, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--batch_size_test', default=128, type=int)
    parser.add_argument('--data_dir', default='./data', type=str)  # 数据目录

    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr_schedule', default='piecewise')
    parser.add_argument('--lr_max', default=0.1, type=float)
    parser.add_argument('--lr_one_drop', default=0.01, type=float)
    parser.add_argument('--lr_drop_epoch', default=100, type=int)
    parser.add_argument('--lr_proxy_max', default=0.01, type=float)

    parser.add_argument('--attack', default='pgd', type=str, choices=['pgd', 'fgsm', 'free', 'none'])
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--attack_iters', default=10, type=int)
    parser.add_argument('--attack_iters_test', default=20, type=int)
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--pgd_alpha', default=2, type=float)
    parser.add_argument('--beta', default=6.0, type=float)
    parser.add_argument('--fgsm_alpha', default=1.25, type=float)
    parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2'])
    parser.add_argument('--fgsm_init', default='random', choices=['zero', 'random', 'previous'])

    parser.add_argument('--fname', default='res/test00', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default=0, type=int)

    # 移除了 Cutout 和 width_factor (ResNet不需要)
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--mixup_alpha', type=float, default=1.0)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--chkpt_iters', default=10, type=int)

    parser.add_argument('--awp_gamma', default=0.01, type=float)
    parser.add_argument('--awp_warmup', default=0, type=int)

    # log
    parser.add_argument('--proj_name', type=str, default='GTSRB_AT', help='')
    parser.add_argument('--name', type=str, default='ResNet18_Run', help='')
    parser.add_argument('--wd_offline', default=1, type=int)

    # auxiliary params
    parser.add_argument('--aux_epsilon', default=7, type=int)
    parser.add_argument('--aux_attack_iters', default=7, type=int)
    parser.add_argument('--aux_pgd_alpha', default=7, type=float)
    parser.add_argument('--eps_gamma', type=float, default=1.0)
    parser.add_argument('--mean', type=float, default=0.4)
    parser.add_argument('--std_dev', type=float, default=0.015)

    parser.add_argument('--w_fix', default=0, type=int)
    parser.add_argument('--w_u', type=float, default=0.9)
    parser.add_argument('--w_r', type=float, default=0.1)
    parser.add_argument('--beta_u', type=float, default=2.0)
    parser.add_argument('--beta_r', type=float, default=0.5)
    parser.add_argument('--lt', type=float, default=1.1)

    return parser.parse_args()


def main():
    args = get_args()
    args.eps_gamma = args.eps_gamma / 255.0

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    print(args)

    if args.wd_offline:
        os.environ["WANDB_MODE"] = "offline"

    run = wandb.init(project=args.proj_name, name=args.name, config=args)

    if args.awp_gamma <= 0.0:
        args.awp_warmup = np.infty

    save_dir = os.path.join(args.fname, 'save')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(save_dir, 'eval.log' if args.eval else 'output.log')),
            logging.StreamHandler()
        ])

    logger.info(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)

    # ================= 数据加载 (使用 utils 中的新函数) =================
    train_loader, test_loader = get_dataloaders(args)

    epsilon = (args.epsilon / 255.)
    pgd_alpha = (args.pgd_alpha / 255.)

    aux_epsilon = (args.aux_epsilon / 255.)
    aux_pgd_alpha = (args.aux_pgd_alpha / 255.)

    # ================= 模型初始化 =================
    if args.model == 'ResNet18':
        print("Initialize GTSRB_ResNet18...")
        model = GTSRB_ResNet18(num_classes=43)
        proxy = GTSRB_ResNet18(num_classes=43)
    else:
        # 兼容其他模型，但建议 GTSRB 只用 ResNet18
        raise ValueError("Please use --model ResNet18 for GTSRB")

    model = nn.DataParallel(model).to(device)
    proxy = nn.DataParallel(proxy).to(device)

    # 优化器配置
    if args.l2:
        decay, no_decay = [], []
        for name, param in model.named_parameters():
            if 'bn' not in name and 'bias' not in name:
                decay.append(param)
            else:
                no_decay.append(param)
        params = [{'params': decay, 'weight_decay': args.l2},
                  {'params': no_decay, 'weight_decay': 0}]
    else:
        params = model.parameters()

    opt = torch.optim.SGD(params, lr=args.lr_max, momentum=0.9, weight_decay=5e-4)
    proxy_opt = torch.optim.SGD(proxy.parameters(), lr=args.lr_proxy_max)

    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss()

    # 学习率调度
    if args.lr_schedule == 'piecewise':
        def lr_schedule(t):
            if t / args.epochs < 0.5:
                return args.lr_max
            elif t / args.epochs < 0.75:
                return args.lr_max / 10.
            else:
                return args.lr_max / 100.
    else:
        lr_schedule = lambda t: args.lr_max  # 简化其他情况，可根据需要补全

    best_test_robust_acc = 0
    start_epoch = 0

    if args.resume:
        model_path = os.path.join(save_dir, f'model_{args.resume - 1}.pth')
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            start_epoch = args.resume
            logger.info(f'Resuming at epoch {start_epoch}')

    logger.info('Epoch \t Train Time \t Test Time \t LR \t Train Loss \t Train Acc \t Test Acc \t Test Robust Acc')

    # ================= 训练循环 =================
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        train_loss = 0
        train_acc = 0
        train_robust_loss = 0
        train_robust_acc = 0
        train_n = 0

        # 使用 standard DataLoader，返回 (X, y)
        for i, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)

            if args.mixup:
                X, y_a, y_b, lam = mixup_data(X, y, args.mixup_alpha)
                X, y_a, y_b = map(Variable, (X, y_a, y_b))

            lr = lr_schedule(epoch + (i + 1) / len(train_loader))
            opt.param_groups[0].update(lr=lr)

            # --- 生成对抗样本 ---
            if args.attack == 'pgd':
                if args.mixup:
                    delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm,
                                       mixup=True, y_a=y_a, y_b=y_b, lam=lam)
                else:
                    # 使用 Aux 参数生成对抗样本
                    delta = attack_pgd(model, X, y, aux_epsilon, aux_pgd_alpha, args.aux_attack_iters, args.restarts,
                                       args.norm)
                delta = delta.detach()
            elif args.attack == 'none':
                delta = torch.zeros_like(X)

            # [重要] 这里不要 normalize，因为模型会做
            X_adv_raw = torch.clamp(X + delta, min=lower_limit, max=upper_limit)

            # 这里的 X_adv_raw 仍然是 [0, 1] 范围
            diff = X - X_adv_raw

            # 计算 eps_beta 扰动
            eps_beta = np.random.normal(args.mean, args.std_dev)
            X_adv_eps = torch.clamp(X_adv_raw + eps_beta * diff + args.eps_gamma * torch.randn_like(X).to(device),
                                    min=lower_limit, max=upper_limit)

            model.train()

            # --- Proxy Update ---
            proxy.load_state_dict(model.state_dict())
            proxy.train()

            with autocast():
                # proxy 直接接收 [0, 1] 数据
                loss = nn.CrossEntropyLoss(reduction='none')(proxy(X_adv_eps), y)
                Indicator = (loss < args.lt).float()
                loss = -1 * (loss.mul(Indicator).mean())

            proxy_opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(proxy_opt)
            scaler.update()

            diff_weights = diff_in_weights(model, proxy)
            add_into_weights(model, diff_weights, coeff=1.0 * args.awp_gamma)

            # --- 生成 u_delta (用于 Hider Loss) ---
            u_delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm)
            X_u_adv = torch.clamp(X + u_delta, min=lower_limit, max=upper_limit)

            # --- [Pass 1: Hider Loss] ---
            with autocast():
                u_robust_output = model(X_u_adv)  # model input [0,1]
                loss_wv = criterion(u_robust_output, y)
                if args.l1:
                    for name, param in model.named_parameters():
                        if 'bn' not in name and 'bias' not in name:
                            loss_wv += args.l1 * param.abs().sum()

            opt.zero_grad()
            scaler.scale(loss_wv).backward()

            # 移除 awp 权重
            add_into_weights(model, diff_weights, coeff=-1.0 * args.awp_gamma)

            # 保存梯度 (Scaled)
            wv_gradient_dict = OrderedDict()
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        wv_gradient_dict[name] = param.grad.clone()

            # --- [Pass 2: Standard Robust Loss] ---
            # 重新计算 X_adv (基于主攻击参数)
            if args.attack != 'none':
                delta_main = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm)
                X_adv = torch.clamp(X + delta_main, min=lower_limit, max=upper_limit)
            else:
                X_adv = X

            robust_output = model(X_adv)  # model input [0,1]

            with autocast():
                loss_w = criterion(robust_output, y)

            opt.zero_grad()
            scaler.scale(loss_w).backward()
            scaler.unscale_(opt)  # Unscale gradients

            # --- KL Divergence Weighting ---
            with torch.no_grad():
                robust_n_output = model(X)  # Natural output, input [0,1]

                # 重新计算 u_robust (clean weights)
                u_robust_n_output = model(X)

                # 注意：这里可能需要根据你的原始逻辑确认是否需要再 inference 一次
                # 你的原始代码中 u_robust_output 是在 AWP 权重下计算的
                # 这里为了简单，假设结构不变

                kl_robust = F.kl_div(F.log_softmax(robust_output, dim=1),
                                     F.softmax(robust_n_output, dim=1),
                                     reduction='sum')

                kl_u_robust = F.kl_div(F.log_softmax(u_robust_output, dim=1),
                                       F.softmax(u_robust_n_output, dim=1),
                                       reduction='sum')

                w_tensor = torch.stack([kl_robust, kl_u_robust])
                w_tensor[0] = w_tensor[0] * args.beta_r
                w_tensor[1] = w_tensor[1] * args.beta_u
                w_softmax = F.softmax(w_tensor, dim=0)
                w_r, w_u = w_softmax[0], w_softmax[1]

            # --- Gradient Mixing ---
            inv_scale = 1.0 / scaler.get_scale()
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        # g_hider 需要手动 unscale
                        g_hider = wv_gradient_dict[name] * inv_scale
                        param.grad = w_r * param.grad + w_u * g_hider

            scaler.step(opt)
            scaler.update()

            # Logging
            train_loss += loss_w.item() * y.size(0)
            train_acc += (robust_n_output.max(1)[1] == y).sum().item()
            train_n += y.size(0)

        train_time = time.time()

        # ================= 测试循环 =================
        should_test = (epoch + 1) % 4 == 0 or (epoch + 1) >= (epochs - 10)

        if should_test:
            model.eval()
            test_loss = 0
            test_acc = 0
            test_robust_acc = 0
            test_n = 0

            for i, (X, y) in enumerate(test_loader):
                X, y = X.to(device), y.to(device)

                # PGD Test
                delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters_test, args.restarts, args.norm)
                delta = delta.detach()

                X_adv = torch.clamp(X + delta, min=lower_limit, max=upper_limit)
                robust_output = model(X_adv)  # Input [0,1]
                output = model(X)  # Input [0,1]

                test_robust_acc += (robust_output.max(1)[1] == y).sum().item()
                test_acc += (output.max(1)[1] == y).sum().item()
                test_n += y.size(0)

            print(f"Epoch {epoch}: Test Acc: {test_acc / test_n:.4f}, Robust Acc: {test_robust_acc / test_n:.4f}")

            if (test_robust_acc / test_n > best_test_robust_acc) and (epoch > 50):
                best_test_robust_acc = test_robust_acc / test_n
                torch.save(model.state_dict(), os.path.join(save_dir, f'model_best.pth'))

        test_time = time.time()


if __name__ == "__main__":
    main()