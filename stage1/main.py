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
# 移除 AMP 相关的引用
# from torch.cuda.amp import autocast, GradScaler

# 引入自定义工具和模型
from utils import get_dataloaders, attack_pgd, mixup_data, mixup_criterion
# 引入原版和DRM增强版 ResNet18
from models.resnet18_gtsrb import GTSRB_ResNet18
# 引入 WideResNet 模型
from models.wideresnet_gtsrb import gtsrb_wideresnet_28_10, gtsrb_wideresnet_34_10, gtsrb_wideresnet_28_20

# 尝试导入DRM增强版模型
try:
    from models.resnet18_gtsrb_drm import GTSRB_ResNet18_DRM, GTSRB_ResNet18_Lite_DRM
    DRM_AVAILABLE = True
    print("✓ DRM enhanced models imported successfully")
except ImportError:
    DRM_AVAILABLE = False
    print("Warning: DRM enhanced models not available")

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

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
    # === 模型选择 ===
    parser.add_argument('--model', default='ResNet18',
                       choices=['ResNet18', 'ResNet18_DRM', 'ResNet18_Lite_DRM',
                               'WideResNet_28_10', 'WideResNet_34_10', 'WideResNet_28_20'],
                       help='Model architecture to use')
    parser.add_argument('--drm_position', default='layer3',
                       choices=['layer2', 'layer3', 'layer4', 'multi'],
                       help='DRM module position (for ResNet18_DRM)')

    # === 原有参数保持不变 ===
    parser.add_argument('--l2', default=0, type=float)
    parser.add_argument('--l1', default=0, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--batch_size_test', default=128, type=int)
    parser.add_argument('--data_dir', default='./data', type=str)

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
    elif torch.backends.mps.is_available():
        print('MPS Device is being used')

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
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(args.seed)

    # ================= 数据加载 =================
    train_loader, test_loader = get_dataloaders(args)

    epsilon = (args.epsilon / 255.)
    pgd_alpha = (args.pgd_alpha / 255.)

    aux_epsilon = (args.aux_epsilon / 255.)
    aux_pgd_alpha = (args.aux_pgd_alpha / 255.)

    # ================= 模型初始化 =================
    def create_model(model_name, drm_position='layer3'):
        """统一的模型创建函数"""
        if model_name == 'ResNet18':
            logger.info("Initialize GTSRB_ResNet18...")
            return GTSRB_ResNet18(num_classes=43)
        elif model_name == 'ResNet18_DRM':
            if not DRM_AVAILABLE:
                raise ValueError("DRM models not available. Please check resnet18_gtsrb_drm.py")
            logger.info(f"Initialize GTSRB_ResNet18_DRM with DRM at {drm_position}...")
            return GTSRB_ResNet18_DRM(num_classes=43, drm_position=drm_position)
        elif model_name == 'ResNet18_Lite_DRM':
            if not DRM_AVAILABLE:
                raise ValueError("DRM models not available. Please check resnet18_gtsrb_drm.py")
            logger.info("Initialize GTSRB_ResNet18_Lite_DRM...")
            return GTSRB_ResNet18_Lite_DRM(num_classes=43)
        elif model_name == 'WideResNet_28_10':
            logger.info("Initialize WideResNet-28-10 for GTSRB...")
            return gtsrb_wideresnet_28_10(num_classes=43)
        elif model_name == 'WideResNet_34_10':
            logger.info("Initialize WideResNet-34-10 for GTSRB...")
            return gtsrb_wideresnet_34_10(num_classes=43)
        elif model_name == 'WideResNet_28_20':
            logger.info("Initialize WideResNet-28-20 for GTSRB...")
            return gtsrb_wideresnet_28_20(num_classes=43)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    # 创建模型
    model = create_model(args.model, args.drm_position)
    proxy = create_model(args.model, args.drm_position)

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

    # 移除 Scaler
    # scaler = GradScaler()
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
    elif args.lr_schedule == 'cosine':
        def lr_schedule(t):
            return args.lr_max * 0.5 * (1 + math.cos(math.pi * t / args.epochs))
    else:
        lr_schedule = lambda t: args.lr_max

    best_test_robust_acc = 0
    start_epoch = 0

    if args.resume:
        model_path = os.path.join(save_dir, f'model_{args.resume - 1}.pth')
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            start_epoch = args.resume
            logger.info(f'Resuming at epoch {start_epoch}')

    logger.info(
        'Epoch \t Train Time \t Test Time \t LR \t Train Loss \t Train Acc \t Train Robust Loss \t Train Robust Acc \t Test Loss \t Test Acc \t Test Robust Loss \t Test Robust Acc')

    # ================= 训练循环 =================
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()

        train_loss_nat = 0  # 自然 Loss
        train_acc = 0  # 自然 Acc
        train_loss_rob = 0  # 鲁棒 Loss
        train_acc_rob = 0  # 鲁棒 Acc
        train_n = 0

        # 训练模式
        model.train()

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
                    delta = attack_pgd(model, X, y, aux_epsilon, aux_pgd_alpha, args.aux_attack_iters, args.restarts,
                                       args.norm)
                delta = delta.detach()
            elif args.attack == 'none':
                delta = torch.zeros_like(X)

            X_adv_raw = torch.clamp(X + delta, min=lower_limit, max=upper_limit)
            diff = X - X_adv_raw

            eps_beta = np.random.normal(args.mean, args.std_dev)
            X_adv_eps = torch.clamp(X_adv_raw + eps_beta * diff + args.eps_gamma * torch.randn_like(X).to(device),
                                    min=lower_limit, max=upper_limit)

            # --- Proxy Update (FP32) ---
            if isinstance(proxy, nn.DataParallel):
                proxy.module.load_state_dict(model.module.state_dict())
            else:
                proxy.load_state_dict(model.state_dict())

            proxy.train()
            # 移除 autocast
            loss = nn.CrossEntropyLoss(reduction='none')(proxy(X_adv_eps), y)
            Indicator = (loss < args.lt).float()
            loss = -1 * (loss.mul(Indicator).mean())
            proxy_opt.zero_grad()
            loss.backward()  # 直接 backward
            proxy_opt.step()  # 直接 step
            # scaler.update()

            diff_weights = diff_in_weights(model, proxy)
            if epoch >= args.awp_warmup:
                add_into_weights(model, diff_weights, coeff=1.0 * args.awp_gamma)

            # --- Hider Loss (FP32) ---
            u_delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm)
            X_u_adv = torch.clamp(X + u_delta, min=lower_limit, max=upper_limit)

            # 移除 autocast
            u_robust_output = model(X_u_adv)
            loss_wv = criterion(u_robust_output, y)
            if args.l1:
                for name, param in model.named_parameters():
                    if 'bn' not in name and 'bias' not in name:
                        loss_wv += args.l1 * param.abs().sum()

            opt.zero_grad()
            loss_wv.backward()  # 直接 backward
            if epoch >= args.awp_warmup:
                add_into_weights(model, diff_weights, coeff=-1.0 * args.awp_gamma)

            wv_gradient_dict = OrderedDict()
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        wv_gradient_dict[name] = param.grad.clone()

            # --- Standard Robust Loss (FP32) ---
            if args.attack != 'none':
                delta_main = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm)
                X_adv = torch.clamp(X + delta_main, min=lower_limit, max=upper_limit)
            else:
                X_adv = X

            robust_output = model(X_adv)
            # 移除 autocast
            loss_w = criterion(robust_output, y)

            opt.zero_grad()
            loss_w.backward()  # 直接 backward
            # scaler.unscale_(opt)

            # --- KL Divergence ---
            with torch.no_grad():
                robust_n_output = model(X)
                u_robust_n_output = model(X)

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

            # --- Gradient Mixing (FP32) ---
            # 移除 scaler.get_scale()
            inv_scale = 1.0

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        g_hider = wv_gradient_dict[name] * inv_scale
                        param.grad = w_r * param.grad + w_u * g_hider

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()  # 直接 step
            # scaler.update()

            # --- 统计训练集指标 ---
            with torch.no_grad():
                loss_nat = criterion(robust_n_output, y)

                train_loss_nat += loss_nat.item() * y.size(0)
                train_acc += (robust_n_output.max(1)[1] == y).sum().item()

                train_loss_rob += loss_w.item() * y.size(0)
                train_acc_rob += (robust_output.max(1)[1] == y).sum().item()

                train_n += y.size(0)

        train_time = time.time()

        # ================= 测试循环 =================
        model.eval()

        test_loss_nat = 0
        test_acc = 0
        test_loss_rob = 0
        test_acc_rob = 0
        test_n = 0

        for i, (X, y) in enumerate(test_loader):
            X, y = X.to(device), y.to(device)

            # PGD Test
            delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters_test, args.restarts, args.norm)
            delta = delta.detach()

            X_adv = torch.clamp(X + delta, min=lower_limit, max=upper_limit)

            with torch.no_grad():
                robust_output = model(X_adv)
                output = model(X)

                # 计算 Loss
                loss_clean = criterion(output, y)
                loss_adv = criterion(robust_output, y)

                # 累加统计
                test_loss_nat += loss_clean.item() * y.size(0)
                test_acc += (output.max(1)[1] == y).sum().item()

                test_loss_rob += loss_adv.item() * y.size(0)
                test_acc_rob += (robust_output.max(1)[1] == y).sum().item()

                test_n += y.size(0)

        test_time = time.time()

        # 计算平均值
        avg_train_loss_nat = train_loss_nat / train_n
        avg_train_acc = train_acc / train_n
        avg_train_loss_rob = train_loss_rob / train_n
        avg_train_acc_rob = train_acc_rob / train_n

        avg_test_loss_nat = test_loss_nat / test_n
        avg_test_acc = test_acc / test_n
        avg_test_loss_rob = test_loss_rob / test_n
        avg_test_acc_rob = test_acc_rob / test_n

        # 打印所有指标
        logger.info(
            f"{epoch} \t {train_time - start_time:.1f} \t {test_time - train_time:.1f} \t {lr:.4f} \t "
            f"{avg_train_loss_nat:.4f} \t {avg_train_acc:.4f} \t {avg_train_loss_rob:.4f} \t {avg_train_acc_rob:.4f} \t "
            f"{avg_test_loss_nat:.4f} \t {avg_test_acc:.4f} \t {avg_test_loss_rob:.4f} \t {avg_test_acc_rob:.4f}"
        )

        # 记录到 WandB
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss_nat,
            "train_acc": avg_train_acc,
            "train_robust_loss": avg_train_loss_rob,
            "train_robust_acc": avg_train_acc_rob,
            "test_loss": avg_test_loss_nat,
            "test_acc": avg_test_acc,
            "test_robust_loss": avg_test_loss_rob,
            "test_robust_acc": avg_test_acc_rob,
            "lr": lr
        })

        # 保存最佳模型
        if avg_test_acc_rob > best_test_robust_acc and epoch > 0:
            best_test_robust_acc = avg_test_acc_rob
            torch.save(model.state_dict(), os.path.join(save_dir, f'model_best.pth'))
            logger.info(f"==> Best Robust Acc: {best_test_robust_acc:.4f} at epoch {epoch}")

        if (epoch + 1) % args.chkpt_iters == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f'model_{epoch}.pth'))


if __name__ == "__main__":
    main()