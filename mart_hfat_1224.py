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

# 引入自定义工具和模型 (需确保这些文件在同级目录)
from utils import get_dataloaders, attack_pgd, mixup_data, mixup_criterion
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


def mart_loss(model, x_natural, y, x_adv, beta):
    """
    MART Loss 计算函数 (用于替换原有的 CrossEntropy)
    """
    kl = nn.KLDivLoss(reduction='none')
    batch_size = len(x_natural)

    # 获取输出
    logits = model(x_natural)
    logits_adv = model(x_adv)

    adv_probs = F.softmax(logits_adv, dim=1)

    # 1. BCE Loss (Boosted Cross Entropy)
    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
    new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])
    loss_bce = F.cross_entropy(logits_adv, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)

    # 2. KL Loss (Conditional)
    nat_probs = F.softmax(logits, dim=1)
    true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()

    loss_kl = kl(torch.log(adv_probs + 1e-12), nat_probs).sum(dim=1)
    loss_robust = (1.0 / batch_size) * torch.sum(loss_kl * (1. - true_probs))

    return loss_bce + beta * loss_robust, logits, logits_adv


def get_args():
    parser = argparse.ArgumentParser(description='HFAT (Focus on Hiders) + MART for GTSRB')

    # 基础设置
    parser.add_argument('--model', default='ResNet18')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--batch_size_test', default=128, type=int)
    parser.add_argument('--data_dir', default='./datasets', type=str)
    parser.add_argument('--epochs', default=50, type=int)

    # 优化器
    parser.add_argument('--lr_schedule', default='cosine', type=str)
    parser.add_argument('--lr_max', default=0.1, type=float)
    parser.add_argument('--lr_proxy_max', default=0.01, type=float)  # HFAT Proxy LR
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)

    # 攻击参数
    parser.add_argument('--attack', default='pgd', type=str)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--attack_iters', default=10, type=int)
    parser.add_argument('--attack_iters_test', default=20, type=int)
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--pgd_alpha', default=2, type=float)
    parser.add_argument('--norm', default='l_inf', type=str)

    # MART 参数
    parser.add_argument('--mart_beta', default=6.0, type=float, help='MART beta parameter')

    # HFAT / Focus-on-hiders 核心参数
    # 注意：这里的 awp_gamma 是 HFAT 中"辅助模型"的步长，并非传统 AWP 正则化
    parser.add_argument('--awp_gamma', default=0.01, type=float)
    parser.add_argument('--awp_warmup', default=0, type=int)
    parser.add_argument('--aux_epsilon', default=7, type=int)
    parser.add_argument('--aux_attack_iters', default=7, type=int)
    parser.add_argument('--aux_pgd_alpha', default=7, type=float)
    parser.add_argument('--eps_gamma', type=float, default=1.0)
    parser.add_argument('--mean', type=float, default=0.4)
    parser.add_argument('--std_dev', type=float, default=0.015)
    parser.add_argument('--lt', type=float, default=1.1)
    parser.add_argument('--beta_u', type=float, default=2.0)  # Hider 权重
    parser.add_argument('--beta_r', type=float, default=0.5)  # Robust 权重

    # Logging
    parser.add_argument('--fname', default='result/HFAT_MART', type=str)
    parser.add_argument('--proj_name', type=str, default='GTSRB_Compare')
    parser.add_argument('--name', type=str, default='HFAT_MART')
    parser.add_argument('--wd_offline', default=1, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--chkpt_iters', default=10, type=int)

    # 兼容性参数 (未使用但保持代码结构)
    parser.add_argument('--l2', default=0, type=float)
    parser.add_argument('--l1', default=0, type=float)
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--mixup_alpha', type=float, default=1.0)

    return parser.parse_args()


def main():
    args = get_args()

    # 归一化参数
    args.eps_gamma = args.eps_gamma / 255.0
    epsilon = (args.epsilon / 255.)
    pgd_alpha = (args.pgd_alpha / 255.)
    aux_epsilon = (args.aux_epsilon / 255.)
    aux_pgd_alpha = (args.aux_pgd_alpha / 255.)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    if args.wd_offline:
        os.environ["WANDB_MODE"] = "offline"
    wandb.init(project=args.proj_name, name=args.name, config=args)

    if args.awp_gamma <= 0.0:
        args.awp_warmup = np.infty

    save_dir = os.path.join(args.fname, 'save')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        handlers=[logging.FileHandler(os.path.join(save_dir, 'output.log')), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    logger.info(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, test_loader = get_dataloaders(args)

    logger.info("Initialize GTSRB_ResNet18 for HFAT + MART...")
    model = GTSRB_ResNet18(num_classes=43).to(device)
    proxy = GTSRB_ResNet18(num_classes=43).to(device)  # Proxy 用于 HFAT 的 Hider 发现

    model = nn.DataParallel(model)
    proxy = nn.DataParallel(proxy)

    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)
    proxy_opt = torch.optim.SGD(proxy.parameters(), lr=args.lr_proxy_max)

    criterion = nn.CrossEntropyLoss()

    if args.lr_schedule == 'cosine':
        def lr_schedule(t):
            return args.lr_max * 0.5 * (1 + math.cos(math.pi * t / args.epochs))
    else:
        def lr_schedule(t):
            if t / args.epochs < 0.5:
                return args.lr_max
            elif t / args.epochs < 0.75:
                return args.lr_max / 10.
            else:
                return args.lr_max / 100.

    best_test_robust_acc = 0

    logger.info(
        "{:<6} | {:<6} | {:<6} | {:<8} | {:<8} | {:<8} | {:<8} || {:<8} | {:<8} | {:<8} | {:<8}".format(
            "Epoch", "TrTime", "TeTime",
            "Tr_N_Acc", "Tr_R_Acc", "Tr_N_Loss", "Tr_R_Loss",
            "Te_N_Acc", "Te_R_Acc", "Te_N_Loss", "Te_R_Loss")
    )

    for epoch in range(0, args.epochs):
        start_time = time.time()

        train_stats = {'nat_loss': 0, 'nat_acc': 0, 'rob_loss': 0, 'rob_acc': 0, 'n': 0}
        model.train()

        for i, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)

            lr = lr_schedule(epoch + (i + 1) / len(train_loader))
            opt.param_groups[0].update(lr=lr)

            # ================= HFAT: 1. 辅助模型更新 (Proxy Update) =================
            # 这一步是为了让 Proxy 变得与当前 Model 略有不同，代表"潜在的未来风险区域"
            delta = attack_pgd(model, X, y, aux_epsilon, aux_pgd_alpha, args.aux_attack_iters, args.restarts, args.norm)
            delta = delta.detach()

            X_adv_raw = torch.clamp(X + delta, min=lower_limit, max=upper_limit)
            diff = X - X_adv_raw
            eps_beta = np.random.normal(args.mean, args.std_dev)
            X_adv_eps = torch.clamp(X_adv_raw + eps_beta * diff + args.eps_gamma * torch.randn_like(X).to(device),
                                    min=lower_limit, max=upper_limit)

            if isinstance(proxy, nn.DataParallel):
                proxy.module.load_state_dict(model.module.state_dict())
            else:
                proxy.load_state_dict(model.state_dict())

            proxy.train()
            loss_proxy = nn.CrossEntropyLoss(reduction='none')(proxy(X_adv_eps), y)
            Indicator = (loss_proxy < args.lt).float()
            loss_proxy = -1 * (loss_proxy.mul(Indicator).mean())  # 寻找容易忽略的样本
            proxy_opt.zero_grad()
            loss_proxy.backward()
            proxy_opt.step()

            # 计算 Model 和 Proxy 的权重差
            diff_weights = diff_in_weights(model, proxy)

            # [HFAT 关键步骤] 将权重推向 Proxy 方向，构建"辅助模型"
            if epoch >= args.awp_warmup:
                add_into_weights(model, diff_weights, coeff=1.0 * args.awp_gamma)

            # ================= HFAT: 2. Hider 梯度计算 =================
            # 在"辅助模型"上攻击，找到 Hiders (那些在当前模型能防住，但在辅助模型上防不住的样本)
            u_delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm)
            X_u_adv = torch.clamp(X + u_delta, min=lower_limit, max=upper_limit)

            u_robust_output = model(X_u_adv)
            loss_wv = criterion(u_robust_output, y)  # Hider 损失仍用 CE，为了纠正分类错误

            opt.zero_grad()
            loss_wv.backward()  # 计算 Hider 梯度

            # [HFAT 关键步骤] 恢复原始权重！确保主训练是在干净权重上进行 (非 AWP 正则化)
            if epoch >= args.awp_warmup:
                add_into_weights(model, diff_weights, coeff=-1.0 * args.awp_gamma)

            # 保存 Hider 梯度
            wv_gradient_dict = OrderedDict()
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        wv_gradient_dict[name] = param.grad.clone()

            # ================= MART: 3. 主分支训练 (替换了原 CrossEntropy) =================
            # 生成标准对抗样本 (在恢复后的干净模型上)
            delta_main = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm)
            X_adv = torch.clamp(X + delta_main, min=lower_limit, max=upper_limit)

            # [修改] 使用 MART Loss 计算主梯度
            loss_mart, logits_nat, logits_adv = mart_loss(model, X, y, X_adv, beta=args.mart_beta)

            opt.zero_grad()
            loss_mart.backward()  # 计算 MART 梯度

            # ================= HFAT: 4. 梯度混合 (Gradient Mixing) =================
            # 计算 KL 散度以动态调整权重
            with torch.no_grad():
                # 重新计算一次 Hider 分支在干净模型上的输出，用于 KL 计算
                u_robust_output_clean = model(X_u_adv)

                kl_robust = F.kl_div(F.log_softmax(logits_adv, dim=1),
                                     F.softmax(logits_nat, dim=1),
                                     reduction='sum')
                kl_u_robust = F.kl_div(F.log_softmax(u_robust_output_clean, dim=1),
                                       F.softmax(logits_nat, dim=1),
                                       reduction='sum')

                w_tensor = torch.stack([kl_robust, kl_u_robust])
                w_tensor[0] = w_tensor[0] * args.beta_r
                w_tensor[1] = w_tensor[1] * args.beta_u
                w_softmax = F.softmax(w_tensor, dim=0)
                w_r, w_u = w_softmax[0], w_softmax[1]

            # 混合 MART 梯度和 Hider 梯度
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None and name in wv_gradient_dict:
                        g_hider = wv_gradient_dict[name]
                        param.grad = w_r * param.grad + w_u * g_hider

            # MART 训练推荐加上梯度裁剪
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            opt.step()

            # --- 统计 ---
            with torch.no_grad():
                nat_acc = (logits_nat.max(1)[1] == y).float().sum().item()
                nat_loss = F.cross_entropy(logits_nat, y, reduction='sum').item()
                rob_acc = (logits_adv.max(1)[1] == y).float().sum().item()
                rob_loss = F.cross_entropy(logits_adv, y, reduction='sum').item()

                train_stats['nat_loss'] += nat_loss
                train_stats['nat_acc'] += nat_acc
                train_stats['rob_loss'] += rob_loss
                train_stats['rob_acc'] += rob_acc
                train_stats['n'] += y.size(0)

        train_time = time.time()

        # ================= 测试循环 (保持 PGD-20 标准) =================
        model.eval()
        test_stats = {'nat_loss': 0, 'nat_acc': 0, 'rob_loss': 0, 'rob_acc': 0, 'n': 0}

        for i, (X, y) in enumerate(test_loader):
            X, y = X.to(device), y.to(device)

            delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters_test, args.restarts, args.norm)
            X_adv = torch.clamp(X + delta, min=lower_limit, max=upper_limit)

            with torch.no_grad():
                output = model(X)
                robust_output = model(X_adv)

                test_stats['nat_loss'] += F.cross_entropy(output, y, reduction='sum').item()
                test_stats['nat_acc'] += (output.max(1)[1] == y).sum().item()
                test_stats['rob_loss'] += F.cross_entropy(robust_output, y, reduction='sum').item()
                test_stats['rob_acc'] += (robust_output.max(1)[1] == y).sum().item()
                test_stats['n'] += y.size(0)

        test_time = time.time()

        # 计算平均值
        tr_n_acc = train_stats['nat_acc'] / train_stats['n']
        tr_r_acc = train_stats['rob_acc'] / train_stats['n']
        tr_n_loss = train_stats['nat_loss'] / train_stats['n']
        tr_r_loss = train_stats['rob_loss'] / train_stats['n']

        te_n_acc = test_stats['nat_acc'] / test_stats['n']
        te_r_acc = test_stats['rob_acc'] / test_stats['n']
        te_n_loss = test_stats['nat_loss'] / test_stats['n']
        te_r_loss = test_stats['rob_loss'] / test_stats['n']

        logger.info(
            "{:<6d} | {:<6.1f} | {:<6.1f} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f} || {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f}".format(
                epoch, train_time - start_time, test_time - train_time,
                tr_n_acc, tr_r_acc, tr_n_loss, tr_r_loss,
                te_n_acc, te_r_acc, te_n_loss, te_r_loss)
        )

        wandb.log({
            "epoch": epoch,
            "lr": lr,
            "Train/Natural_Acc": tr_n_acc,
            "Train/Robust_Acc": tr_r_acc,
            "Test/Natural_Acc": te_n_acc,
            "Test/Robust_Acc": te_r_acc,
        })

        if te_r_acc > best_test_robust_acc:
            best_test_robust_acc = te_r_acc
            torch.save(model.state_dict(), os.path.join(save_dir, 'model_best.pth'))
            logger.info(f"==> Best Robust Acc: {best_test_robust_acc:.4f}")

        if (epoch + 1) % args.chkpt_iters == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f'model_{epoch}.pth'))


if __name__ == '__main__':
    main()