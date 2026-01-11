import argparse
import logging
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import wandb
from collections import OrderedDict

# 引入自定义工具和模型
from utils import get_dataloaders, attack_pgd
from models.resnet18_gtsrb import GTSRB_ResNet18
from models.wideresnet_gtsrb import gtsrb_wideresnet_28_10, gtsrb_wideresnet_34_10, gtsrb_wideresnet_28_20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch TRADES Adversarial Training for GTSRB')

    # 基础参数
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--batch_size_test', default=128, type=int)
    parser.add_argument('--data_dir', default='./data', type=str)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--model', default='ResNet18',
                       choices=['ResNet18', 'WideResNet_28_10', 'WideResNet_34_10', 'WideResNet_28_20'],
                       help='Model architecture to use')

    # 优化器与调度参数
    parser.add_argument('--lr_max', default=0.1, type=float)
    parser.add_argument('--lr_schedule', default='cosine', type=str, choices=['piecewise', 'cosine'])
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)

    # 对抗训练参数
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--num_steps', default=10, type=int)
    parser.add_argument('--step_size', default=2, type=float)
    parser.add_argument('--norm', default='l_inf', type=str)

    # 测试参数
    parser.add_argument('--attack_iters_test', default=20, type=int)
    parser.add_argument('--restarts', default=1, type=int)

    # TRADES 参数
    parser.add_argument('--beta', default=6.0, type=float)

    # 杂项
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--fname', default='result/TRADES_Baseline', type=str)
    parser.add_argument('--proj_name', type=str, default='GTSRB_Compare')
    parser.add_argument('--name', type=str, default='TRADES_Baseline')
    parser.add_argument('--wd_offline', default=1, type=int)

    return parser.parse_args()


def create_model(model_name):
    """创建指定的模型"""
    if model_name == 'ResNet18':
        return GTSRB_ResNet18(num_classes=43)
    elif model_name == 'WideResNet_28_10':
        return gtsrb_wideresnet_28_10(num_classes=43)
    elif model_name == 'WideResNet_34_10':
        return gtsrb_wideresnet_34_10(num_classes=43)
    elif model_name == 'WideResNet_28_20':
        return gtsrb_wideresnet_28_20(num_classes=43)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def trades_loss(model, x_natural, y, optimizer, step_size, epsilon, perturb_steps, beta, distance='l_inf'):
    """
    TRADES Loss 计算。
    返回:
    - loss: 用于反向传播的总 Loss (TRADES Loss)
    - logits_natural: 自然样本的输出 logits
    - logits_adv: 对抗样本的输出 logits
    """
    criterion_kl = nn.KLDivLoss(reduction='sum')
    model.eval()
    batch_size = len(x_natural)

    # 初始化对抗样本
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()

    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()
    x_adv = torch.autograd.Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    optimizer.zero_grad()

    logits_natural = model(x_natural)
    logits_adv = model(x_adv)

    loss_natural = F.cross_entropy(logits_natural, y)
    loss_robust_kl = (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_adv, dim=1),
                                                       F.softmax(logits_natural, dim=1))

    loss = loss_natural + beta * loss_robust_kl

    return loss, logits_natural, logits_adv


def main():
    args = get_args()
    args.epsilon_float = args.epsilon / 255.0
    args.step_size_float = args.step_size / 255.0

    if args.wd_offline:
        os.environ["WANDB_MODE"] = "offline"
    wandb.init(project=args.proj_name, name=args.name, config=args)

    save_dir = os.path.join(args.fname, 'save')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 设置 Logging
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

    logger.info("Initialize GTSRB_ResNet18 for TRADES...")
    model = GTSRB_ResNet18(num_classes=43).to(device)
    model = nn.DataParallel(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # 学习率调度
    if args.lr_schedule == 'cosine':
        def lr_schedule(t):
            return args.lr_max * 0.5 * (1 + math.cos(math.pi * t / args.epochs))
    elif args.lr_schedule == 'piecewise':
        def lr_schedule(t):
            if t / args.epochs < 0.5:
                return args.lr_max
            elif t / args.epochs < 0.75:
                return args.lr_max / 10.
            else:
                return args.lr_max / 100.
    else:
        def lr_schedule(t):
            return args.lr_max

    best_test_robust_acc = 0

    # 打印表头
    logger.info(
        "{:<6} | {:<6} | {:<6} | {:<8} | {:<8} | {:<8} | {:<8} || {:<8} | {:<8} | {:<8} | {:<8}".format(
            "Epoch", "TrTime", "TeTime",
            "Tr_N_Acc", "Tr_R_Acc", "Tr_N_Loss", "Tr_R_Loss",
            "Te_N_Acc", "Te_R_Acc", "Te_N_Loss", "Te_R_Loss")
    )

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        model.train()

        # 统计变量
        train_stats = {
            'nat_loss': 0, 'nat_acc': 0,
            'rob_loss': 0, 'rob_acc': 0,
            'n': 0
        }

        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # Update LR
            now_epoch = (epoch - 1) + (i + 1) / len(train_loader)
            lr = lr_schedule(now_epoch)
            optimizer.param_groups[0].update(lr=lr)

            # 计算 Loss 和 Logits
            loss, logits_nat, logits_adv = trades_loss(
                model=model, x_natural=data, y=target, optimizer=optimizer,
                step_size=args.step_size_float, epsilon=args.epsilon_float,
                perturb_steps=args.num_steps, beta=args.beta, distance=args.norm
            )

            loss.backward()
            optimizer.step()

            # --- 统计训练集指标 ---
            with torch.no_grad():
                # 自然指标
                nat_acc = (logits_nat.max(1)[1] == target).float().sum().item()
                nat_loss = F.cross_entropy(logits_nat, target, reduction='sum').item()

                # 鲁棒指标 (基于TRADES生成的对抗样本)
                # 注意：这里的 Loss 是 CE Loss，用于监控，而非 TRADES 优化的 KL Loss
                rob_acc = (logits_adv.max(1)[1] == target).float().sum().item()
                rob_loss = F.cross_entropy(logits_adv, target, reduction='sum').item()

                train_stats['nat_loss'] += nat_loss
                train_stats['nat_acc'] += nat_acc
                train_stats['rob_loss'] += rob_loss
                train_stats['rob_acc'] += rob_acc
                train_stats['n'] += target.size(0)

        train_time = time.time()

        # --- 测试阶段 ---
        model.eval()
        test_stats = {
            'nat_loss': 0, 'nat_acc': 0,
            'rob_loss': 0, 'rob_acc': 0,
            'n': 0
        }

        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            # 1. 自然样本评估
            with torch.no_grad():
                output = model(data)
                test_stats['nat_loss'] += F.cross_entropy(output, target, reduction='sum').item()
                test_stats['nat_acc'] += (output.max(1)[1] == target).sum().item()

            # 2. 对抗样本生成与评估 (PGD-20)
            delta = attack_pgd(model, data, target, epsilon=args.epsilon_float,
                               alpha=args.step_size_float, attack_iters=args.attack_iters_test,
                               restarts=args.restarts, norm=args.norm)

            with torch.no_grad():
                robust_output = model(torch.clamp(data + delta, 0, 1))
                test_stats['rob_loss'] += F.cross_entropy(robust_output, target, reduction='sum').item()
                test_stats['rob_acc'] += (robust_output.max(1)[1] == target).sum().item()

            test_stats['n'] += target.size(0)

        test_time = time.time()

        # --- 计算平均值 ---
        tr_n_acc = train_stats['nat_acc'] / train_stats['n']
        tr_r_acc = train_stats['rob_acc'] / train_stats['n']
        tr_n_loss = train_stats['nat_loss'] / train_stats['n']
        tr_r_loss = train_stats['rob_loss'] / train_stats['n']

        te_n_acc = test_stats['nat_acc'] / test_stats['n']
        te_r_acc = test_stats['rob_acc'] / test_stats['n']
        te_n_loss = test_stats['nat_loss'] / test_stats['n']
        te_r_loss = test_stats['rob_loss'] / test_stats['n']

        # --- 日志打印 ---
        logger.info(
            "{:<6d} | {:<6.1f} | {:<6.1f} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f} || {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f}".format(
                epoch, train_time - start_time, test_time - train_time,
                tr_n_acc, tr_r_acc, tr_n_loss, tr_r_loss,
                te_n_acc, te_r_acc, te_n_loss, te_r_loss)
        )

        # --- WandB 记录 ---
        wandb.log({
            "epoch": epoch,
            "lr": lr,
            "Train/Natural_Acc": tr_n_acc,
            "Train/Robust_Acc": tr_r_acc,
            "Train/Natural_Loss": tr_n_loss,
            "Train/Robust_Loss": tr_r_loss,
            "Test/Natural_Acc": te_n_acc,
            "Test/Robust_Acc": te_r_acc,
            "Test/Natural_Loss": te_n_loss,
            "Test/Robust_Loss": te_r_loss
        })

        if te_r_acc > best_test_robust_acc:
            best_test_robust_acc = te_r_acc
            torch.save(model.state_dict(), os.path.join(save_dir, 'model_best.pth'))

        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f'model_{epoch}.pth'))


if __name__ == '__main__':
    main()