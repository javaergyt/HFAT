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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch AT-PGD Adversarial Training for GTSRB')

    # 基础参数
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--batch_size_test', default=128, type=int)
    parser.add_argument('--data_dir', default='./datasets', type=str)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--model', default='ResNet18')

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

    # 杂项
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--fname', default='result/AT_PGD_Baseline', type=str)
    parser.add_argument('--proj_name', type=str, default='GTSRB_Compare')
    parser.add_argument('--name', type=str, default='AT_PGD_Baseline')
    parser.add_argument('--wd_offline', default=1, type=int)

    return parser.parse_args()


def main():
    args = get_args()
    # 归一化参数
    args.epsilon_float = args.epsilon / 255.0
    args.step_size_float = args.step_size / 255.0

    if args.wd_offline:
        os.environ["WANDB_MODE"] = "offline"
    wandb.init(project=args.proj_name, name=args.name, config=args)

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

    # 加载数据
    train_loader, test_loader = get_dataloaders(args)

    logger.info("Initialize GTSRB_ResNet18 for AT-PGD...")
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

        # --- 训练阶段 ---
        model.train()  # 确保模型处于训练模式 (Batch Norm 更新)

        train_stats = {
            'nat_loss': 0, 'nat_acc': 0,
            'rob_loss': 0, 'rob_acc': 0,
            'n': 0
        }

        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # 更新学习率
            now_epoch = (epoch - 1) + (i + 1) / len(train_loader)
            lr = lr_schedule(now_epoch)
            optimizer.param_groups[0].update(lr=lr)

            # 1. 生成对抗样本
            # 通常在 Eval 模式下生成，避免 BN 统计量被对抗样本污染，保持 BN 学习的是自然分布特征
            model.eval()

            # 使用 utils.py 中的 attack_pgd (目标是最大化 CrossEntropy)
            delta = attack_pgd(model, data, target,
                               epsilon=args.epsilon_float,
                               alpha=args.step_size_float,
                               attack_iters=args.num_steps,
                               restarts=1,
                               norm=args.norm)

            model.train()  # 恢复训练模式进行参数更新

            # 2. 对抗训练更新
            # AT-PGD 只优化对抗样本上的 Loss (Madry Original)
            # 也可以是 Mix (Clean + Adv)，但最经典的 AT-PGD 指的是仅在 Adv 上训练
            x_adv = torch.clamp(data + delta, 0, 1)

            optimizer.zero_grad()
            logits_adv = model(x_adv)
            loss = F.cross_entropy(logits_adv, target)

            loss.backward()
            optimizer.step()

            # --- 统计指标 ---
            with torch.no_grad():
                # 计算自然样本表现 (仅用于监控)
                logits_nat = model(data)
                nat_acc = (logits_nat.max(1)[1] == target).float().sum().item()
                nat_loss = F.cross_entropy(logits_nat, target, reduction='sum').item()

                # 记录鲁棒指标 (训练时的对抗样本)
                rob_acc = (logits_adv.max(1)[1] == target).float().sum().item()
                rob_loss = loss.item() * target.size(0)

                train_stats['nat_loss'] += nat_loss
                train_stats['nat_acc'] += nat_acc
                train_stats['rob_loss'] += rob_loss
                train_stats['rob_acc'] += rob_acc
                train_stats['n'] += target.size(0)

        train_time = time.time()

        # --- 测试阶段 (与 TRADES/MART 保持一致) ---
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
            delta = attack_pgd(model, data, target,
                               epsilon=args.epsilon_float,
                               alpha=args.step_size_float,
                               attack_iters=args.attack_iters_test,
                               restarts=args.restarts,
                               norm=args.norm)

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