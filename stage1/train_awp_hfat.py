# train_hfat_awp.py
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

# 引入自定义工具和模型
from utils import get_dataloaders, attack_pgd, mixup_data, mixup_criterion
# 引入模型
from models.resnet18_gtsrb import GTSRB_ResNet18
from models.wideresnet_gtsrb import gtsrb_wideresnet_28_10, gtsrb_wideresnet_34_10, gtsrb_wideresnet_28_20

# 引入攻击库用于综合评估
try:
    import torchattacks

    HAS_TORCHATTACKS = True
except ImportError:
    HAS_TORCHATTACKS = False
    print("Warning: torchattacks not found. Comprehensive evaluation will be skipped.")

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

upper_limit, lower_limit = 1, 0
EPS = 1E-20


# ================= HFAT 原有的辅助函数 (保持不变) =================
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


# ================= 新增：标准 AWP 实现类 =================
class AWP_Fast:
    """
    针对标准对抗训练分支的 AWP 实现
    与 HFAT 的 Proxy 逻辑分离，确保控制变量
    """

    def __init__(self, model, optimizer, gamma, awp_warmup):
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma
        self.awp_warmup = awp_warmup

    def calc_awp(self, inputs, targets, criterion):
        """计算扰动"""
        self.model.zero_grad()
        outputs = self.model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        perturbation = OrderedDict()
        for name, param in self.model.named_parameters():
            if param.grad is not None and len(param.shape) > 1:  # 仅扰动权重层
                grad = param.grad
                norm = grad.norm()
                if norm > EPS:
                    # 方向：梯度上升方向 (Maximize Loss)
                    perturbation[name] = self.gamma * param.data.norm() * (grad / norm)
                else:
                    perturbation[name] = torch.zeros_like(param)

        # 清除梯度，以免影响后续步骤
        self.model.zero_grad()
        return perturbation

    def perturb(self, perturbation):
        if perturbation is None: return
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in perturbation:
                    param.add_(perturbation[name])

    def restore(self, perturbation):
        if perturbation is None: return
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in perturbation:
                    param.sub_(perturbation[name])


# ================= 综合评估函数 (Table 1 复现) =================
def evaluate_comprehensive(model, test_loader, logger, epoch):
    if not HAS_TORCHATTACKS:
        return {}

    logger.info(f"\n[Epoch {epoch}] Starting Comprehensive Evaluation (Fig 3 Metrics)...")
    model.eval()

    # 定义攻击列表 (参考论文 Table 1 / Fig 3)
    attacks = {
        'PGD-100': torchattacks.PGD(model, eps=8 / 255, alpha=2 / 255, steps=100),
        'MIM': torchattacks.MIFGSM(model, eps=8 / 255, alpha=2 / 255, steps=20),
        'CW': torchattacks.CW(model, c=1, kappa=0, steps=50, lr=0.01),
        'AA': torchattacks.AutoAttack(model, norm='Linf', eps=8 / 255, version='standard', verbose=False)
    }

    results = {}
    max_samples = 1000  # 限制样本数以加快训练过程

    for name, attacker in attacks.items():
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(test_loader):
            if total >= max_samples:
                break

            images, labels = images.to(device), labels.to(device)
            adv_images = attacker(images, labels)

            with torch.no_grad():
                outputs = model(adv_images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        acc = 100. * correct / total
        results[name] = acc
        logger.info(f"  -> {name:<8}: {acc:.2f}%")

        wandb.log({f"Eval/{name}": acc, "epoch": epoch}, commit=False)

    return results


def get_args():
    parser = argparse.ArgumentParser()
    # === 模型选择 ===
    parser.add_argument('--model', default='ResNet18',
                        choices=['ResNet18', 'WideResNet_28_10', 'WideResNet_34_10'],
                        help='Model architecture to use')

    # === 原有参数 ===
    parser.add_argument('--l2', default=0, type=float)
    parser.add_argument('--l1', default=0, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--batch_size_test', default=128, type=int)
    parser.add_argument('--data_dir', default='./datasets', type=str)

    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr_schedule', default='piecewise')
    parser.add_argument('--lr_max', default=0.1, type=float)
    parser.add_argument('--lr_one_drop', default=0.01, type=float)
    parser.add_argument('--lr_drop_epoch', default=100, type=int)
    parser.add_argument('--lr_proxy_max', default=0.01, type=float)

    parser.add_argument('--attack', default='pgd', type=str, choices=['pgd', 'none'])
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--attack_iters', default=10, type=int)
    parser.add_argument('--attack_iters_test', default=20, type=int)
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--pgd_alpha', default=2, type=float)
    parser.add_argument('--beta', default=6.0, type=float)
    parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2'])

    parser.add_argument('--fname', default='result/HFAT_AWP', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default=0, type=int)

    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--mixup_alpha', type=float, default=1.0)
    parser.add_argument('--chkpt_iters', default=10, type=int)

    # AWP 参数
    parser.add_argument('--awp_gamma', default=0.01, type=float, help="AWP gamma for Standard Loss")
    parser.add_argument('--awp_warmup', default=0, type=int)

    # log
    parser.add_argument('--proj_name', type=str, default='GTSRB_HFAT_AWP', help='')
    parser.add_argument('--name', type=str, default='ResNet18_HFAT_AWP', help='')
    parser.add_argument('--wd_offline', default=1, type=int)

    # HFAT auxiliary params (保持默认)
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

    if args.wd_offline:
        os.environ["WANDB_MODE"] = "offline"

    wandb.init(project=args.proj_name, name=args.name, config=args)

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
            logging.FileHandler(os.path.join(save_dir, 'output.log')),
            logging.StreamHandler()
        ])

    logger.info(args)

    # 随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # ================= 数据加载 =================
    train_loader, test_loader = get_dataloaders(args)

    epsilon = (args.epsilon / 255.)
    pgd_alpha = (args.pgd_alpha / 255.)
    aux_epsilon = (args.aux_epsilon / 255.)
    aux_pgd_alpha = (args.aux_pgd_alpha / 255.)

    # ================= 模型初始化 =================
    def create_model(model_name):
        if model_name == 'ResNet18':
            return GTSRB_ResNet18(num_classes=43)
        elif model_name == 'WideResNet_28_10':
            return gtsrb_wideresnet_28_10(num_classes=43)
        elif model_name == 'WideResNet_34_10':
            return gtsrb_wideresnet_34_10(num_classes=43)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    # HFAT 需要两个模型：Standard Model 和 Auxiliary Model (Proxy)
    model = create_model(args.model)
    proxy = create_model(args.model)  # 这里的 Proxy 是 HFAT 的辅助模型

    model = nn.DataParallel(model).to(device)
    proxy = nn.DataParallel(proxy).to(device)

    # 优化器
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

    criterion = nn.CrossEntropyLoss()

    # 初始化 AWP 管理器 (用于标准损失分支)
    awp_adversary = AWP_Fast(model, opt, gamma=args.awp_gamma, awp_warmup=args.awp_warmup)

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

    logger.info('Start Training: HFAT + AWP with Comprehensive Eval')

    # ================= 训练循环 =================
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()

        train_stats = {'nat_loss': 0, 'nat_acc': 0, 'rob_loss': 0, 'rob_acc': 0, 'n': 0}
        model.train()

        for i, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)

            lr = lr_schedule(epoch + (i + 1) / len(train_loader))
            opt.param_groups[0].update(lr=lr)

            # -----------------------------------------------------------------
            # 1. HFAT 核心部分：辅助模型 (Proxy) 更新 (保持不变)
            # -----------------------------------------------------------------
            if args.attack == 'pgd':
                # 使用辅助模型的参数 epsilon 生成对抗样本
                delta = attack_pgd(model, X, y, aux_epsilon, aux_pgd_alpha,
                                   args.aux_attack_iters, args.restarts, args.norm)
                delta = delta.detach()
            else:
                delta = torch.zeros_like(X)

            X_adv_raw = torch.clamp(X + delta, min=lower_limit, max=upper_limit)
            diff = X - X_adv_raw

            # 生成带噪声的对抗样本 (用于训练辅助模型发现 Hiders)
            eps_beta = np.random.normal(args.mean, args.std_dev)
            X_adv_eps = torch.clamp(X_adv_raw + eps_beta * diff + args.eps_gamma * torch.randn_like(X).to(device),
                                    min=lower_limit, max=upper_limit)

            # Proxy 更新
            if isinstance(proxy, nn.DataParallel):
                proxy.module.load_state_dict(model.module.state_dict())
            else:
                proxy.load_state_dict(model.state_dict())

            proxy.train()
            loss_proxy = nn.CrossEntropyLoss(reduction='none')(proxy(X_adv_eps), y)
            Indicator = (loss_proxy < args.lt).float()
            loss_proxy = -1 * (loss_proxy.mul(Indicator).mean())  # 辅助模型试图最大化 Loss (逆向训练)
            proxy_opt.zero_grad()
            loss_proxy.backward()
            proxy_opt.step()

            # 计算 HFAT 的权重扰动方向 (从 Proxy 到 Model 的方向)
            diff_weights = diff_in_weights(model, proxy)

            # -----------------------------------------------------------------
            # 2. HFAT 核心部分：Hider Loss (保持不变)
            # -----------------------------------------------------------------
            # 预先注入 HFAT 扰动 (Simulating future weights)
            if epoch >= args.awp_warmup:
                # 注意：这里使用的是 diff_weights，这是 HFAT 专有的，不是 AWP
                # main.py 原名为 awp_gamma，实为 HFAT 的权重系数
                add_into_weights(model, diff_weights, coeff=1.0 * args.awp_gamma)

            # 计算 Hider Loss
            u_delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm)
            X_u_adv = torch.clamp(X + u_delta, min=lower_limit, max=upper_limit)

            u_robust_output = model(X_u_adv)
            loss_wv = criterion(u_robust_output, y)

            opt.zero_grad()
            loss_wv.backward()  # 获取 Hider 梯度

            # 恢复权重
            if epoch >= args.awp_warmup:
                add_into_weights(model, diff_weights, coeff=-1.0 * args.awp_gamma)

            # 保存 Hider 梯度
            wv_gradient_dict = OrderedDict()
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        wv_gradient_dict[name] = param.grad.clone()

            # -----------------------------------------------------------------
            # 3. 标准分支 (Standard Robust Branch) - 修改为 AWP 实现
            # -----------------------------------------------------------------
            # 3.1 生成标准对抗样本
            delta_main = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm)
            X_adv = torch.clamp(X + delta_main, min=lower_limit, max=upper_limit)

            # 3.2 [新] 计算 AWP 扰动 (针对 Standard Loss)
            # 此时的 loss_w 目标是 PGD-AT Loss，我们用 AWP 来优化它
            awp_perturbation = None
            if epoch >= args.awp_warmup:
                # 计算针对 loss_w 的最大化扰动
                awp_perturbation = awp_adversary.calc_awp(X_adv, y, criterion)
                # 注入 AWP 扰动
                awp_adversary.perturb(awp_perturbation)

            # 3.3 计算标准 Loss (在 AWP 扰动后的权重上)
            robust_output = model(X_adv)
            loss_w = criterion(robust_output, y)

            opt.zero_grad()
            loss_w.backward()  # 获取 Standard 梯度 (基于 AWP 扰动后的 Landscape)

            # 3.4 恢复 AWP 扰动
            if epoch >= args.awp_warmup:
                awp_adversary.restore(awp_perturbation)

            # -----------------------------------------------------------------
            # 4. HFAT 梯度融合 (Adaptive Weighting) (保持不变)
            # -----------------------------------------------------------------
            with torch.no_grad():
                robust_n_output = model(X)  # Clean output
                u_robust_n_output = model(X)  # 这里再算一次是为了对齐逻辑，其实是一样的

                # 计算 KL 散度来动态调整 Hider分支 和 Standard分支 的权重
                kl_robust = F.kl_div(F.log_softmax(robust_output, dim=1),
                                     F.softmax(robust_n_output, dim=1),
                                     reduction='sum')
                # 注意：这里 u_robust_output 是之前 Hider 分支计算的
                kl_u_robust = F.kl_div(F.log_softmax(u_robust_output, dim=1),
                                       F.softmax(u_robust_n_output, dim=1),
                                       reduction='sum')

                w_tensor = torch.stack([kl_robust, kl_u_robust])
                w_tensor[0] = w_tensor[0] * args.beta_r
                w_tensor[1] = w_tensor[1] * args.beta_u
                w_softmax = F.softmax(w_tensor, dim=0)
                w_r, w_u = w_softmax[0], w_softmax[1]

            # 混合梯度： Standard_Grad (param.grad) + Hider_Grad (wv_gradient_dict)
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        g_hider = wv_gradient_dict[name]
                        # 融合
                        param.grad = w_r * param.grad + w_u * g_hider

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            # --- 统计 ---
            with torch.no_grad():
                loss_nat = criterion(robust_n_output, y)
                train_stats['nat_loss'] += loss_nat.item() * y.size(0)
                train_stats['nat_acc'] += (robust_n_output.max(1)[1] == y).sum().item()
                train_stats['rob_loss'] += loss_w.item() * y.size(0)
                train_stats['rob_acc'] += (robust_output.max(1)[1] == y).sum().item()
                train_stats['n'] += y.size(0)

        train_time = time.time()

        # ================= 测试循环 (Standard PGD-20) =================
        model.eval()
        test_stats = {'nat_acc': 0, 'rob_acc': 0, 'n': 0}

        for i, (X, y) in enumerate(test_loader):
            X, y = X.to(device), y.to(device)
            # Clean
            out = model(X)
            test_stats['nat_acc'] += (out.max(1)[1] == y).sum().item()
            # PGD-20
            delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters_test, args.restarts, args.norm)
            out_rob = model(torch.clamp(X + delta, 0, 1))
            test_stats['rob_acc'] += (out_rob.max(1)[1] == y).sum().item()
            test_stats['n'] += y.size(0)

        test_time = time.time()

        avg_trn_nat = train_stats['nat_acc'] / train_stats['n']
        avg_trn_rob = train_stats['rob_acc'] / train_stats['n']
        avg_tst_nat = test_stats['nat_acc'] / test_stats['n']
        avg_tst_rob = test_stats['rob_acc'] / test_stats['n']

        logger.info(
            f"{epoch} \t {train_time - start_time:.1f} \t {test_time - train_time:.1f} \t {lr:.4f} \t "
            f"Tr_Nat: {avg_trn_nat:.4f} \t Tr_Rob: {avg_trn_rob:.4f} \t "
            f"Te_Nat: {avg_tst_nat:.4f} \t Te_Rob: {avg_tst_rob:.4f}"
        )

        wandb_log = {
            "epoch": epoch,
            "train_acc": avg_trn_nat,
            "train_robust_acc": avg_trn_rob,
            "test_acc": avg_tst_nat,
            "test_robust_acc": avg_tst_rob,
            "lr": lr
        }

        # ================= 额外评估 (Every 5 epochs) =================
        if (epoch + 1) % 5 == 0:
            comprehensive_results = evaluate_comprehensive(model, test_loader, logger, epoch)
            # wandb log 已在函数内处理

        wandb.log(wandb_log)

        if avg_tst_rob > best_test_robust_acc and epoch > 0:
            best_test_robust_acc = avg_tst_rob
            torch.save(model.state_dict(), os.path.join(save_dir, 'model_best.pth'))

        if (epoch + 1) % args.chkpt_iters == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f'model_{epoch}.pth'))


if __name__ == "__main__":
    main()