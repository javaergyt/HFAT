# utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataloaders(args):
    """
    使用 torchvision 标准方式加载 GTSRB 数据集
    """
    print(f"==> Preparing GTSRB data from {args.data_dir}...")

    # 训练集增强：Resize -> RandomCrop -> Flip -> ToTensor
    transform_train = transforms.Compose([
        transforms.Resize((32, 32)),  # 确保统一尺寸
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # 输出范围 [0, 1]
    ])

    # 测试集：Resize -> ToTensor
    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),  # 输出范围 [0, 1]
    ])

    # 自动下载并加载 GTSRB
    # 注意：torchvision.datasets.GTSRB 会自动处理文件夹结构和 CSV 读取
    train_set = torchvision.datasets.GTSRB(
        root=args.data_dir, split='train', transform=transform_train, download=True
    )
    test_set = torchvision.datasets.GTSRB(
        root=args.data_dir, split='test', transform=transform_test, download=True
    )

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=False
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size_test, shuffle=False,
        num_workers=4, pin_memory=True
    )

    return train_loader, test_loader


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts,
               norm, early_stop=False,
               mixup=False, y_a=None, y_b=None, lam=None):
    """
    PGD 攻击函数
    注意：输入 X 为 [0,1]，模型内部处理归一化，因此这里不需要显式调用 normalize()
    """
    upper_limit, lower_limit = 1, 0
    max_loss = torch.zeros(y.shape[0]).to(device)
    max_delta = torch.zeros_like(X).to(device)

    for _ in range(restarts):
        delta = torch.zeros_like(X).to(device)
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0), -1)
            n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r / n * epsilon
        else:
            raise ValueError("Norm must be l_inf or l_2")

        delta = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True

        for _ in range(attack_iters):
            # 模型直接接收 [0,1] 的输入
            output = model(X + delta)

            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None, None, None)

            if not isinstance(index, slice) and len(index) == 0:
                break

            if mixup:
                criterion = nn.CrossEntropyLoss()
                loss = mixup_criterion(criterion, model(X + delta), y_a, y_b, lam)
            else:
                loss = F.cross_entropy(output, y)

            loss.backward()
            grad = delta.grad.detach()

            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]

            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
                scaled_g = g / (g_norm + 1e-10)
                d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)

            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()

        if mixup:
            criterion = nn.CrossEntropyLoss(reduction='none')
            all_loss = mixup_criterion(criterion, model(X + delta), y_a, y_b, lam)
        else:
            all_loss = F.cross_entropy(model(X + delta), y, reduction='none')

        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)

    return max_delta