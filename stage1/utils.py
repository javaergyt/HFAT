import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# --- 新增 Cutout 类 ---
class Cutout(object):
    """Randomly masks out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

def get_dataloaders(args):
    """
    加载 GTSRB 数据集
    """
    print(f"==> Preparing GTSRB data from {args.data_dir}...")

    # --- 关键修改：训练集增强 ---
    # 1. Resize 到 32x32
    # 2. RandomCrop (保留)
    # 3. 移除了 RandomHorizontalFlip (GTSRB不能翻转！)
    # 4. 加入了 Cutout (增强鲁棒性，防止过拟合)
    transform_train = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),  # <--- 已删除：防止语义反转
        transforms.ToTensor(),                # 输出范围 [0, 1]
        Cutout(n_holes=1, length=10)          # <--- 新增：挖去 10x10 的孔
    ])

    # 测试集：Resize -> ToTensor
    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    # 自动下载并加载 GTSRB
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


# [保留原有的 import, Cutout, get_dataloaders, clamp, mixup 等函数...]
# 请在文件头部添加: import torchattacks

# ... (保留上面的代码) ...

def evaluate_comprehensive(model, test_loader, logger, epoch, args):
    """
    使用 torchattacks 进行全面的鲁棒性评估
    包含: Clean, PGD-20, PGD-100, MIM, CW, AutoAttack
    """
    import torchattacks

    model.eval()
    logger.info(f"==> Start Comprehensive Evaluation at Epoch {epoch}...")

    # 定义攻击方法列表
    attacks = {
        'Clean': None,
        'PGD-20': torchattacks.PGD(model, eps=8 / 255, alpha=2 / 255, steps=20, random_start=True),
        'PGD-100': torchattacks.PGD(model, eps=8 / 255, alpha=2 / 255, steps=100, random_start=True),
        'MIM': torchattacks.MIFGSM(model, eps=8 / 255, alpha=2 / 255, steps=20, decay=1.0),
        # CW 通常比较慢，steps设小一点或者仅在关键节点跑
        'CW': torchattacks.CW(model, c=1, kappa=0, steps=50, lr=0.01),
        # AutoAttack 是当前最强的一组攻击，非常耗时，建议设为可选或仅在最后跑
        'AA': torchattacks.AutoAttack(model, norm='Linf', eps=8 / 255, version='standard', verbose=False)
    }

    results = {}

    # 为了节省时间，我们可以只从 test_loader 中取一部分数据进行测试 (例如前 1000 张)
    # 如果显卡足够强，可以跑全量
    max_samples = 1000

    for name, attacker in attacks.items():
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(test_loader):
            if total >= max_samples:
                break

            images, labels = images.to(device), labels.to(device)

            if name == 'Clean':
                adv_images = images
            else:
                # 生成对抗样本
                adv_images = attacker(images, labels)

            with torch.no_grad():
                outputs = model(adv_images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        acc = 100. * correct / total
        results[name] = acc
        logger.info(f"    {name:<10}: {acc:.2f}%")

    logger.info("==> Evaluation Complete.")
    return results