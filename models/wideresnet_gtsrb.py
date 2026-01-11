import torch
import torch.nn as nn
import math


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropout=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(self.relu1(self.bn1(x))))
        out = self.conv2(self.relu2(self.bn2(out)))
        out += self.shortcut(x)
        return out


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropout=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropout)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropout):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, num_classes=10, dropout=0.0):
        super(WideResNet, self).__init__()
        n_channels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock

        # 初始卷积层 - 针对32x32输入优化
        self.conv1 = nn.Conv2d(3, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False)

        # 残差块
        self.block1 = NetworkBlock(n, n_channels[0], n_channels[1], block, 1, dropout)
        self.block2 = NetworkBlock(n, n_channels[1], n_channels[2], block, 2, dropout)
        self.block3 = NetworkBlock(n, n_channels[2], n_channels[3], block, 2, dropout)

        # 最终层
        self.bn1 = nn.BatchNorm2d(n_channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(n_channels[3], num_classes)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class GTSRB_WideResNet(nn.Module):
    """
    WideResNet for GTSRB dataset (32x32 images, 43 classes)
    Compatible interface with GTSRB_ResNet18
    """
    def __init__(self, depth=28, widen_factor=10, num_classes=43, dropout=0.3):
        super(GTSRB_WideResNet, self).__init__()

        # WideResNet 主体
        self.model = WideResNet(depth=depth, widen_factor=widen_factor,
                               num_classes=num_classes, dropout=dropout)

        # --- 内置归一化参数 (与ResNet18保持一致) ---
        self.register_buffer('mu', torch.tensor([0.3337, 0.3064, 0.3171]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.2672, 0.2564, 0.2629]).view(1, 3, 1, 1))

    def forward(self, x):
        # 输入 x 在 [0, 1] 范围内
        x = (x - self.mu) / self.std
        return self.model(x)


# 为了修复 forward 方法中的 F.adaptive_avg_pool2d
import torch.nn.functional as F

# 重新定义 WideResNet 类来修复这个问题
class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, num_classes=10, dropout=0.0):
        super(WideResNet, self).__init__()
        n_channels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock

        # 初始卷积层 - 针对32x32输入优化
        self.conv1 = nn.Conv2d(3, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False)

        # 残差块
        self.block1 = NetworkBlock(n, n_channels[0], n_channels[1], block, 1, dropout)
        self.block2 = NetworkBlock(n, n_channels[1], n_channels[2], block, 2, dropout)
        self.block3 = NetworkBlock(n, n_channels[2], n_channels[3], block, 2, dropout)

        # 最终层
        self.bn1 = nn.BatchNorm2d(n_channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(n_channels[3], num_classes)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# 几种常用的 WideResNet 变体
def gtsrb_wideresnet_28_10(num_classes=43, dropout=0.3):
    """WideResNet-28-10 for GTSRB"""
    return GTSRB_WideResNet(depth=28, widen_factor=10, num_classes=num_classes, dropout=dropout)

def gtsrb_wideresnet_34_10(num_classes=43, dropout=0.3):
    """WideResNet-34-10 for GTSRB"""
    return GTSRB_WideResNet(depth=34, widen_factor=10, num_classes=num_classes, dropout=dropout)

def gtsrb_wideresnet_28_20(num_classes=43, dropout=0.3):
    """WideResNet-28-20 for GTSRB (更宽的网络)"""
    return GTSRB_WideResNet(depth=28, widen_factor=20, num_classes=num_classes, dropout=dropout)


if __name__ == "__main__":
    # 测试模型
    model = gtsrb_wideresnet_28_10()
    print(f"Model: {model.__class__.__name__}")

    # 测试输入
    x = torch.randn(4, 3, 32, 32)  # batch_size=4, GTSRB 32x32 images
    print(f"Input shape: {x.shape}")

    # 前向传播
    with torch.no_grad():
        output = model(x)
        print(f"Output shape: {output.shape}")
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\n✓ WideResNet for GTSRB created successfully!")
    print("✓ Compatible interface with existing ResNet18 model")
    print("✓ Ready for adversarial training framework")