import torch
import torch.nn as nn
from torchvision.models import resnet18


class GTSRB_ResNet18(nn.Module):
    def __init__(self, num_classes=43):
        super(GTSRB_ResNet18, self).__init__()
        # 加载标准 ResNet18，不使用预训练权重 (对抗训练通常从头开始)
        self.model = resnet18(weights=None)

        # --- 关键修改：适配 32x32 输入 ---
        # 原版 7x7, stride=2 会导致信息过早丢失
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # 移除 MaxPool，避免特征图过小
        self.model.maxpool = nn.Identity()

        # 修改全连接层
        self.model.fc = nn.Linear(512, num_classes)

        # --- 内置归一化参数 ---
        self.register_buffer('mu', torch.tensor([0.3337, 0.3064, 0.3171]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.2672, 0.2564, 0.2629]).view(1, 3, 1, 1))

    def forward(self, x):
        # 输入 x 必须在 [0, 1] 范围内
        x = (x - self.mu) / self.std
        return self.model(x)