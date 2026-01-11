import torch
import torch.nn as nn
from torchvision.models import resnet18


class GTSRB_ResNet18(nn.Module):
    def __init__(self, num_classes=43):
        super(GTSRB_ResNet18, self).__init__()
        # 加载标准 ResNet18，不使用预训练权重
        self.model = resnet18(weights=None)

        # --- 适配 32x32 输入 (关键优化) ---
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()  # 移除MaxPool避免特征图过小

        # --- ✅ 关键修改：添加Dropout + 调整全连接层 ---
        # 原版：self.model.fc = nn.Linear(512, num_classes)
        self.model.fc = nn.Sequential(
            # nn.Dropout(0.5),  # ✅ 抗过拟合核心！
            nn.Linear(512, num_classes)
        )

        # --- 内置归一化参数 (保持不变) ---
        self.register_buffer('mu', torch.tensor([0.3337, 0.3064, 0.3171]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.2672, 0.2564, 0.2629]).view(1, 3, 1, 1))

    def forward(self, x):
        # 输入 x 在 [0, 1] 范围内
        x = (x - self.mu) / self.std
        return self.model(x)