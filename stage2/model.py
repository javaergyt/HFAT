# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import resnet18, resnet34, resnet50, densenet121, vgg19

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import random

# 尝试导入，如果失败则忽略，依靠下文的本地定义
try:
    from models import *
except ImportError:
    pass


# ==========================================
#  Stage 1 模型定义 (本地嵌入版)
# ==========================================

# --- 1. GTSRB ResNet18 ---
class GTSRB_ResNet18(nn.Module):
    def __init__(self, num_classes=43):
        super(GTSRB_ResNet18, self).__init__()
        # 加载标准 ResNet18，不使用预训练权重
        self.model = resnet18(weights=None)

        # --- 适配 32x32 输入 (关键优化) ---
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()  # 移除MaxPool避免特征图过小

        # --- 保持与 Stage 1 一致的 FC 定义 ---
        self.model.fc = nn.Sequential(
            # nn.Dropout(0.5), # 保持注释状态，与stage1文件一致
            nn.Linear(512, num_classes)
        )

        # --- 内置归一化参数 ---
        self.register_buffer('mu', torch.tensor([0.3337, 0.3064, 0.3171]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.2672, 0.2564, 0.2629]).view(1, 3, 1, 1))

    def forward(self, x):
        # 输入 x 在 [0, 1] 范围内
        x = (x - self.mu) / self.std
        return self.model(x)


# --- 2. GTSRB WideResNet (适配 Stage 1) ---

class BasicBlockWRN(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropout=0.0):
        super(BasicBlockWRN, self).__init__()
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


class NetworkBlockWRN(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropout=0.0):
        super(NetworkBlockWRN, self).__init__()
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
        n_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        n = (depth - 4) / 6
        block = BasicBlockWRN
        self.conv1 = nn.Conv2d(3, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlockWRN(n, n_channels[0], n_channels[1], block, 1, dropout)
        self.block2 = NetworkBlockWRN(n, n_channels[1], n_channels[2], block, 2, dropout)
        self.block3 = NetworkBlockWRN(n, n_channels[2], n_channels[3], block, 2, dropout)
        self.bn1 = nn.BatchNorm2d(n_channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(n_channels[3], num_classes)

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


class GTSRB_WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, num_classes=43, dropout=0.3):
        super(GTSRB_WideResNet, self).__init__()
        # 默认使用 WideResNet-28-10, dropout=0.3
        self.model = WideResNet(depth=depth, widen_factor=widen_factor, num_classes=num_classes, dropout=dropout)
        self.register_buffer('mu', torch.tensor([0.3337, 0.3064, 0.3171]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.2672, 0.2564, 0.2629]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.std
        return self.model(x)


# ========================================================

def create_model(model_name, input_size, num_classes, device, patch_size=4, resume=None):
    model = None

    # === GTSRB Models (Stage 1 Compatible) ===
    if model_name == "ResNet18" and num_classes == 43:
        print("==> Creating GTSRB_ResNet18 (Stage 1 Model)...")
        model = GTSRB_ResNet18(num_classes=43)

    # 只要名字包含 WideResNet，就默认使用 WRN-28-10 (Stage 1 默认配置)
    elif "WideResNet" in model_name and num_classes == 43:
        print(f"==> Creating GTSRB_WideResNet (Stage 1 Model) for {model_name}...")
        # 注意：这里默认使用 depth=28, widen_factor=10。如果你在 Stage 1 用了 WRN-34-10，需要在这里手动修改 depth=34
        model = GTSRB_WideResNet(depth=28, widen_factor=10, num_classes=43, dropout=0.3)

    # === Other Models ===
    elif model_name == "ResNet34":
        model = resnet34(num_classes=num_classes)
        # 适配 CIFAR/TinyImageNet 输入
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    elif model_name == "ResNet18":
        model = resnet18(num_classes=num_classes)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    elif model_name == "ResNet50":
        model = resnet50(num_classes=num_classes)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    elif model_name == "DenseNet121":
        model = densenet121(num_classes=num_classes)
    elif model_name == "VGG19":
        # 简化处理，使用 torchvision 或自定义 VGG
        pass

    if model is None:
        raise ValueError(f"Model {model_name} not supported or implemented in this file.")

    model = model.to(device)

    if device == 'cuda':
        model = torch.nn.DataParallel(model)

    if resume is not None:
        if os.path.isfile(resume):
            print(f"==> Loading checkpoint from {resume}")
            checkpoint = torch.load(resume, map_location=device)

            state_dict = checkpoint
            if isinstance(checkpoint, dict):
                if "net" in checkpoint.keys():
                    state_dict = checkpoint["net"]
                elif "state_dict" in checkpoint.keys():
                    state_dict = checkpoint["state_dict"]
                elif "model" in checkpoint.keys():
                    state_dict = checkpoint["model"]

            # 处理 DataParallel 带来的 'module.' 前缀不匹配问题
            try:
                model.load_state_dict(state_dict)
            except RuntimeError as e:
                print("Strict loading failed, trying to ignore 'module.' prefix or missing keys...")
                new_state_dict = {}
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith('module.') else k
                    new_state_dict[name] = v

                # 如果当前模型被包了 DataParallel (Stage 2 默认行为)，而 checkpoint 没有 module.
                if isinstance(model, torch.nn.DataParallel):
                    # 尝试给 checkpoint 加 module.
                    new_state_dict_dp = {f'module.{k}': v for k, v in new_state_dict.items()}
                    try:
                        model.load_state_dict(new_state_dict_dp, strict=False)
                    except:
                        model.load_state_dict(new_state_dict, strict=False)
                else:
                    model.load_state_dict(new_state_dict, strict=False)
        else:
            print(f"==> Error: Checkpoint file {resume} not found!")

    return model