from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OptionAShortcut(nn.Module):
    """Parameter-free CIFAR shortcut from the original ResNet paper."""

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride != 1:
            x = x[:, :, :: self.stride, :: self.stride]
        channel_delta = self.out_channels - self.in_channels
        if channel_delta > 0:
            left = channel_delta // 2
            right = channel_delta - left
            x = F.pad(x, (0, 0, 0, 0, left, right))
        return x


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else OptionAShortcut(in_channels, out_channels, stride)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity, inplace=True)


class PlainBlock(nn.Module):
    """Shape-matched block without a residual shortcut."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        return F.relu(self.bn2(self.conv2(out)), inplace=True)


class CifarNet20(nn.Module):
    def __init__(self, block: type[nn.Module], num_classes: int = 10) -> None:
        super().__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, blocks=3, stride=1)
        self.layer2 = self._make_layer(block, 32, blocks=3, stride=2)
        self.layer3 = self._make_layer(block, 64, blocks=3, stride=2)
        self.fc = nn.Linear(64, num_classes)
        self._initialize_weights()

    def _make_layer(
        self, block: type[nn.Module], out_channels: int, blocks: int, stride: int
    ) -> nn.Sequential:
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for block_stride in strides:
            layers.append(block(self.in_channels, out_channels, block_stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = torch.flatten(x, 1)
        return self.fc(x)


def build_model(name: str, num_classes: int = 10) -> nn.Module:
    if name == "plain20":
        return CifarNet20(PlainBlock, num_classes=num_classes)
    if name in {"resnet20", "resnet20_cutout"}:
        return CifarNet20(ResidualBlock, num_classes=num_classes)
    raise ValueError(f"Unknown model variant: {name}")


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
