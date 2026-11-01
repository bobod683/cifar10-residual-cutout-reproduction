from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class Cutout:
    """Mask one fixed-size square in a normalized tensor image."""

    def __init__(self, length: int = 16) -> None:
        if length <= 0:
            raise ValueError("Cutout length must be positive")
        self.length = length

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 3:
            raise ValueError("Cutout expects a CHW tensor")
        _, height, width = image.shape
        center_y = int(torch.randint(0, height, (1,)).item())
        center_x = int(torch.randint(0, width, (1,)).item())
        half = self.length // 2
        y1 = max(center_y - half, 0)
        y2 = min(center_y + self.length - half, height)
        x1 = max(center_x - half, 0)
        x2 = min(center_x + self.length - half, width)
        result = image.clone()
        result[:, y1:y2, x1:x2] = 0.0
        return result


@dataclass(frozen=True)
class DatasetInfo:
    train_size: int
    test_size: int
    classes: tuple[str, ...]


def build_datasets(
    data_root: str | Path,
    use_cutout: bool,
    cutout_length: int,
    download: bool,
    limit_train: Optional[int] = None,
    limit_test: Optional[int] = None,
) -> tuple[Dataset, Dataset, DatasetInfo]:
    train_transforms: list[object] = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]
    if use_cutout:
        train_transforms.append(Cutout(cutout_length))

    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    )
    root = str(Path(data_root).expanduser().resolve())
    train_dataset: Dataset = datasets.CIFAR10(
        root=root, train=True, transform=transforms.Compose(train_transforms), download=download
    )
    test_dataset: Dataset = datasets.CIFAR10(
        root=root, train=False, transform=test_transform, download=download
    )
    class_names = tuple(getattr(train_dataset, "classes"))

    if limit_train is not None:
        train_dataset = Subset(train_dataset, range(min(limit_train, len(train_dataset))))
    if limit_test is not None:
        test_dataset = Subset(test_dataset, range(min(limit_test, len(test_dataset))))

    info = DatasetInfo(len(train_dataset), len(test_dataset), class_names)
    return train_dataset, test_dataset, info


def make_train_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    seed: int,
    epoch: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed + epoch)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        generator=generator,
    )


def make_test_loader(dataset: Dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )
