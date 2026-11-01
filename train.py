from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.data import build_datasets, make_test_loader, make_train_loader
from src.models import build_model, count_trainable_parameters
from src.repro import environment_metadata, seed_everything, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CIFAR-10 PlainNet/ResNet experiment")
    parser.add_argument(
        "--variant", choices=["plain20", "resnet20", "resnet20_cutout"], required=True
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-root", default="outputs/full")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--cutout-length", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-test", type=int)
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size
    return total_loss / total_samples, 100.0 * total_correct / total_samples


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    all_logits = []
    all_labels = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()
        total_samples += batch_size
        labels_np = labels.cpu().numpy()
        predictions_np = predictions.cpu().numpy()
        np.add.at(confusion, (labels_np, predictions_np), 1)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels_np)
    return (
        total_loss / total_samples,
        100.0 * total_correct / total_samples,
        confusion,
        np.concatenate(all_logits),
        np.concatenate(all_labels),
    )


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available; inspect the environment first")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch size must be positive")

    seed_everything(args.seed, args.deterministic)
    device = torch.device(args.device)
    run_dir = Path(args.output_root) / f"{args.variant}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, test_dataset, dataset_info = build_datasets(
        data_root=args.data_root,
        use_cutout=args.variant == "resnet20_cutout",
        cutout_length=args.cutout_length,
        download=args.download,
        limit_train=args.limit_train,
        limit_test=args.limit_test,
    )
    test_loader = make_test_loader(test_dataset, args.batch_size, args.num_workers)
    model = build_model(args.variant).to(device)
    parameter_count = count_trainable_parameters(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

    metadata = environment_metadata()
    metadata.update(
        {
            "variant": args.variant,
            "seed": args.seed,
            "device": str(device),
            "train_size": dataset_info.train_size,
            "test_size": dataset_info.test_size,
            "classes": dataset_info.classes,
            "parameter_count": parameter_count,
            "hyperparameters": vars(args),
            "test_policy": "The test set is evaluated once after the final training epoch.",
        }
    )
    write_json(run_dir / "metadata.json", metadata)

    log_path = run_dir / "train_log.csv"
    start_time = time.perf_counter()
    with log_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "learning_rate", "train_loss", "train_accuracy", "epoch_seconds"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.perf_counter()
            train_loader = make_train_loader(
                train_dataset, args.batch_size, args.num_workers, args.seed, epoch
            )
            train_loss, train_accuracy = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            epoch_seconds = time.perf_counter() - epoch_start
            learning_rate = optimizer.param_groups[0]["lr"]
            writer.writerow(
                {
                    "epoch": epoch,
                    "learning_rate": f"{learning_rate:.10g}",
                    "train_loss": f"{train_loss:.10f}",
                    "train_accuracy": f"{train_accuracy:.8f}",
                    "epoch_seconds": f"{epoch_seconds:.4f}",
                }
            )
            file.flush()
            print(
                f"variant={args.variant} seed={args.seed} epoch={epoch:03d}/{args.epochs} "
                f"lr={learning_rate:.5g} loss={train_loss:.4f} acc={train_accuracy:.2f}% "
                f"seconds={epoch_seconds:.1f}",
                flush=True,
            )
            scheduler.step()

    test_loss, test_accuracy, confusion, logits, labels = evaluate(
        model, test_loader, criterion, device, num_classes=10
    )
    total_seconds = time.perf_counter() - start_time
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "variant": args.variant,
        "seed": args.seed,
        "epochs": args.epochs,
        "parameter_count": parameter_count,
        "classes": dataset_info.classes,
    }
    torch.save(checkpoint, run_dir / "final_model.pt")
    np.savez_compressed(run_dir / "test_predictions.npz", logits=logits, labels=labels)

    per_class_accuracy = {}
    for index, class_name in enumerate(dataset_info.classes):
        denominator = int(confusion[index].sum())
        per_class_accuracy[class_name] = (
            100.0 * float(confusion[index, index]) / denominator if denominator else None
        )
    summary = {
        "variant": args.variant,
        "seed": args.seed,
        "epochs": args.epochs,
        "parameter_count": parameter_count,
        "train_size": dataset_info.train_size,
        "test_size": dataset_info.test_size,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_error": 100.0 - test_accuracy,
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": confusion.tolist(),
        "total_training_and_test_seconds": total_seconds,
    }
    write_json(run_dir / "summary.json", summary)
    print(
        f"FINAL variant={args.variant} seed={args.seed} test_loss={test_loss:.6f} "
        f"test_accuracy={test_accuracy:.4f}% total_seconds={total_seconds:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
