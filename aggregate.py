from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


VARIANT_ORDER = ["plain20", "resnet20", "resnet20_cutout"]
DISPLAY_NAMES = {
    "plain20": "Plain-20",
    "resnet20": "ResNet-20",
    "resnet20_cutout": "ResNet-20 + Cutout",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate completed experiment runs")
    parser.add_argument("--input-root", default="outputs/full")
    parser.add_argument("--output-dir", default="outputs/aggregate")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def exact_sign_flip_pvalue(differences: list[float]) -> float | None:
    if not differences:
        return None
    observed = abs(float(np.mean(differences)))
    means = []
    for signs in itertools.product([-1.0, 1.0], repeat=len(differences)):
        means.append(abs(float(np.mean(np.asarray(differences) * np.asarray(signs)))))
    return sum(value >= observed - 1e-12 for value in means) / len(means)


def load_runs(root: Path) -> list[dict]:
    runs = []
    for summary_path in sorted(root.glob("*/summary.json")):
        summary = read_json(summary_path)
        summary["run_dir"] = str(summary_path.parent.resolve())
        runs.append(summary)
    return runs


def validate_runs(runs: list[dict]) -> None:
    expected = {(variant, seed) for variant in VARIANT_ORDER for seed in (42, 3407, 2026)}
    actual = {(run["variant"], int(run["seed"])) for run in runs}
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise RuntimeError(f"Run set is incomplete or unexpected. missing={missing}, unexpected={unexpected}")
    for run in runs:
        run_dir = Path(run["run_dir"])
        for required in ("metadata.json", "train_log.csv", "final_model.pt", "test_predictions.npz"):
            if not (run_dir / required).is_file():
                raise RuntimeError(f"Missing {required} in {run_dir}")


def plot_training_curves(runs: list[dict], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for variant in VARIANT_ORDER:
        series_loss = []
        series_accuracy = []
        epochs = None
        for run in runs:
            if run["variant"] != variant:
                continue
            with (Path(run["run_dir"]) / "train_log.csv").open(encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            current_epochs = np.asarray([int(row["epoch"]) for row in rows])
            if epochs is not None and not np.array_equal(epochs, current_epochs):
                raise RuntimeError("Epoch grids differ across runs")
            epochs = current_epochs
            series_loss.append([float(row["train_loss"]) for row in rows])
            series_accuracy.append([float(row["train_accuracy"]) for row in rows])
        loss = np.asarray(series_loss)
        accuracy = np.asarray(series_accuracy)
        axes[0].plot(epochs, loss.mean(axis=0), label=DISPLAY_NAMES[variant])
        axes[0].fill_between(epochs, loss.min(axis=0), loss.max(axis=0), alpha=0.14)
        axes[1].plot(epochs, accuracy.mean(axis=0), label=DISPLAY_NAMES[variant])
        axes[1].fill_between(epochs, accuracy.min(axis=0), accuracy.max(axis=0), alpha=0.14)
    axes[0].set(xlabel="Epoch", ylabel="Training cross-entropy", title="Training loss")
    axes[1].set(xlabel="Epoch", ylabel="Training accuracy (%)", title="Training accuracy")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_test_accuracy(groups: dict[str, list[dict]], output_dir: Path) -> None:
    means = [np.mean([run["test_accuracy"] for run in groups[v]]) for v in VARIANT_ORDER]
    stds = [np.std([run["test_accuracy"] for run in groups[v]], ddof=1) for v in VARIANT_ORDER]
    fig, axis = plt.subplots(figsize=(7.2, 4.6))
    positions = np.arange(len(VARIANT_ORDER))
    bars = axis.bar(positions, means, yerr=stds, capsize=5, color=["#8c8c8c", "#4c78a8", "#59a14f"])
    axis.set_xticks(positions, [DISPLAY_NAMES[v] for v in VARIANT_ORDER])
    axis.set_ylabel("Final test accuracy (%)")
    axis.set_title("CIFAR-10 final test accuracy across three seeds")
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, means):
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_dir / "test_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(input_root)
    validate_runs(runs)
    groups: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        groups[run["variant"]].append(run)
    for variant in groups:
        groups[variant].sort(key=lambda item: int(item["seed"]))

    aggregate_rows = []
    aggregate_json: dict[str, object] = {"variants": {}, "paired_comparisons": {}}
    for variant in VARIANT_ORDER:
        accuracies = np.asarray([run["test_accuracy"] for run in groups[variant]], dtype=float)
        errors = 100.0 - accuracies
        seconds = np.asarray([run["total_training_and_test_seconds"] for run in groups[variant]], dtype=float)
        record = {
            "variant": variant,
            "display_name": DISPLAY_NAMES[variant],
            "n": len(accuracies),
            "test_accuracy_mean": float(accuracies.mean()),
            "test_accuracy_sd": float(accuracies.std(ddof=1)),
            "test_error_mean": float(errors.mean()),
            "test_error_sd": float(errors.std(ddof=1)),
            "runtime_seconds_mean": float(seconds.mean()),
            "parameter_count": int(groups[variant][0]["parameter_count"]),
            "seeds": [int(run["seed"]) for run in groups[variant]],
            "test_accuracies": accuracies.tolist(),
        }
        aggregate_rows.append(record)
        aggregate_json["variants"][variant] = record

    comparisons = [("plain20", "resnet20"), ("resnet20", "resnet20_cutout")]
    for baseline, candidate in comparisons:
        base_by_seed = {int(run["seed"]): float(run["test_accuracy"]) for run in groups[baseline]}
        candidate_by_seed = {int(run["seed"]): float(run["test_accuracy"]) for run in groups[candidate]}
        common_seeds = sorted(base_by_seed.keys() & candidate_by_seed.keys())
        differences = [candidate_by_seed[seed] - base_by_seed[seed] for seed in common_seeds]
        key = f"{candidate}_minus_{baseline}"
        aggregate_json["paired_comparisons"][key] = {
            "seeds": common_seeds,
            "accuracy_point_differences": differences,
            "mean_difference": float(np.mean(differences)),
            "sd_difference": float(np.std(differences, ddof=1)),
            "exact_two_sided_sign_flip_p": exact_sign_flip_pvalue(differences),
            "interpretation_note": "With only three paired runs, the exact test has very low power; effect size and consistency matter more than a significance label.",
        }

    with (output_dir / "aggregate.csv").open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "variant",
            "display_name",
            "n",
            "test_accuracy_mean",
            "test_accuracy_sd",
            "test_error_mean",
            "test_error_sd",
            "runtime_seconds_mean",
            "parameter_count",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(aggregate_rows)
    (output_dir / "aggregate.json").write_text(
        json.dumps(aggregate_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# 实验结果自动汇总",
        "",
        "以下数字由实验文件自动生成，不应手工修改。",
        "",
        "| 方法 | 最终测试准确率 均值 ± 标准差 | 最终测试错误率 均值 ± 标准差 | 参数量 |",
        "|---|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            f"| {row['display_name']} | {row['test_accuracy_mean']:.4f} ± {row['test_accuracy_sd']:.4f} | "
            f"{row['test_error_mean']:.4f} ± {row['test_error_sd']:.4f} | {row['parameter_count']:,} |"
        )
    lines.extend(["", "## 配对差值", ""])
    for key, record in aggregate_json["paired_comparisons"].items():
        lines.append(
            f"- `{key}`：平均准确率差 {record['mean_difference']:.4f} 个百分点，"
            f"差值标准差 {record['sd_difference']:.4f}，精确双侧符号翻转检验 p={record['exact_two_sided_sign_flip_p']:.4f}。"
        )
    (output_dir / "paper_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot_training_curves(runs, output_dir)
    plot_test_accuracy(groups, output_dir)
    print(f"Aggregation complete: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
