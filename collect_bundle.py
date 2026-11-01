from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and package experiment evidence")
    parser.add_argument("--input-root", default="outputs/full")
    parser.add_argument("--aggregate-dir", default="outputs/aggregate")
    parser.add_argument("--output", default="experiment_bundle.zip")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    aggregate_dir = Path(args.aggregate_dir)
    output = Path(args.output)
    expected_runs = [
        f"{variant}_seed{seed}"
        for variant in ("plain20", "resnet20", "resnet20_cutout")
        for seed in (42, 3407, 2026)
    ]
    required_run_files = [
        "metadata.json",
        "train_log.csv",
        "summary.json",
        "final_model.pt",
        "test_predictions.npz",
    ]
    files: list[tuple[Path, str]] = []
    for run_name in expected_runs:
        run_dir = input_root / run_name
        for filename in required_run_files:
            path = run_dir / filename
            if not path.is_file():
                raise RuntimeError(f"Missing required artifact: {path}")
            files.append((path, f"runs/{run_name}/{filename}"))
    for filename in (
        "aggregate.csv",
        "aggregate.json",
        "paper_results.md",
        "training_curves.png",
        "test_accuracy.png",
    ):
        path = aggregate_dir / filename
        if not path.is_file():
            raise RuntimeError(f"Missing aggregate artifact: {path}")
        files.append((path, f"aggregate/{filename}"))

    project_root = Path(__file__).resolve().parent
    source_files = [
        project_root / "README.md",
        project_root / "SERVER_DBW_SETUP.md",
        project_root / "EXPERIMENT_PROTOCOL.md",
        project_root / "REFERENCES.bib",
        project_root / "requirements.txt",
        project_root / "train.py",
        project_root / "aggregate.py",
        project_root / "collect_bundle.py",
        project_root / "check_environment.py",
        *sorted((project_root / "src").glob("*.py")),
        *sorted((project_root / "scripts").glob("*.sh")),
    ]
    for path in source_files:
        if not path.is_file():
            raise RuntimeError(f"Missing source artifact: {path}")
        archive_name = f"source/{path.relative_to(project_root).as_posix()}"
        files.append((path, archive_name))

    manifest = {
        archive_name: {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path, archive_name in files
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, archive_name in files:
            archive.write(path, archive_name)
        archive.writestr("MANIFEST_SHA256.json", json.dumps(manifest, indent=2))
    print(f"Bundle created: {output.resolve()}")
    print(f"Bundle SHA256: {sha256(output)}")


if __name__ == "__main__":
    main()
