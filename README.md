# Reproducible Residual Learning and Cutout Study on CIFAR-10

This repository contains the code and verified aggregate results for a controlled CIFAR-10 reproduction study comparing parameter-matched Plain-20, ResNet-20, and ResNet-20 with Cutout.

The project does not claim a new residual architecture or a new Cutout method. Its purpose is to provide a traceable comparison under a fixed training protocol, multiple random seeds, and archived evidence.

## Experimental design

| Variant | Residual shortcuts | Cutout | Trainable parameters |
|---|---:|---:|---:|
| Plain-20 | No | No | 269,722 |
| ResNet-20 | Yes, parameter-free Option A | No | 269,722 |
| ResNet-20 + Cutout | Yes, parameter-free Option A | Yes, length 16 | 269,722 |

- Dataset: official CIFAR-10 split with 50,000 training images and 10,000 test images.
- Seeds: `42`, `2026`, and `3407`.
- Training: 200 epochs, SGD, batch size 128, initial learning rate 0.1, momentum 0.9, and weight decay 0.0001.
- Learning-rate schedule: multiply by 0.1 after epochs 100 and 150.
- Test policy: evaluate the test set once, after the final training epoch of each run.
- Total formal runs: 3 variants × 3 seeds = 9.

## Verified aggregate results

| Variant | Test accuracy, mean ± sample SD (%) | Test error, mean ± sample SD (%) |
|---|---:|---:|
| Plain-20 | 90.8600 ± 0.0346 | 9.1400 ± 0.0346 |
| ResNet-20 | 91.8433 ± 0.0404 | 8.1567 ± 0.0404 |
| ResNet-20 + Cutout | 92.7467 ± 0.0611 | 7.2533 ± 0.0611 |

ResNet-20 improved mean accuracy over Plain-20 by 0.9833 percentage points. Adding Cutout to ResNet-20 produced a further 0.9033-point improvement. Each comparison contains only three paired runs; the exact two-sided sign-flip value is `p = 0.25`, so these results are not presented as a claim of statistical significance.

The machine-readable results are in [`results/aggregate/aggregate.json`](results/aggregate/aggregate.json) and [`results/aggregate/aggregate.csv`](results/aggregate/aggregate.csv).

## Repository structure

```text
.
├── train.py
├── aggregate.py
├── collect_bundle.py
├── check_environment.py
├── src/
├── scripts/
├── docs/EXPERIMENT_PROTOCOL.md
├── results/aggregate/
├── requirements.txt
└── REFERENCES.bib
```

The complete nine-run evidence bundle, including weights, per-epoch logs, final predictions, confusion matrices, environment metadata, and the internal SHA-256 manifest, is distributed as the release asset `experiment_bundle.zip`. Supporting installation and execution logs are distributed separately as `supporting_logs.tar.gz`.

## Reproduction

Create an isolated Python environment and install the declared dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Check the runtime environment:

```bash
bash scripts/01_check_environment.sh
```

Run the smoke test before any formal experiment:

```bash
bash scripts/02_smoke_test.sh
```

Run all nine formal experiments:

```bash
bash scripts/03_run_full_experiments.sh | tee full_experiments.txt
```

Aggregate and package the evidence:

```bash
python aggregate.py --input-root outputs/full --output-dir outputs/aggregate
python collect_bundle.py --input-root outputs/full --aggregate-dir outputs/aggregate --output experiment_bundle.zip
```

See [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md) for the fixed protocol and academic-integrity constraints.

## Evidence integrity

The verified release-asset checksums are:

```text
85f79b849aa47871493fcac44ce651a6a76480073161d676387cc5d9a5d3eb36  experiment_bundle.zip
9ccf1e3130eabc24713a77e19669cd3866445237a0897f96a5d1d604f4f36f1e  supporting_logs.tar.gz
```

The CIFAR-10 dataset is not redistributed in this repository or in the evidence bundle.

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). The public source repository is <https://github.com/bobod683/cifar10-residual-cutout-reproduction>. An archival DOI will be added only after a corresponding public record actually exists.

## License

The source code in this repository is released under the [MIT License](LICENSE). The CIFAR-10 dataset is not included and remains subject to its original terms. Third-party software and cited publications remain subject to their respective licenses and copyrights.
