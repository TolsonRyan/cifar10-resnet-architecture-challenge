# Parameter-Efficient Residual Classification

NYU Tandon Deep Learning Project 1 — Sheng Tao.

A custom PyramidNet-110 with ShakeDrop, trained from scratch on the permitted CIFAR-10 training data under a five-million-parameter budget. The final inference model averages the weights of two short continuation branches and uses 50-view test-time augmentation (TTA).

| Measure | Result |
|---|---:|
| Trainable parameters | 4,804,512 |
| Training / development images | 45,000 / 5,000 |
| Development accuracy, 50 views | 97.90% |
| Official final hidden-test accuracy | 87.60% |

The development split was reused for model selection. The official final test result was confirmed by the participant; hidden test labels are not available in this repository.

## Architecture and training

The network has 54 basic residual blocks with gradually increasing channels, average-pooling and zero-padding shortcuts, and global average pooling. Stage-end widths are 48, 80, and 112. The base model trains for 400 epochs with SGD/Nesterov, RandAugment, Cutout, cosine learning-rate decay, and EMA. Two branches start from its final EMA weights and each train for 20 epochs with seeds 62026 and 62029. During continuation ShakeDrop uses its inference multiplier while BN remains trainable.

Final Raw floating tensors, including BN statistics, are averaged equally; integer buffers come from the second branch. This produces one model, not a prediction ensemble. TTA averages softmax probabilities over all 25 integer translations within two pixels and their horizontal flips. No pretrained weights or external training images are used.

## Installation

Tested with Python 3.11.4, PyTorch 2.6.0, CUDA 12.4, and an RTX 4060 Laptop GPU (8 GB).

```bash
python -m venv .venv
# Activate the environment using your operating system's standard command.
python -m pip install -r requirements-notebook.txt
```

Training requires a CUDA GPU supporting BF16. For CPU inference, install the matching CPU builds of torch and torchvision instead of the CUDA requirements, then install requirements.txt and the notebook dependencies.

## Data preparation

Obtain the official competition files through the course/Kaggle access provided to you. Do not substitute another CIFAR download: row order and file names are part of reproduction. Place these files under `data/`:

```text
data/train_images/          data/test_images/
data/train_labels.csv       data/test.csv
data/sample_submission.csv
```

Run `python src/audit_data.py`. It verifies RGB images and IDs, builds uint8 caches, and reconstructs the stratified seed-2026 split. Original datasets and generated caches are excluded from Git.

## Evaluate and reproduce predictions

Place the released final checkpoint at `artifacts/final.pt`. Its SHA-256 is `ada2fc2fbbc0310665312412ef01b8caec98a3a3c76e9c237c83f1580d6ae9ca`.

```bash
python src/inference.py --validate
python src/inference.py --output outputs/submission.csv
```

Both commands use FP32 and the exact ordered 50-view policy. CPU inference is available with `--device cpu`. The submitted CSV SHA-256 is `9ad0bec166bb59bda06feb1e9be1922005bea2c8060fd790836e23e8c7386e12`. The exporter refuses to overwrite an existing file. Cross-hardware numerical differences should be investigated if predictions differ; retraining is not guaranteed to produce bitwise-identical weights.

## Train the complete recipe

Run these commands from the repository root, using a fresh `runs/` directory:

```bash
python src/train_round7.py --config configs/final/pyramid110_a96_sd_seed2026.json
python src/train_final_sprint.py --config configs/final/deterministic20_seed62026.json
python src/train_final_sprint.py --config configs/final/c3_replicate_seed62029.json
python src/average_weights.py --first runs/final_sprint/deterministic20_seed62026/final_raw.pt --second runs/final_sprint/c3_replicate_seed62029/final_raw.pt --output artifacts/retrained_average.pt
python src/inference.py --checkpoint artifacts/retrained_average.pt --validate
python src/inference.py --checkpoint artifacts/retrained_average.pt --output outputs/retrained_submission.csv
```

The historical training entry-point names are retained to minimize changes to tested code. Use `--resume` to restore a run's saved optimizer, EMA, schedule position, and RNG states. Recorded training-workflow time for the base and two branches was approximately 7.86 hours. Allow additional time and disk space for checkpoint writes and evaluation. A retrained model follows the same recipe; it is not expected to share the archived checkpoint hash.

## Notebook and report

`notebooks/final_model.ipynb` contains executed parameter checks, training curves, full development evaluation, a confusion matrix, and complete submission reproduction. Run it from the repository root or notebook directory. Set `CIFAR_DATA` to use an existing official dataset directory. Remove or rename its previously generated `outputs/notebook_submission.csv` before rerunning the export cell.

The AAAI report source and PDF are in `report/`. `results/manifest.json` records final artifact provenance, and `results/evidence/` contains the three training logs used in the report. Full historical experiments remain outside this focused release.

## References and sources

The report cites ResNet (He et al., 2016), PyramidNet (Han et al., 2017), ShakeDrop (Yamada et al., 2019), RandAugment (Cubuk et al., 2020), Cutout (DeVries and Taylor, 2017), and weight averaging (Izmailov et al., 2018). See `docs/SOURCES_AND_TOOLS.md` for implementation and tool disclosure. The torchvision BSD-3-Clause notice is retained under `third_party/`.
