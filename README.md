# HCDR-Fusion

Image-level hyperspectral and LiDAR fusion classification on MUUFL, Houston 2013, and Trento.

## Environment

Use Linux, Python 3.10, PyTorch 2.3.0, and CUDA 12.1. A CUDA toolkit with `nvcc` and a compatible C++ compiler is required for Houston.

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
export CUDA_HOME=/usr/local/cuda-12.1
export PATH="$CUDA_HOME/bin:$PATH"
```

Set `CUDA_HOME` to the location of your CUDA toolkit if it is installed elsewhere. The Houston CUDA operator is compiled automatically on first use.

## Datasets

Download the datasets from the providers listed in [datasets/README.md](datasets/README.md), then prepare them:

```bash
python datasets/prepare.py --dataset muufl
python datasets/prepare.py --dataset houston
python datasets/prepare.py --dataset trento
```

## Train and test

Each dataset has one script supporting both training and testing:

| Dataset | Train | Test |
| --- | --- | --- |
| MUUFL | `bash scripts/muufl.sh train` | `bash scripts/muufl.sh test` |
| Houston 2013 | `bash scripts/houston.sh train` | `bash scripts/houston.sh test` |
| Trento | `bash scripts/trento.sh train` | `bash scripts/trento.sh test` |

Training saves the final-epoch checkpoint to `outputs/<dataset>/model.pt`. Testing uses its saved random seed to recover the disjoint train/test partition, and writes OA, AA, Kappa, per-class accuracy, and predictions to the same directory. Test scores are not used for checkpoint selection.

For data stored elsewhere, append `--data-root /path/to/prepared`. To test a specific checkpoint, append `--checkpoint /path/to/model.pt`. Use `CUDA_VISIBLE_DEVICES` to choose a GPU.
