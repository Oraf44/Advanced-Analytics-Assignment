# Assignment 2 — LEGO Minifig Category Classification

Deep learning pipeline for multiclass image classification of LEGO Minifigures using transfer learning with ResNet-50 and EfficientNet-B0.

---

## Project Structure

```
Final_assignment_2/
|
+-- src/                            # Core Python modules
|   +-- config.py                   # All paths and hyperparameters
|   +-- data_utils.py               # EDA, cleaning, dataset, dataloaders
|   +-- models.py                   # MinifigClassifier (ResNet-50 / EfficientNet-B0)
|   +-- trainer.py                  # Two-phase training loop, metrics, checkpoints
|   +-- visualisation.py            # Plots, confusion matrix, Grad-CAM
|
+-- notebooks/                      # Experiments (run in order)
|   +-- 01_eda.ipynb                # Exploratory data analysis
|   +-- 02_comparison.ipynb         # Transfer learning baseline (both models)
|   +-- main_simple.ipynb           # From-scratch baseline (no pretrained weights)
|   +-- main_resnet_augment.ipynb   # ResNet-50 with regularisation experiment
|   +-- main_resnet50.ipynb         # Final selected model
|
+-- outputs/
|   +-- models/                     # Saved .pth checkpoints
|   +-- figures/                    # All generated plots and Grad-CAM images
|   +-- reports/                    # JSON classification reports
|
+-- llm_test/                       # Images used for LLM zero-shot test
+-- README.md
```

The shared dataset (images + metadata JSON) lives **outside** this folder at:
```
project/
+-- data/
|   +-- metadata/
|   |   +-- minifigs.json
|   +-- images/
|       +-- images/
|           +-- *.jpg
+-- Final_assignment_2/
```

---

## Requirements

### Python Version
Python 3.9 or higher is recommended.

### Install all dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas scikit-learn pillow matplotlib seaborn grad-cam jupyter
```

> **Note:** The `--index-url https://download.pytorch.org/whl/cu121` flag installs the CUDA 12.1 build of PyTorch for GPU support (tested on RTX 4060 with CUDA 12.7). If you do not have a GPU, replace with the standard CPU install:
> ```bash
> pip install torch torchvision torchaudio
> ```

### Full list of libraries used

| Library | Purpose |
|---|---|
| `torch` | Model definition, training loop, loss function |
| `torchvision` | Pretrained ResNet-50 and EfficientNet-B0 weights, transforms |
| `numpy` | Numerical operations |
| `pandas` | Metadata loading and dataframe manipulation |
| `scikit-learn` | Train/val/test split, F1 score computation |
| `Pillow (PIL)` | Image loading and RGBA to RGB conversion |
| `matplotlib` | Training curves, confusion matrix, Grad-CAM plots |
| `seaborn` | Styled heatmaps and distribution plots |
| `grad-cam` | Grad-CAM heatmap generation (`pytorch-grad-cam` package) |
| `jupyter` | Running `.ipynb` notebooks |

---

## How Paths Work (pathlib)

All file paths are defined **once** in `src/config.py` using Python's `pathlib.Path`. Every other module imports from there — you never need to hardcode a path anywhere else.

```python
# src/config.py
from pathlib import Path

BASE_DIR    = Path(__file__).resolve().parents[1]   # Final_assignment_2/
PROJECT_DIR = BASE_DIR.parent                        # project/

DATA_DIR    = PROJECT_DIR / "data"
JSON_PATH   = DATA_DIR / "metadata" / "minifigs.json"
IMAGE_DIR   = DATA_DIR / "images"

OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR  = OUTPUTS_DIR / "models"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"
```

**How it works:**
- `Path(__file__)` gets the absolute path of `config.py` itself.
- `.resolve()` converts it to a full absolute path (no relative parts).
- `.parents[1]` goes two levels up from the file to reach `Final_assignment_2/`.
- The `/` operator on a `Path` object joins directories — equivalent to `os.path.join()` but cleaner.
- `BASE_DIR.parent` goes one further level up to reach `project/`.

This means the code works on **any machine** regardless of where the project is saved, as long as the folder structure is kept intact. You never need to change any path manually.

The output directories are created automatically at import time:
```python
for _d in (INTERIM_DIR, FIGURES_DIR, MODELS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
```

---

## How to Run

### Option 1 — Run notebooks in order

Open Jupyter and run the notebooks inside `notebooks/` in this order:

1. `01_eda.ipynb` — explores the dataset, checks class distribution and image quality
2. `02_comparison.ipynb` — trains the transfer learning baseline for both models
3. `main_simple.ipynb` — trains both models from scratch (no pretrained weights)
4. `main_resnet_augment.ipynb` — applies regularisation to ResNet-50
5. `main_resnet50.ipynb` — trains the final selected model (ResNet-50, best result)

```bash
cd Final_assignment_2
jupyter notebook
```

### Option 2 — GPU Training (Recommended)

Make sure PyTorch detects your GPU before running:

```python
import torch
print(torch.cuda.is_available())   # should print True
print(torch.cuda.get_device_name(0))
```

If it prints `False`, reinstall PyTorch with the correct CUDA build (see Requirements above) and restart the kernel.

---

## Key Hyperparameters

All hyperparameters are set in `src/config.py` and can be changed there:

| Parameter | Value | Description |
|---|---|---|
| `MIN_SAMPLES` | 100 | Minimum images per class |
| `TRAIN_RATIO` | 0.70 | Train split proportion |
| `IMG_SIZE` | 224 | Final image size fed to model |
| `BATCH_SIZE` | 64 | Training batch size |
| `PHASE1_EPOCHS` | 5 | Epochs with backbone frozen |
| `PHASE2_EPOCHS` | 20 | Max epochs for full fine-tuning |
| `DROPOUT_RATE` | 0.3 | Dropout in classification head |
| `EARLY_STOPPING_PATIENCE` | 5 | Epochs to wait before stopping |
| `SEED` | 42 | Random seed for reproducibility |

> In the final model notebook (`main_resnet50.ipynb`), some of these are overridden locally: `DROPOUT_RATE=0.4`, `PHASE2_EPOCHS=16`, `PATIENCE=3`.

---

## Results Summary

| Model | Test Accuracy | Macro F1 | Top-3 Accuracy | Train Time |
|---|---|---|---|---|
| ResNet-50 (no pretrain) | 52.2% | 0.468 | — | — |
| EfficientNet-B0 (no pretrain) | 61.9% | 0.590 | — | — |
| EfficientNet-B0 (transfer learning) | 73.2% | 0.703 | 89.3% | 279.3 min |
| ResNet-50 (transfer learning) | 81.5% | 0.768 | 93.3% | 474.7 min |
| ResNet-50 (regularised) | 78.7% | 0.741 | 92.6% | 53.4 min |
| **ResNet-50 (final)** | **82.0%** | **0.774** | **93.0%** | **47.4 min** |

The final model is saved at `outputs/models/resnet50_best.pth`.

---

## Notes

- `NUM_WORKERS = 0` is set in `config.py` for Windows compatibility. On Linux/Mac this can be increased (e.g. `4`) for faster data loading.
- All outputs (figures, model checkpoints, reports) are saved automatically to the `outputs/` folder — you do not need to create these directories manually.
- The dataset folder (`project/data/`) is **shared** with Assignment 1 and is not included inside `Final_assignment_2/`.
