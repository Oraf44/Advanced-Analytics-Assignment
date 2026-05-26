"""
config.py — Centralised paths and hyperparameters for Final Assignment 2.
All other modules import from here; change values once, applies everywhere.
"""
from pathlib import Path

# ── Directory layout ───────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parents[1]   # Final_assignment_2/
PROJECT_DIR = BASE_DIR.parent                        # project/

# Shared data (same dataset as previous pipeline)
DATA_DIR    = PROJECT_DIR / "data"
JSON_PATH   = DATA_DIR / "metadata" / "minifigs.json"
IMAGE_DIR   = DATA_DIR / "images"          # images live at IMAGE_DIR/images/*.jpg

# Outputs local to this assignment
OUTPUTS_DIR = BASE_DIR / "outputs"
INTERIM_DIR = OUTPUTS_DIR / "interim"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR  = OUTPUTS_DIR / "models"
REPORTS_DIR = OUTPUTS_DIR / "reports"

for _d in (INTERIM_DIR, FIGURES_DIR, MODELS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Data settings ──────────────────────────────────────────────────────────
MIN_SAMPLES = 100       # drop categories with fewer images (EDA: 94 dropped)
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15      # test = 1 - train - val = 0.15
SEED        = 42

# ── Image settings ─────────────────────────────────────────────────────────
IMG_SIZE      = 224     # final crop fed to model
RESIZE_SIZE   = 256     # resize before crop
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
# EDA note: our images are brighter (mean ~0.74) than ImageNet (0.485)
# due to white backgrounds — ImageNet normalisation is still correct for
# pretrained weights; augmentation handles the domain gap.

# ── DataLoader settings ────────────────────────────────────────────────────
BATCH_SIZE  = 64
NUM_WORKERS = 0         # set >0 on Linux/Mac for faster loading
PIN_MEMORY  = False

# ── Training hyperparameters ───────────────────────────────────────────────
# Phase 1 — backbone frozen, only head trains
PHASE1_EPOCHS = 5
PHASE1_LR     = 1e-3

# Phase 2 — full fine-tune with differential learning rates
PHASE2_EPOCHS             = 20
PHASE2_LR_HEAD            = 1e-4
PHASE2_LR_LATE_LAYERS     = 5e-5
PHASE2_LR_EARLY_LAYERS    = 1e-5

# Regularisation
DROPOUT_RATE             = 0.3
LABEL_SMOOTHING          = 0.1
EARLY_STOPPING_PATIENCE  = 5
GRAD_CLIP_NORM           = 1.0

# ── Models to compare ──────────────────────────────────────────────────────
MODEL_NAMES = ["efficientnet_b0", "resnet50"]
