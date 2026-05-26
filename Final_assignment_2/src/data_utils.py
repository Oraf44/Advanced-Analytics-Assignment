"""
data_utils.py — Data loading, cleaning, dataset and dataloader creation.

EDA-informed design decisions:
  - RGBA images (29% of dataset) converted to RGB via PIL .convert('RGB')
  - Class imbalance (31.4x within kept categories) handled by WeightedRandomSampler
  - White backgrounds: standard ImageNet normalisation kept (correct for pretrained weights)
  - Various image sizes (100-600px): standardised to 224x224 via Resize + Crop
"""
import json
import logging
import warnings
from collections import Counter
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ── Label mapping ──────────────────────────────────────────────────────────

class LabelMapping(NamedTuple):
    label2idx:   dict
    idx2label:   dict
    num_classes: int


def create_label_encoding(categories: List[str]) -> LabelMapping:
    """Map sorted category names to integer indices."""
    sorted_cats = sorted(categories)
    label2idx   = {c: i for i, c in enumerate(sorted_cats)}
    idx2label   = {i: c for c, i in label2idx.items()}
    return LabelMapping(label2idx, idx2label, len(sorted_cats))


# ── Data splits ────────────────────────────────────────────────────────────

class DataSplits(NamedTuple):
    train_df: pd.DataFrame
    val_df:   pd.DataFrame
    test_df:  pd.DataFrame


def split_data(df: pd.DataFrame, train_ratio: float, val_ratio: float, seed: int) -> DataSplits:
    """Stratified 70/15/15 split preserving class proportions."""
    test_ratio = 1.0 - train_ratio - val_ratio
    train_df, temp_df = train_test_split(
        df, test_size=(1 - train_ratio), stratify=df["label"], random_state=seed
    )
    rel_val = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - rel_val), stratify=temp_df["label"], random_state=seed
    )
    logger.info(f"Split → train:{len(train_df)}  val:{len(val_df)}  test:{len(test_df)}")
    return DataSplits(train_df.reset_index(drop=True),
                      val_df.reset_index(drop=True),
                      test_df.reset_index(drop=True))


# ── Data loading & cleaning ────────────────────────────────────────────────

def load_and_clean(
    json_path:   Path,
    image_dir:   Path,
    min_samples: int = 100,
) -> Tuple[pd.DataFrame, List[str], pd.Series]:
    """
    Load JSON metadata, verify images exist, filter rare categories.

    Returns
    -------
    df              : cleaned DataFrame with 'label' column added
    valid_categories: list of kept category names
    cat_counts      : value_counts of kept categories
    """
    # Load
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    df = pd.DataFrame(raw)
    logger.info(f"Loaded {len(df)} records")

    # Drop missing image paths
    df = df.dropna(subset=["img_local_path"]).copy()

    # Verify images exist on disk
    df["_exists"] = df["img_local_path"].apply(
        lambda p: (image_dir / p).exists()
    )
    missing = (~df["_exists"]).sum()
    if missing:
        logger.warning(f"{missing} images missing on disk — removing")
    df = df[df["_exists"]].drop(columns=["_exists"]).reset_index(drop=True)
    logger.info(f"Images verified: {len(df)} records remain")

    # Filter rare categories
    counts = df["category"].value_counts()
    valid  = counts[counts >= min_samples].index.tolist()
    df     = df[df["category"].isin(valid)].copy().reset_index(drop=True)
    logger.info(f"Kept {len(valid)}/{len(counts)} categories with >={min_samples} images → {len(df)} rows")

    # Label encoding
    mapping = create_label_encoding(valid)
    df["label"] = df["category"].map(mapping.label2idx)

    cat_counts = df["category"].value_counts()
    return df, valid, cat_counts


# ── Dataset ────────────────────────────────────────────────────────────────

class MinifigDataset(Dataset):
    """
    PyTorch Dataset for Lego minifig images.

    EDA note: ~29% of images are RGBA (transparent backgrounds).
    All images are converted to RGB so the model always receives 3 channels.
    """

    def __init__(self, df: pd.DataFrame, image_dir: Path, transform=None):
        self.df        = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row      = self.df.iloc[idx]
        img_path = self.image_dir / row["img_local_path"]
        label    = int(row["label"])
        try:
            img = Image.open(img_path).convert("RGB")   # handles RGBA → RGB
        except (OSError, IOError):
            logger.warning(f"Could not load {img_path}, using blank image")
            img = Image.new("RGB", (224, 224), (255, 255, 255))
        if self.transform:
            img = self.transform(img)
        return img, label


# ── Transforms ────────────────────────────────────────────────────────────

def get_transforms(
    img_size:    int,
    resize_size: int,
    mean:        List[float],
    std:         List[float],
):
    """
    Training: augmentation to improve generalisation given white backgrounds.
    Val/Test: deterministic centre crop only.
    """
    train_tf = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, eval_tf


# ── DataLoaders ───────────────────────────────────────────────────────────

class DataLoaders(NamedTuple):
    train: DataLoader
    val:   DataLoader
    test:  DataLoader


def create_dataloaders(
    splits:        DataSplits,
    image_dir:     Path,
    label_mapping: LabelMapping,
    batch_size:    int,
    num_workers:   int,
    pin_memory:    bool,
    img_size:      int,
    resize_size:   int,
    mean:          List[float],
    std:           List[float],
) -> DataLoaders:
    """Build train/val/test DataLoaders with WeightedRandomSampler on train."""
    train_tf, eval_tf = get_transforms(img_size, resize_size, mean, std)

    train_ds = MinifigDataset(splits.train_df, image_dir, train_tf)
    val_ds   = MinifigDataset(splits.val_df,   image_dir, eval_tf)
    test_ds  = MinifigDataset(splits.test_df,  image_dir, eval_tf)

    # WeightedRandomSampler — fixes 31.4x class imbalance in training set
    labels       = splits.train_df["label"].tolist()
    class_counts = Counter(labels)
    weights      = [1.0 / class_counts[l] for l in labels]
    sampler      = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)

    logger.info(f"DataLoaders ready — train:{len(train_ds)}  val:{len(val_ds)}  test:{len(test_ds)}")
    return DataLoaders(train_loader, val_loader, test_loader)
