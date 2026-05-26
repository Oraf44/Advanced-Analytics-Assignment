"""
visualisation.py — All plotting functions for the model comparison pipeline.

Functions:
  plot_training_curves      — train vs val loss/acc per model (overfitting check)
  plot_overfitting_comparison — both models on same axes
  plot_model_comparison     — side-by-side metric bar chart
  plot_confusion_matrix     — heatmap
  plot_per_class_accuracy   — bar chart per class
  plot_gradcam              — Grad-CAM heatmap overlays
"""
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from PIL import Image

# ── Helper ────────────────────────────────────────────────────────────────

def _save_or_show(fig, save_path=None, show=True):
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    plt.close(fig)


# ── Training curves (overfitting check) ───────────────────────────────────

def plot_training_curves(
    history,
    model_name:  str,
    save_path:   Optional[Path] = None,
    show:        bool = True,
):
    """
    Plot train vs val loss and accuracy for one model.

    The vertical dashed line marks the Phase 1 / Phase 2 boundary.
    Overfitting shows as: train_loss keeps falling while val_loss rises.
    """
    epochs = list(range(1, len(history.train_loss) + 1))
    pb     = history.phase_boundary   # epoch index where Phase 2 starts

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Training Curves — {model_name}", fontsize=13)

    # Loss
    axes[0].plot(epochs, history.train_loss, label="Train Loss",      color="steelblue",  lw=2)
    axes[0].plot(epochs, history.val_loss,   label="Val Loss",        color="tomato",     lw=2)
    axes[0].axvline(x=pb + 0.5, color="gray", linestyle="--", lw=1.5, label="Phase 2 start")
    axes[0].fill_between(epochs, history.train_loss, history.val_loss,
                          alpha=0.1, color="tomato", label="Gap (overfitting risk)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend(fontsize=8)

    # Accuracy
    axes[1].plot(epochs, [a * 100 for a in history.train_acc], label="Train Acc", color="steelblue", lw=2)
    axes[1].plot(epochs, [a * 100 for a in history.val_acc],   label="Val Acc",   color="tomato",    lw=2)
    axes[1].axvline(x=pb + 0.5, color="gray", linestyle="--", lw=1.5, label="Phase 2 start")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    _save_or_show(fig, save_path, show)


def plot_overfitting_comparison(
    histories:  dict,          # {model_name: TrainingHistory}
    save_path:  Optional[Path] = None,
    show:       bool = True,
):
    """
    Plot val_loss for both models on the same axes to compare overfitting behaviour.
    Also shows the train-val gap (loss gap = val_loss - train_loss) per epoch.
    """
    colours = {"efficientnet_b0": "steelblue", "resnet50": "tomato"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Overfitting Analysis — Both Models", fontsize=13)

    for name, hist in histories.items():
        epochs = list(range(1, len(hist.train_loss) + 1))
        colour = colours.get(name, "gray")
        gap    = [v - t for v, t in zip(hist.val_loss, hist.train_loss)]

        axes[0].plot(epochs, hist.val_loss,   label=f"{name} val",   color=colour, lw=2)
        axes[0].plot(epochs, hist.train_loss, label=f"{name} train", color=colour, lw=2, linestyle="--")
        axes[1].plot(epochs, gap,             label=name,            color=colour, lw=2)

    axes[0].set_title("Train vs Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=8)

    axes[1].axhline(y=0, color="black", lw=0.8, linestyle="--")
    axes[1].set_title("Loss Gap (val − train)  ← higher = more overfitting")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Gap")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    _save_or_show(fig, save_path, show)


# ── Model comparison ──────────────────────────────────────────────────────

def plot_model_comparison(
    results:    dict,          # {model_name: ModelResult}
    save_path:  Optional[Path] = None,
    show:       bool = True,
):
    """Bar chart comparing Accuracy, Macro F1, Weighted F1, Top-3 Acc."""
    metric_names = ["Accuracy", "Macro F1", "Weighted F1", "Top-3 Acc"]
    model_names  = list(results.keys())
    colours      = ["steelblue", "tomato"]

    values = {
        m: [
            getattr(results[m].metrics, k)
            for k in ("accuracy", "macro_f1", "weighted_f1", "top3_acc")
        ]
        for m in model_names
    }

    x    = np.arange(len(metric_names))
    w    = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (name, colour) in enumerate(zip(model_names, colours)):
        bars = ax.bar(x + i * w, values[name], w, label=name, color=colour, alpha=0.85)
        for bar, val in zip(bars, values[name]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — EfficientNet-B0 vs ResNet-50")
    ax.legend()
    plt.tight_layout()
    _save_or_show(fig, save_path, show)


# ── Confusion matrix ──────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm:          np.ndarray,
    class_names: List[str],
    title:       str = "Confusion Matrix",
    save_path:   Optional[Path] = None,
    show:        bool = True,
):
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center", fontsize=5,
                        color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    _save_or_show(fig, save_path, show)


# ── Per-class accuracy ────────────────────────────────────────────────────

def plot_per_class_accuracy_comparison(
    cm1:         np.ndarray,
    cm2:         np.ndarray,
    class_names: List[str],
    name1:       str = "EfficientNet-B0",
    name2:       str = "ResNet-50",
    save_path:   Optional[Path] = None,
    show:        bool = True,
):
    """Compare per-class accuracy for both models side by side."""
    def per_class_acc(cm):
        row_sums = cm.sum(axis=1)
        return np.where(row_sums > 0, cm.diagonal() / row_sums, 0.0)

    acc1 = per_class_acc(cm1)
    acc2 = per_class_acc(cm2)
    x    = np.arange(len(class_names))
    w    = 0.4

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - w/2, acc1 * 100, w, label=name1, color="steelblue", alpha=0.85)
    ax.bar(x + w/2, acc2 * 100, w, label=name2, color="tomato",    alpha=0.85)
    ax.axhline(y=50, color="black", linestyle="--", lw=0.8, label="50% line")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Per-Class Accuracy — Model Comparison")
    ax.legend()
    plt.tight_layout()
    _save_or_show(fig, save_path, show)


# ── Grad-CAM ──────────────────────────────────────────────────────────────

def plot_gradcam_grid(
    original_imgs: List,
    cam_imgs:      List,
    true_names:    List[str],
    pred_names:    List[str],
    title:         str = "Grad-CAM",
    save_path:     Optional[Path] = None,
    show:          bool = True,
):
    """Show original image + Grad-CAM heatmap side by side for each sample."""
    n = len(original_imgs)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    if n == 1:
        axes = axes[:, np.newaxis]
    fig.suptitle(title, fontsize=12)

    for i in range(n):
        axes[0, i].imshow(original_imgs[i])
        axes[0, i].set_title(f"True: {true_names[i][:12]}", fontsize=7)
        axes[0, i].axis("off")

        axes[1, i].imshow(cam_imgs[i])
        axes[1, i].set_title(f"Pred: {pred_names[i][:12]}", fontsize=7,
                              color="green" if true_names[i] == pred_names[i] else "red")
        axes[1, i].axis("off")

    plt.tight_layout()
    _save_or_show(fig, save_path, show)


def build_gradcam_results(model, dataset, indices, idx2label, device):
    """Compute Grad-CAM overlays for given sample indices."""
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        print("Install pytorch-grad-cam: pip install grad-cam")
        return []

    target_layer = model.get_gradcam_layer()
    cam          = GradCAM(model=model, target_layers=[target_layer])
    results      = []

    model.eval()
    for idx in indices:
        img_tensor, true_label = dataset[idx]
        inp = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(inp)
        pred_label = logits.argmax(1).item()

        grayscale_cam = cam(input_tensor=inp)[0]

        # Denormalise for display
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        orig = (img_tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

        cam_img = show_cam_on_image(orig.astype(np.float32), grayscale_cam, use_rgb=True)

        results.append({
            "orig_img":  orig,
            "cam_img":   cam_img,
            "true_name": idx2label[true_label],
            "pred_name": idx2label[pred_label],
        })

    return results
