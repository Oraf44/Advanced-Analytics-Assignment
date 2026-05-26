"""
trainer.py — Training loop, evaluation, metrics, and checkpoint management.

Key features:
  - Two-phase training (frozen backbone → full fine-tune)
  - Combined history across both phases for overfitting analysis
  - Early stopping based on validation loss
  - Gradient clipping
  - Top-1 and Top-3 accuracy
"""
import copy
import logging
import time
from typing import List, NamedTuple, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


# ── Result types ──────────────────────────────────────────────────────────

class TrainingHistory(NamedTuple):
    """Combined history across Phase 1 + Phase 2 for overfitting analysis."""
    train_loss:       List[float]
    train_acc:        List[float]
    val_loss:         List[float]
    val_acc:          List[float]
    phase_boundary:   int          # epoch index where Phase 2 begins


class EvalResult(NamedTuple):
    preds:  List[int]
    labels: List[int]
    probs:  List[List[float]]


class TestMetrics(NamedTuple):
    accuracy:    float
    macro_f1:    float
    weighted_f1: float
    top3_acc:    float


class ModelResult(NamedTuple):
    """Everything produced by training and evaluating one model."""
    model_name:  str
    history:     TrainingHistory
    eval_result: EvalResult
    metrics:     TestMetrics
    train_time:  float           # seconds
    num_params:  int


# ── Epoch-level functions ────────────────────────────────────────────────

def train_one_epoch(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
    grad_clip: float = 1.0,
) -> Tuple[float, float]:
    """Run one training epoch. Returns (loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += len(labels)

    return total_loss / total, correct / total


def evaluate_epoch(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> Tuple[float, float, EvalResult]:
    """Evaluate model. Returns (loss, accuracy, EvalResult)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            probs   = torch.softmax(outputs, dim=1)
            preds   = outputs.argmax(1)

            total_loss += loss.item() * len(labels)
            correct    += (preds == labels).sum().item()
            total      += len(labels)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    return (
        total_loss / total,
        correct / total,
        EvalResult(all_preds, all_labels, all_probs),
    )


# ── Metrics ──────────────────────────────────────────────────────────────

def compute_metrics(eval_result: EvalResult, num_classes: int) -> TestMetrics:
    """Compute accuracy, macro F1, weighted F1, and top-3 accuracy."""
    preds  = np.array(eval_result.preds)
    labels = np.array(eval_result.labels)
    probs  = np.array(eval_result.probs)

    accuracy    = (preds == labels).mean()
    macro_f1    = f1_score(labels, preds, average="macro",    zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)

    # Top-3 accuracy: correct if true label is in top-3 predicted classes
    top3_preds = np.argsort(probs, axis=1)[:, -3:]
    top3_acc   = np.mean([labels[i] in top3_preds[i] for i in range(len(labels))])

    return TestMetrics(float(accuracy), float(macro_f1), float(weighted_f1), float(top3_acc))


# ── Main training function ───────────────────────────────────────────────

def train_model(
    model:                   nn.Module,
    train_loader:            torch.utils.data.DataLoader,
    val_loader:              torch.utils.data.DataLoader,
    device:                  torch.device,
    label_smoothing:         float = 0.1,
    phase1_epochs:           int   = 5,
    phase1_lr:               float = 1e-3,
    phase2_epochs:           int   = 20,
    phase2_lr_early:         float = 1e-5,
    phase2_lr_late:          float = 5e-5,
    phase2_lr_head:          float = 1e-4,
    early_stopping_patience: int   = 5,
    grad_clip:               float = 1.0,
) -> TrainingHistory:
    """
    Two-phase training:
      Phase 1 — backbone frozen, train head only (fast convergence)
      Phase 2 — full fine-tune with differential LRs + early stopping

    Returns a combined TrainingHistory across both phases.
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    train_loss_hist, train_acc_hist = [], []
    val_loss_hist,   val_acc_hist   = [], []

    # ── Phase 1 ──────────────────────────────────────────────────────────
    logger.info(f"Phase 1 — {phase1_epochs} epochs (backbone frozen)")
    model.freeze_backbone()
    optimizer1 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=phase1_lr
    )

    for epoch in range(1, phase1_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer1, device, grad_clip)
        va_loss, va_acc, _ = evaluate_epoch(model, val_loader, criterion, device)
        train_loss_hist.append(tr_loss)
        train_acc_hist.append(tr_acc)
        val_loss_hist.append(va_loss)
        val_acc_hist.append(va_acc)
        logger.info(f"  P1 Epoch {epoch:2d}/{phase1_epochs} | "
                    f"train_loss={tr_loss:.4f} acc={tr_acc:.3f} | "
                    f"val_loss={va_loss:.4f} acc={va_acc:.3f}")

    phase_boundary = len(train_loss_hist)   # marks start of Phase 2

    # ── Phase 2 ──────────────────────────────────────────────────────────
    logger.info(f"Phase 2 — up to {phase2_epochs} epochs (full fine-tune)")
    model.unfreeze_backbone()
    param_groups = model.get_param_groups(phase2_lr_early, phase2_lr_late, phase2_lr_head)
    optimizer2   = torch.optim.Adam(param_groups)
    scheduler    = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=phase2_epochs)

    best_val_loss  = float("inf")
    best_state     = copy.deepcopy(model.state_dict())
    patience_count = 0

    for epoch in range(1, phase2_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer2, device, grad_clip)
        va_loss, va_acc, _ = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        train_loss_hist.append(tr_loss)
        train_acc_hist.append(tr_acc)
        val_loss_hist.append(va_loss)
        val_acc_hist.append(va_acc)

        logger.info(f"  P2 Epoch {epoch:2d}/{phase2_epochs} | "
                    f"train_loss={tr_loss:.4f} acc={tr_acc:.3f} | "
                    f"val_loss={va_loss:.4f} acc={va_acc:.3f}")

        if va_loss < best_val_loss:
            best_val_loss  = va_loss
            best_state     = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= early_stopping_patience:
                logger.info(f"  Early stopping at epoch {epoch} (patience={early_stopping_patience})")
                break

    # Restore best weights
    model.load_state_dict(best_state)
    logger.info(f"Training complete. Best val_loss={best_val_loss:.4f}")

    return TrainingHistory(
        train_loss_hist, train_acc_hist,
        val_loss_hist,   val_acc_hist,
        phase_boundary,
    )


# ── Checkpoint ───────────────────────────────────────────────────────────

def save_checkpoint(model: nn.Module, path, label_mapping, metrics: TestMetrics,
                    history: Optional[TrainingHistory] = None):
    payload = {
        "state_dict":  model.state_dict(),
        "backbone":    model.backbone_name,
        "label2idx":   label_mapping.label2idx,
        "idx2label":   label_mapping.idx2label,
        "num_classes": label_mapping.num_classes,
        "metrics":     metrics._asdict(),
    }
    if history is not None:
        payload["history"] = {
            "train_loss":     history.train_loss,
            "train_acc":      history.train_acc,
            "val_loss":       history.val_loss,
            "val_acc":        history.val_acc,
            "phase_boundary": history.phase_boundary,
        }
    torch.save(payload, path)
    logger.info(f"Checkpoint saved → {path}")


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    logger.info(f"Checkpoint loaded from {path}")
    return ckpt


def history_from_checkpoint(ckpt) -> Optional[TrainingHistory]:
    """Reconstruct TrainingHistory from a checkpoint dict if history was saved."""
    if "history" not in ckpt:
        return None
    h = ckpt["history"]
    return TrainingHistory(
        h["train_loss"], h["train_acc"],
        h["val_loss"],   h["val_acc"],
        h["phase_boundary"],
    )
