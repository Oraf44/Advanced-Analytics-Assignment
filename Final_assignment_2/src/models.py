"""
models.py — MinifigClassifier supporting both EfficientNet-B0 and ResNet-50.

Architecture:
    Pretrained backbone  →  Dropout(0.3)  →  Linear(num_classes)

Both models use the identical head so the comparison is fair.
The only difference is the backbone (feature extractor).
"""
import torch.nn as nn
import torchvision.models as models
from torchvision.models import (
    EfficientNet_B0_Weights, EfficientNet_B1_Weights, EfficientNet_B2_Weights,
    ResNet50_Weights,
)


class MinifigClassifier(nn.Module):
    """
    Transfer-learning classifier for Lego minifig categories.

    Parameters
    ----------
    backbone_name : 'efficientnet_b0' or 'resnet50'
    num_classes   : number of output classes (28)
    dropout_rate  : dropout probability before the final linear layer
    """

    def __init__(self, backbone_name: str, num_classes: int, dropout_rate: float = 0.3):
        super().__init__()
        self.backbone_name = backbone_name

        if backbone_name == "efficientnet_b0":
            self.backbone  = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            in_features    = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()

        elif backbone_name == "efficientnet_b1":
            self.backbone  = models.efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V1)
            in_features    = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()

        elif backbone_name == "efficientnet_b2":
            self.backbone  = models.efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
            in_features    = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()

        elif backbone_name == "resnet50":
            self.backbone  = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            in_features    = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        else:
            raise ValueError(f"Unknown backbone '{backbone_name}'. "
                             f"Choose efficientnet_b0/b1/b2 or resnet50.")

        # Shared classification head (same for both models — fair comparison)
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes),
        )

    # ── Forward ──────────────────────────────────────────────────────────
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

    # ── Freeze / Unfreeze ────────────────────────────────────────────────
    def freeze_backbone(self):
        """Phase 1: lock backbone, only train the head."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.head.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self):
        """Phase 2: allow all parameters to be updated."""
        for param in self.parameters():
            param.requires_grad = True

    # ── Differential LR groups ──────────────────────────────────────────
    def get_param_groups(
        self,
        lr_early: float,
        lr_late:  float,
        lr_head:  float,
    ):
        """
        Return parameter groups with differential learning rates for Phase 2.

        Early layers (generic features) → small LR to preserve pretrained knowledge.
        Late layers  (task-specific)    → medium LR.
        Head                            → largest LR (random init, needs fast learning).
        """
        if self.backbone_name in ("efficientnet_b0", "efficientnet_b1", "efficientnet_b2"):
            early_params = list(self.backbone.features[:5].parameters())
            late_params  = list(self.backbone.features[5:].parameters())

        elif self.backbone_name == "resnet50":
            early_params = (
                list(self.backbone.conv1.parameters()) +
                list(self.backbone.bn1.parameters()) +
                list(self.backbone.layer1.parameters()) +
                list(self.backbone.layer2.parameters())
            )
            late_params = (
                list(self.backbone.layer3.parameters()) +
                list(self.backbone.layer4.parameters())
            )

        head_params = list(self.head.parameters())

        return [
            {"params": early_params, "lr": lr_early},
            {"params": late_params,  "lr": lr_late},
            {"params": head_params,  "lr": lr_head},
        ]

    # ── Grad-CAM target layer ────────────────────────────────────────────
    def get_gradcam_layer(self):
        """Return the last convolutional layer used for Grad-CAM."""
        if self.backbone_name in ("efficientnet_b0", "efficientnet_b1", "efficientnet_b2"):
            return self.backbone.features[-1]
        elif self.backbone_name == "resnet50":
            return self.backbone.layer4[-1]


def count_parameters(model: nn.Module) -> dict:
    """Return total and trainable parameter counts."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
