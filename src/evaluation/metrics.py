"""Segmentation metrics for multi-class semantic segmentation.

Wraps torchmetrics. Call update() on each batch, compute() at the end of the epoch.
"""
from typing import Dict, List, Optional

import torch
from torchmetrics import Accuracy, F1Score, JaccardIndex


class SegmentationMetrics:
    """Accumulates per-batch predictions and computes mIoU, F1, accuracy.

    Usage:
        metrics = SegmentationMetrics(num_classes=7, device=device, class_names=[...])
        for images, masks in val_loader:
            logits = model(images)
            metrics.update(logits, masks)
        results = metrics.compute()
        metrics.reset()  # important for the next epoch
    """

    def __init__(
        self,
        num_classes: int,
        device: torch.device,
        class_names: Optional[List[str]] = None,
    ) -> None:
        self.num_classes = num_classes
        self.device = device
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]

        # task='multiclass' + average='macro' gives the mean of per-class IoU
        # i.e. the classical mIoU
        self.miou = JaccardIndex(
            task="multiclass", num_classes=num_classes, average="macro",
        ).to(device)
        self.per_class_iou = JaccardIndex(
            task="multiclass", num_classes=num_classes, average="none",
        ).to(device)
        self.f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="macro",
        ).to(device)
        # 'micro' = pixel accuracy (aggregate correct / total)
        self.acc = Accuracy(
            task="multiclass", num_classes=num_classes, average="micro",
        ).to(device)

    def reset(self) -> None:
        self.miou.reset()
        self.per_class_iou.reset()
        self.f1.reset()
        self.acc.reset()

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """Update metrics with one batch of predictions and targets.

        Args:
            logits:  (B, C, H, W) model output before softmax.
            targets: (B, H, W) ground-truth class IDs (long tensor).
        """
        preds = logits.argmax(dim=1)  # (B, H, W) of class IDs
        self.miou.update(preds, targets)
        self.per_class_iou.update(preds, targets)
        self.f1.update(preds, targets)
        self.acc.update(preds, targets)

    def compute(self) -> Dict:
        """Aggregate and return a dict of metric values (all Python scalars)."""
        per_class = self.per_class_iou.compute().detach().cpu().numpy()
        return {
            "mIoU": float(self.miou.compute().item()),
            "F1": float(self.f1.compute().item()),
            "accuracy": float(self.acc.compute().item()),
            "per_class_iou": {
                name: float(per_class[i]) for i, name in enumerate(self.class_names)
            },
        }
