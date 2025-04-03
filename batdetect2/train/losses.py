from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F
from pydantic import Field

from batdetect2.configs import BaseConfig
from batdetect2.models.typing import ModelOutput
from batdetect2.train.dataset import TrainExample

__all__ = [
    "bbox_size_loss",
    "compute_loss",
    "focal_loss",
    "mse_loss",
]


class SizeLossConfig(BaseConfig):
    weight: float = 0.1


def bbox_size_loss(
    pred_size: torch.Tensor,
    gt_size: torch.Tensor,
) -> torch.Tensor:
    """
    Bounding box size loss. Only compute loss where there is a bounding box.
    """
    gt_size_mask = (gt_size > 0).float()
    return F.l1_loss(pred_size * gt_size_mask, gt_size, reduction="sum") / (
        gt_size_mask.sum() + 1e-5
    )


class FocalLossConfig(BaseConfig):
    beta: float = 4
    alpha: float = 2


def focal_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    beta: float = 4,
    alpha: float = 2,
) -> torch.Tensor:
    """
    Focal loss adapted from CornerNet: Detecting Objects as Paired Keypoints
    pred  (batch x c x h x w)
    gt    (batch x c x h x w)
    """

    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    pos_loss = torch.log(pred + eps) * torch.pow(1 - pred, alpha) * pos_inds
    neg_loss = (
        torch.log(1 - pred + eps)
        * torch.pow(pred, alpha)
        * torch.pow(1 - gt, beta)
        * neg_inds
    )

    if weights is not None:
        pos_loss = pos_loss * torch.tensor(weights)
        # neg_loss = neg_loss*weights

    if valid_mask is not None:
        pos_loss = pos_loss * valid_mask
        neg_loss = neg_loss * valid_mask

    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    num_pos = pos_inds.float().sum()
    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss


def mse_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Mean squared error loss.
    """
    if valid_mask is None:
        op = ((gt - pred) ** 2).mean()
    else:
        op = (valid_mask * ((gt - pred) ** 2)).sum() / valid_mask.sum()
    return op


class DetectionLossConfig(BaseConfig):
    weight: float = 1.0
    focal: FocalLossConfig = Field(default_factory=FocalLossConfig)


class ClassificationLossConfig(BaseConfig):
    weight: float = 2.0
    focal: FocalLossConfig = Field(default_factory=FocalLossConfig)
    class_weights: Optional[list[float]] = None


class LossConfig(BaseConfig):
    detection: DetectionLossConfig = Field(default_factory=DetectionLossConfig)
    size: SizeLossConfig = Field(default_factory=SizeLossConfig)
    classification: ClassificationLossConfig = Field(
        default_factory=ClassificationLossConfig
    )


class Losses(NamedTuple):
    detection: torch.Tensor
    size: torch.Tensor
    classification: torch.Tensor
    total: torch.Tensor


def compute_loss(
    batch: TrainExample,
    outputs: ModelOutput,
    conf: LossConfig,
    class_weights: Optional[torch.Tensor] = None,
) -> Losses:
    detection_loss = focal_loss(
        outputs.detection_probs,
        batch.detection_heatmap,
        beta=conf.detection.focal.beta,
        alpha=conf.detection.focal.alpha,
    )

    size_loss = bbox_size_loss(
        outputs.size_preds,
        batch.size_heatmap,
    )

    valid_mask = batch.class_heatmap.any(dim=1, keepdim=True).float()
    classification_loss = focal_loss(
        outputs.class_probs,
        batch.class_heatmap,
        weights=class_weights,
        valid_mask=valid_mask,
        beta=conf.classification.focal.beta,
        alpha=conf.classification.focal.alpha,
    )

    total = (
        detection_loss * conf.detection.weight
        + size_loss * conf.size.weight
        + classification_loss * conf.classification.weight
    )

    return Losses(
        detection=detection_loss,
        size=size_loss,
        classification=classification_loss,
        total=total,
    )
