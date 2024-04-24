from typing import Optional

import torch
import torch.nn.functional as F


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
        pos_loss = pos_loss * weights
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
