"""Loss functions and configurations for training BatDetect2 models.

This module defines the loss functions used to train BatDetect2 models,
including individual loss components for different prediction tasks (detection,
classification, size regression) and a main coordinating loss function that
combines them.

It utilizes common loss types like L1 loss (`BBoxLoss`) for regression and
Focal Loss (`FocalLoss`) for handling class imbalance in dense detection and
classification tasks. Configuration objects (`LossConfig`, etc.) allow for easy
customization of loss parameters and weights via configuration files.

The primary entry points are:
- `LossFunction`: An `nn.Module` that computes the weighted sum of individual
  loss components given model outputs and ground truth targets.
- `build_loss`: A factory function that constructs the `LossFunction` based
  on a `LossConfig` object.
- `LossConfig`: The Pydantic model for configuring loss weights and parameters.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import Field
from torch import nn

from batdetect2.configs import BaseConfig
from batdetect2.models.types import ModelOutput
from batdetect2.train.dataset import TrainExample
from batdetect2.train.types import Losses, LossProtocol

__all__ = [
    "BBoxLoss",
    "ClassificationLossConfig",
    "DetectionLossConfig",
    "FocalLoss",
    "FocalLossConfig",
    "LossConfig",
    "LossFunction",
    "MSELoss",
    "SizeLossConfig",
    "build_loss",
]


class SizeLossConfig(BaseConfig):
    """Configuration for the bounding box size loss component.

    Attributes
    ----------
    weight : float, default=0.1
        The weighting factor applied to the size loss when combining it with
        other losses (detection, classification) to form the total training
        loss.
    """

    weight: float = 0.1


class BBoxLoss(nn.Module):
    """Computes L1 loss for bounding box size regression.

    Calculates the Mean Absolute Error (MAE or L1 loss) between the predicted
    size dimensions (`pred`) and the ground truth size dimensions (`gt`).
    Crucially, the loss is only computed at locations where the ground truth
    size heatmap (`gt`) contains non-zero values (i.e., at the reference points
    of actual annotated sound events). This prevents the model from being
    penalized for size predictions in background regions.

    The loss is summed over all valid locations and normalized by the number
    of valid locations.
    """

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Calculate masked L1 loss for size prediction.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted size tensor, typically shape `(B, 2, H, W)`, where
            channels represent scaled width and height.
        gt : torch.Tensor
            Ground truth size tensor, same shape as `pred`. Non-zero values
            indicate locations and target sizes of actual annotations.

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the calculated masked L1 loss.
        """
        gt_size_mask = (gt > 0).float()
        masked_pred = pred * gt_size_mask
        loss = F.l1_loss(masked_pred, gt, reduction="sum")
        num_pos = gt_size_mask.sum() + 1e-5
        return loss / num_pos


class FocalLossConfig(BaseConfig):
    """Configuration parameters for the Focal Loss function.

    Attributes
    ----------
    beta : float, default=4
        Exponent controlling the down-weighting of easy negative examples.
        Higher values increase down-weighting (focus more on hard negatives).
    alpha : float, default=2
        Exponent controlling the down-weighting based on prediction confidence.
        Higher values focus more on misclassified examples (both positive and
        negative).
    """

    beta: float = 4
    alpha: float = 2


class FocalLoss(nn.Module):
    """Focal Loss implementation, adapted from CornerNet.

    Addresses class imbalance in dense object detection/classification tasks by
    down-weighting the loss contribution from easy examples (both positive and
    negative), allowing the model to focus more on hard-to-classify examples.

    Parameters
    ----------
    eps : float, default=1e-5
        Small epsilon value added for numerical stability.
    beta : float, default=4
        Exponent focusing on hard negative examples (modulates `(1-gt)^beta`).
    alpha : float, default=2
        Exponent focusing on misclassified examples (modulates `(1-p)^alpha`
        for positives and `p^alpha` for negatives).
    class_weights : torch.Tensor, optional
        Optional tensor containing weights for each class (applied to positive
        loss). Shape should be broadcastable to the channel dimension of the
        input tensors.
    mask_zero : bool, default=False
        If True, ignores loss contributions from spatial locations where the
        ground truth `gt` tensor is zero across *all* channels. Useful for
        classification heatmaps where some areas might have no assigned class.

    References
    ----------
    - Lin, T. Y., et al. "Focal loss for dense object detection." ICCV 2017.
    - Law, H., & Deng, J. "CornerNet: Detecting Objects as Paired Keypoints."
      ECCV 2018.
    """

    def __init__(
        self,
        eps: float = 1e-5,
        beta: float = 4,
        alpha: float = 2,
        class_weights: Optional[torch.Tensor] = None,
        mask_zero: bool = False,
    ):
        super().__init__()
        self.class_weights = class_weights
        self.eps = eps
        self.beta = beta
        self.alpha = alpha
        self.mask_zero = mask_zero

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the Focal Loss.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted probabilities or logits (typically sigmoid output for
            detection, or softmax/sigmoid for classification). Must be in the
            range [0, 1] after potential activation. Shape `(B, C, H, W)`.
        gt : torch.Tensor
            Ground truth heatmap tensor. Shape `(B, C, H, W)`. Values typically
            represent target probabilities (e.g., Gaussian peaks for detection,
            one-hot encoding or smoothed labels for classification). For the
            adapted CornerNet loss, `gt=1` indicates a positive location, and
            values `< 1` indicate negative locations (with potential Gaussian
            weighting `(1-gt)^beta` for negatives near positives).

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the computed focal loss, normalized by
            the number of positive locations.
        """

        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        pos_loss = (
            torch.log(pred + self.eps)
            * torch.pow(1 - pred, self.alpha)
            * pos_inds
        )
        neg_loss = (
            torch.log(1 - pred + self.eps)
            * torch.pow(pred, self.alpha)
            * torch.pow(1 - gt, self.beta)
            * neg_inds
        )

        if self.class_weights is not None:
            pos_loss = pos_loss * torch.tensor(self.class_weights)

        if self.mask_zero:
            valid_mask = gt.any(dim=1, keepdim=True).float()
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


class MSELoss(nn.Module):
    """Mean Squared Error (MSE) Loss module.

    Calculates the mean squared difference between predictions and ground
    truth. Optionally masks contributions where the ground truth is zero across
    channels.

    Parameters
    ----------
    mask_zero : bool, default=False
        If True, calculates the loss only over spatial locations (H, W) where
        at least one channel in the ground truth `gt` tensor is non-zero. The
        loss is then averaged over these valid locations. If False (default),
        the standard MSE over all elements is computed.
    """

    def __init__(self, mask_zero: bool = False):
        super().__init__()
        self.mask_zero = mask_zero

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the Mean Squared Error loss.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted tensor, shape `(B, C, H, W)`.
        gt : torch.Tensor
            Ground truth tensor, shape `(B, C, H, W)`.

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the calculated MSE loss.
        """
        if not self.mask_zero:
            return ((gt - pred) ** 2).mean()

        valid_mask = gt.any(dim=1, keepdim=True).float()
        return (valid_mask * ((gt - pred) ** 2)).sum() / valid_mask.sum()


class DetectionLossConfig(BaseConfig):
    """Configuration for the detection loss component.

    Attributes
    ----------
    weight : float, default=1.0
        Weighting factor for the detection loss in the combined total loss.
    focal : FocalLossConfig
        Configuration for the Focal Loss used for detection. Defaults to
        standard Focal Loss parameters (`alpha=2`, `beta=4`).
    """

    weight: float = 1.0
    focal: FocalLossConfig = Field(default_factory=FocalLossConfig)


class ClassificationLossConfig(BaseConfig):
    """Configuration for the classification loss component.

    Attributes
    ----------
    weight : float, default=2.0
        Weighting factor for the classification loss in the combined total loss.
    focal : FocalLossConfig
        Configuration for the Focal Loss used for classification. Defaults to
        standard Focal Loss parameters (`alpha=2`, `beta=4`).
    """

    weight: float = 2.0
    focal: FocalLossConfig = Field(default_factory=FocalLossConfig)


class LossConfig(BaseConfig):
    """Aggregated configuration for all loss components.

    Defines the configuration and weighting for detection, size regression,
    and classification losses used in the main `LossFunction`.

    Attributes
    ----------
    detection : DetectionLossConfig
        Configuration for the detection loss (Focal Loss).
    size : SizeLossConfig
        Configuration for the size regression loss (L1 loss).
    classification : ClassificationLossConfig
        Configuration for the classification loss (Focal Loss).
    """

    detection: DetectionLossConfig = Field(default_factory=DetectionLossConfig)
    size: SizeLossConfig = Field(default_factory=SizeLossConfig)
    classification: ClassificationLossConfig = Field(
        default_factory=ClassificationLossConfig
    )


class LossFunction(nn.Module, LossProtocol):
    """Computes the combined training loss for the BatDetect2 model.

    Aggregates individual loss functions for detection, size regression, and
    classification tasks. Calculates each component loss based on model outputs
    and ground truth targets, applies configured weights, and sums them to get
    the final total loss used for optimization. Also returns individual
    components for monitoring.

    Parameters
    ----------
    size_loss : nn.Module
        Instantiated loss module for size regression (e.g., `BBoxLoss`).
    detection_loss : nn.Module
        Instantiated loss module for detection (e.g., `FocalLoss`).
    classification_loss : nn.Module
        Instantiated loss module for classification (e.g., `FocalLoss`).
    size_weight : float, default=0.1
        Weighting factor for the size loss component.
    detection_weight : float, default=1.0
        Weighting factor for the detection loss component.
    classification_weight : float, default=2.0
        Weighting factor for the classification loss component.

    Attributes
    ----------
    size_loss_fn : nn.Module
    detection_loss_fn : nn.Module
    classification_loss_fn : nn.Module
    size_weight : float
    detection_weight : float
    classification_weight : float
    """

    def __init__(
        self,
        size_loss: nn.Module,
        detection_loss: nn.Module,
        classification_loss: nn.Module,
        size_weight: float = 0.1,
        detection_weight: float = 1.0,
        classification_weight: float = 2.0,
    ):
        super().__init__()
        self.size_loss_fn = size_loss
        self.detection_loss_fn = detection_loss
        self.classification_loss_fn = classification_loss

        self.size_weight = size_weight
        self.detection_weight = detection_weight
        self.classification_weight = classification_weight

    def forward(
        self,
        pred: ModelOutput,
        gt: TrainExample,
    ) -> Losses:
        """Calculate the combined loss and individual components.

        Parameters
        ----------
        pred: ModelOutput
            A NamedTuple containing the model's prediction tensors for the
            batch: `detection_probs`, `size_preds`, `class_probs`.
        gt: TrainExample
            A structure containing the ground truth targets for the batch,
            expected to have attributes like `detection_heatmap`,
            `size_heatmap`, and `class_heatmap` (as `torch.Tensor`).

        Returns
        -------
        Losses
            A NamedTuple containing the scalar loss values for detection, size,
            classification, and the total weighted loss.
        """
        size_loss = self.size_loss_fn(pred.size_preds, gt.size_heatmap)
        detection_loss = self.detection_loss_fn(
            pred.detection_probs,
            gt.detection_heatmap,
        )
        classification_loss = self.classification_loss_fn(
            pred.class_probs,
            gt.class_heatmap,
        )
        total_loss = (
            size_loss * self.size_weight
            + classification_loss * self.classification_weight
            + detection_loss * self.detection_weight
        )
        return Losses(
            detection=detection_loss,
            size=size_loss,
            classification=classification_loss,
            total=total_loss,
        )


def build_loss(
    config: Optional[LossConfig] = None,
    class_weights: Optional[np.ndarray] = None,
) -> nn.Module:
    """Factory function to build the main LossFunction from configuration.

    Instantiates the necessary loss components (`BBoxLoss`, `FocalLoss`) based
    on the provided `LossConfig` (or defaults) and optional `class_weights`,
    then assembles them into the main `LossFunction` module with the specified
    component weights.

    Parameters
    ----------
    config : LossConfig, optional
        Configuration object defining weights and parameters (e.g., alpha, beta
        for Focal Loss) for each loss component. If None, default settings
        from `LossConfig` and its nested configs are used.
    class_weights : np.ndarray, optional
        An array of weights for each specific class, used to adjust the
        classification loss (typically Focal Loss). If provided, this overrides
        any `class_weights` specified within `config.classification`. If None,
        weights from the config (or default of equal weights) are used.

    Returns
    -------
    LossFunction
        An initialized `LossFunction` module ready for training.
    """
    config = config or LossConfig()

    class_weights_tensor = (
        torch.tensor(class_weights) if class_weights else None
    )

    detection_loss_fn = FocalLoss(
        beta=config.detection.focal.beta,
        alpha=config.detection.focal.alpha,
        mask_zero=False,
    )

    classification_loss_fn = FocalLoss(
        beta=config.classification.focal.beta,
        alpha=config.classification.focal.alpha,
        class_weights=class_weights_tensor,
        mask_zero=True,
    )

    size_loss_fn = BBoxLoss()

    return LossFunction(
        size_loss=size_loss_fn,
        classification_loss=classification_loss_fn,
        detection_loss=detection_loss_fn,
        size_weight=config.size.weight,
        detection_weight=config.detection.weight,
        classification_weight=config.classification.weight,
    )
