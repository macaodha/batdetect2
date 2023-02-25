from typing import NamedTuple, Optional

import torch
import torch.fft
import torch.nn.functional as F
from torch import nn

from bat_detect.detector.model_helpers import (
    ConvBlockDownCoordF,
    ConvBlockDownStandard,
    ConvBlockUpF,
    ConvBlockUpStandard,
    SelfAttention,
)

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol

__all__ = [
    "Net2DFast",
    "Net2DFastNoAttn",
    "Net2DFastNoCoordConv",
    "ModelOutput",
    "DetectionModel",
]


class ModelOutput(NamedTuple):
    """Output of the detection model."""

    pred_det: torch.Tensor
    """Tensor with predict detection probabilities."""

    pred_size: torch.Tensor
    """Tensor with predicted bounding box sizes."""

    pred_class: torch.Tensor
    """Tensor with predicted class probabilities."""

    pred_class_un_norm: torch.Tensor
    """Tensor with predicted class probabilities before softmax."""

    pred_emb: Optional[torch.Tensor]
    """Tensor with embeddings."""

    features: Optional[torch.Tensor]
    """Tensor with intermediate features."""


class DetectionModel(Protocol):
    """Protocol for detection models.

    This protocol is used to define the interface for the detection models.
    This allows us to use the same code for training and inference, even
    though the models are different.
    """

    num_classes: int
    """Number of classes the model can classify."""

    emb_dim: int
    """Dimension of the embedding vector."""

    num_filts: int
    """Number of filters in the model."""

    resize_factor: float
    """Factor by which the input is resized."""

    ip_height: int
    """Height of the input image."""

    def forward(
        self,
        ip: torch.Tensor,
        return_feats: bool = False,
    ) -> ModelOutput:
        """Forward pass of the model.

        When `return_feats` is `True`, the model should return the
        intermediate features of the model.
        """

    def __call__(
        self,
        ip: torch.Tensor,
        return_feats: bool = False,
    ) -> ModelOutput:
        """Forward pass of the model.

        When `return_feats` is `True`, the model should return the
        int
        """


class Net2DFast(nn.Module):
    def __init__(
        self,
        num_filts,
        num_classes=0,
        emb_dim=0,
        ip_height=128,
        resize_factor=0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.num_filts = num_filts
        self.resize_factor = resize_factor
        self.ip_height_rs = ip_height
        self.bneck_height = self.ip_height_rs // 32

        # encoder
        self.conv_dn_0 = ConvBlockDownCoordF(
            1,
            num_filts // 4,
            self.ip_height_rs,
            k_size=3,
            pad_size=1,
            stride=1,
        )
        self.conv_dn_1 = ConvBlockDownCoordF(
            num_filts // 4,
            num_filts // 2,
            self.ip_height_rs // 2,
            k_size=3,
            pad_size=1,
            stride=1,
        )
        self.conv_dn_2 = ConvBlockDownCoordF(
            num_filts // 2,
            num_filts,
            self.ip_height_rs // 4,
            k_size=3,
            pad_size=1,
            stride=1,
        )
        self.conv_dn_3 = nn.Conv2d(num_filts, num_filts * 2, 3, padding=1)
        self.conv_dn_3_bn = nn.BatchNorm2d(num_filts * 2)

        # bottleneck
        self.conv_1d = nn.Conv2d(
            num_filts * 2,
            num_filts * 2,
            (self.ip_height_rs // 8, 1),
            padding=0,
        )
        self.conv_1d_bn = nn.BatchNorm2d(num_filts * 2)
        self.att = SelfAttention(num_filts * 2, num_filts * 2)

        # decoder
        self.conv_up_2 = ConvBlockUpF(
            num_filts * 2, num_filts // 2, self.ip_height_rs // 8
        )
        self.conv_up_3 = ConvBlockUpF(
            num_filts // 2, num_filts // 4, self.ip_height_rs // 4
        )
        self.conv_up_4 = ConvBlockUpF(
            num_filts // 4, num_filts // 4, self.ip_height_rs // 2
        )

        # output
        # +1 to include background class for class output
        self.conv_op = nn.Conv2d(
            num_filts // 4, num_filts // 4, kernel_size=3, padding=1
        )
        self.conv_op_bn = nn.BatchNorm2d(num_filts // 4)
        self.conv_size_op = nn.Conv2d(
            num_filts // 4, 2, kernel_size=1, padding=0
        )
        self.conv_classes_op = nn.Conv2d(
            num_filts // 4, self.num_classes + 1, kernel_size=1, padding=0
        )

        if self.emb_dim > 0:
            self.conv_emb = nn.Conv2d(
                num_filts, self.emb_dim, kernel_size=1, padding=0
            )

    def forward(self, ip, return_feats=False) -> ModelOutput:

        # encoder
        x1 = self.conv_dn_0(ip)
        x2 = self.conv_dn_1(x1)
        x3 = self.conv_dn_2(x2)
        x3 = F.relu(self.conv_dn_3_bn(self.conv_dn_3(x3)), inplace=True)

        # bottleneck
        x = F.relu(self.conv_1d_bn(self.conv_1d(x3)), inplace=True)
        x = self.att(x)
        x = x.repeat([1, 1, self.bneck_height * 4, 1])

        # decoder
        x = self.conv_up_2(x + x3)
        x = self.conv_up_3(x + x2)
        x = self.conv_up_4(x + x1)

        # output
        x = F.relu(self.conv_op_bn(self.conv_op(x)), inplace=True)
        cls = self.conv_classes_op(x)
        comb = torch.softmax(cls, 1)

        return ModelOutput(
            pred_det=comb[:, :-1, :, :].sum(1).unsqueeze(1),
            pred_size=F.relu(self.conv_size_op(x), inplace=True),
            pred_class=comb,
            pred_class_un_norm=cls,
            pred_emb=self.conv_emb(x) if self.emb_dim > 0 else None,
            features=x if return_feats else None,
        )


class Net2DFastNoAttn(nn.Module):
    def __init__(
        self,
        num_filts,
        num_classes=0,
        emb_dim=0,
        ip_height=128,
        resize_factor=0.5,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.num_filts = num_filts
        self.resize_factor = resize_factor
        self.ip_height_rs = ip_height
        self.bneck_height = self.ip_height_rs // 32

        self.conv_dn_0 = ConvBlockDownCoordF(
            1,
            num_filts // 4,
            self.ip_height_rs,
            k_size=3,
            pad_size=1,
            stride=1,
        )
        self.conv_dn_1 = ConvBlockDownCoordF(
            num_filts // 4,
            num_filts // 2,
            self.ip_height_rs // 2,
            k_size=3,
            pad_size=1,
            stride=1,
        )
        self.conv_dn_2 = ConvBlockDownCoordF(
            num_filts // 2,
            num_filts,
            self.ip_height_rs // 4,
            k_size=3,
            pad_size=1,
            stride=1,
        )
        self.conv_dn_3 = nn.Conv2d(num_filts, num_filts * 2, 3, padding=1)
        self.conv_dn_3_bn = nn.BatchNorm2d(num_filts * 2)

        self.conv_1d = nn.Conv2d(
            num_filts * 2,
            num_filts * 2,
            (self.ip_height_rs // 8, 1),
            padding=0,
        )
        self.conv_1d_bn = nn.BatchNorm2d(num_filts * 2)

        self.conv_up_2 = ConvBlockUpF(
            num_filts * 2, num_filts // 2, self.ip_height_rs // 8
        )
        self.conv_up_3 = ConvBlockUpF(
            num_filts // 2, num_filts // 4, self.ip_height_rs // 4
        )
        self.conv_up_4 = ConvBlockUpF(
            num_filts // 4, num_filts // 4, self.ip_height_rs // 2
        )

        # output
        # +1 to include background class for class output
        self.conv_op = nn.Conv2d(
            num_filts // 4, num_filts // 4, kernel_size=3, padding=1
        )
        self.conv_op_bn = nn.BatchNorm2d(num_filts // 4)
        self.conv_size_op = nn.Conv2d(
            num_filts // 4, 2, kernel_size=1, padding=0
        )
        self.conv_classes_op = nn.Conv2d(
            num_filts // 4, self.num_classes + 1, kernel_size=1, padding=0
        )

        if self.emb_dim > 0:
            self.conv_emb = nn.Conv2d(
                num_filts, self.emb_dim, kernel_size=1, padding=0
            )

    def forward(self, ip, return_feats=False) -> ModelOutput:
        x1 = self.conv_dn_0(ip)
        x2 = self.conv_dn_1(x1)
        x3 = self.conv_dn_2(x2)
        x3 = F.relu(self.conv_dn_3_bn(self.conv_dn_3(x3)), inplace=True)

        x = F.relu(self.conv_1d_bn(self.conv_1d(x3)), inplace=True)
        x = x.repeat([1, 1, self.bneck_height * 4, 1])

        x = self.conv_up_2(x + x3)
        x = self.conv_up_3(x + x2)
        x = self.conv_up_4(x + x1)

        x = F.relu(self.conv_op_bn(self.conv_op(x)), inplace=True)
        cls = self.conv_classes_op(x)
        comb = torch.softmax(cls, 1)

        return ModelOutput(
            pred_det=comb[:, :-1, :, :].sum(1).unsqueeze(1),
            pred_size=F.relu(self.conv_size_op(x), inplace=True),
            pred_class=comb,
            pred_class_un_norm=cls,
            pred_emb=self.conv_emb(x) if self.emb_dim > 0 else None,
            features=x if return_feats else None,
        )


class Net2DFastNoCoordConv(nn.Module):
    def __init__(
        self,
        num_filts,
        num_classes=0,
        emb_dim=0,
        ip_height=128,
        resize_factor=0.5,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.num_filts = num_filts
        self.resize_factor = resize_factor
        self.ip_height_rs = ip_height
        self.bneck_height = self.ip_height_rs // 32

        self.conv_dn_0 = ConvBlockDownStandard(
            1,
            num_filts // 4,
            self.ip_height_rs,
            k_size=3,
            pad_size=1,
            stride=1,
        )
        self.conv_dn_1 = ConvBlockDownStandard(
            num_filts // 4,
            num_filts // 2,
            self.ip_height_rs // 2,
            k_size=3,
            pad_size=1,
            stride=1,
        )
        self.conv_dn_2 = ConvBlockDownStandard(
            num_filts // 2,
            num_filts,
            self.ip_height_rs // 4,
            k_size=3,
            pad_size=1,
            stride=1,
        )
        self.conv_dn_3 = nn.Conv2d(num_filts, num_filts * 2, 3, padding=1)
        self.conv_dn_3_bn = nn.BatchNorm2d(num_filts * 2)

        self.conv_1d = nn.Conv2d(
            num_filts * 2,
            num_filts * 2,
            (self.ip_height_rs // 8, 1),
            padding=0,
        )
        self.conv_1d_bn = nn.BatchNorm2d(num_filts * 2)

        self.att = SelfAttention(num_filts * 2, num_filts * 2)

        self.conv_up_2 = ConvBlockUpStandard(
            num_filts * 2, num_filts // 2, self.ip_height_rs // 8
        )
        self.conv_up_3 = ConvBlockUpStandard(
            num_filts // 2, num_filts // 4, self.ip_height_rs // 4
        )
        self.conv_up_4 = ConvBlockUpStandard(
            num_filts // 4, num_filts // 4, self.ip_height_rs // 2
        )

        # output
        # +1 to include background class for class output
        self.conv_op = nn.Conv2d(
            num_filts // 4, num_filts // 4, kernel_size=3, padding=1
        )
        self.conv_op_bn = nn.BatchNorm2d(num_filts // 4)
        self.conv_size_op = nn.Conv2d(
            num_filts // 4, 2, kernel_size=1, padding=0
        )
        self.conv_classes_op = nn.Conv2d(
            num_filts // 4, self.num_classes + 1, kernel_size=1, padding=0
        )

        if self.emb_dim > 0:
            self.conv_emb = nn.Conv2d(
                num_filts, self.emb_dim, kernel_size=1, padding=0
            )

    def forward(self, ip, return_feats=False) -> ModelOutput:

        x1 = self.conv_dn_0(ip)
        x2 = self.conv_dn_1(x1)
        x3 = self.conv_dn_2(x2)
        x3 = F.relu(self.conv_dn_3_bn(self.conv_dn_3(x3)), inplace=True)

        x = F.relu(self.conv_1d_bn(self.conv_1d(x3)), inplace=True)
        x = self.att(x)
        x = x.repeat([1, 1, self.bneck_height * 4, 1])

        x = self.conv_up_2(x + x3)
        x = self.conv_up_3(x + x2)
        x = self.conv_up_4(x + x1)

        x = F.relu(self.conv_op_bn(self.conv_op(x)), inplace=True)
        cls = self.conv_classes_op(x)
        comb = torch.softmax(cls, 1)

        return ModelOutput(
            pred_det=comb[:, :-1, :, :].sum(1).unsqueeze(1),
            pred_size=F.relu(self.conv_size_op(x), inplace=True),
            pred_class=comb,
            pred_class_un_norm=cls,
            pred_emb=self.conv_emb(x) if self.emb_dim > 0 else None,
            features=x if return_feats else None,
        )
