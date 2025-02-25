import torch
import torch.nn.functional as F
from torch import nn

from batdetect2.detector.model_helpers import (
    ConvBlockDownCoordF,
    ConvBlockDownStandard,
    ConvBlockUpF,
    ConvBlockUpStandard,
    SelfAttention,
)
from batdetect2.types import ModelOutput

__all__ = [
    "Net2DFast",
    "Net2DFastNoAttn",
    "Net2DFastNoCoordConv",
]


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

    def forward(self, spec: torch.Tensor) -> ModelOutput:
        # encoder
        x1 = self.conv_dn_0(spec)
        x2 = self.conv_dn_1(x1)
        x3 = self.conv_dn_2(x2)
        x3 = F.relu_(self.conv_dn_3_bn(self.conv_dn_3(x3)))

        # bottleneck
        x = F.relu_(self.conv_1d_bn(self.conv_1d(x3)))
        x = self.att(x)
        x = x.repeat([1, 1, self.bneck_height * 4, 1])

        # decoder
        x = self.conv_up_2(x + x3)
        x = self.conv_up_3(x + x2)
        x = self.conv_up_4(x + x1)

        # output
        x = F.relu_(self.conv_op_bn(self.conv_op(x)))
        cls = self.conv_classes_op(x)
        comb = torch.softmax(cls, 1)

        return ModelOutput(
            pred_det=comb[:, :-1, :, :].sum(1).unsqueeze(1),
            pred_size=F.relu(self.conv_size_op(x)),
            pred_class=comb,
            pred_class_un_norm=cls,
            features=x,
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
            num_filts // 4,
            self.num_classes + 1,
            kernel_size=1,
            padding=0,
        )

        if self.emb_dim > 0:
            self.conv_emb = nn.Conv2d(
                num_filts, self.emb_dim, kernel_size=1, padding=0
            )

    def forward(self, spec: torch.Tensor) -> ModelOutput:
        x1 = self.conv_dn_0(spec)
        x2 = self.conv_dn_1(x1)
        x3 = self.conv_dn_2(x2)
        x3 = F.relu_(self.conv_dn_3_bn(self.conv_dn_3(x3)))

        x = F.relu_(self.conv_1d_bn(self.conv_1d(x3)))
        x = x.repeat([1, 1, self.bneck_height * 4, 1])

        x = self.conv_up_2(x + x3)
        x = self.conv_up_3(x + x2)
        x = self.conv_up_4(x + x1)

        x = F.relu_(self.conv_op_bn(self.conv_op(x)))
        cls = self.conv_classes_op(x)
        comb = torch.softmax(cls, 1)

        return ModelOutput(
            pred_det=comb[:, :-1, :, :].sum(1).unsqueeze(1),
            pred_size=F.relu_(self.conv_size_op(x)),
            pred_class=comb,
            pred_class_un_norm=cls,
            features=x,
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

    def forward(self, spec: torch.Tensor) -> ModelOutput:
        x1 = self.conv_dn_0(spec)
        x2 = self.conv_dn_1(x1)
        x3 = self.conv_dn_2(x2)
        x3 = F.relu_(self.conv_dn_3_bn(self.conv_dn_3(x3)))

        x = F.relu_(self.conv_1d_bn(self.conv_1d(x3)))
        x = self.att(x)
        x = x.repeat([1, 1, self.bneck_height * 4, 1])

        x = self.conv_up_2(x + x3)
        x = self.conv_up_3(x + x2)
        x = self.conv_up_4(x + x1)

        x = F.relu_(self.conv_op_bn(self.conv_op(x)))
        cls = self.conv_classes_op(x)
        comb = torch.softmax(cls, 1)

        return ModelOutput(
            pred_det=comb[:, :-1, :, :].sum(1).unsqueeze(1),
            pred_size=F.relu_(self.conv_size_op(x)),
            pred_class=comb,
            pred_class_un_norm=cls,
            features=x,
        )
