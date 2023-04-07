import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "SelfAttention",
    "ConvBlockDownCoordF",
    "ConvBlockDownStandard",
    "ConvBlockUpF",
    "ConvBlockUpStandard",
]


class SelfAttention(nn.Module):
    def __init__(self, ip_dim, att_dim):
        super(SelfAttention, self).__init__()
        # Note, does not encode position information (absolute or realtive)
        self.temperature = 1.0
        self.att_dim = att_dim
        self.key_fun = nn.Linear(ip_dim, att_dim)
        self.val_fun = nn.Linear(ip_dim, att_dim)
        self.que_fun = nn.Linear(ip_dim, att_dim)
        self.pro_fun = nn.Linear(att_dim, ip_dim)

    def forward(self, x):
        x = x.squeeze(2).permute(0, 2, 1)

        kk = torch.matmul(
            x, self.key_fun.weight.T
        ) + self.key_fun.bias.unsqueeze(0).unsqueeze(0)
        qq = torch.matmul(
            x, self.que_fun.weight.T
        ) + self.que_fun.bias.unsqueeze(0).unsqueeze(0)
        vv = torch.matmul(
            x, self.val_fun.weight.T
        ) + self.val_fun.bias.unsqueeze(0).unsqueeze(0)

        kk_qq = torch.bmm(kk, qq.permute(0, 2, 1)) / (
            self.temperature * self.att_dim
        )
        att_weights = F.softmax(
            kk_qq, 1
        )  # each col of each attention matrix sums to 1
        att = torch.bmm(vv.permute(0, 2, 1), att_weights)

        op = torch.matmul(
            att.permute(0, 2, 1), self.pro_fun.weight.T
        ) + self.pro_fun.bias.unsqueeze(0).unsqueeze(0)
        op = op.permute(0, 2, 1).unsqueeze(2)

        return op


class ConvBlockDownCoordF(nn.Module):
    def __init__(
        self, in_chn, out_chn, ip_height, k_size=3, pad_size=1, stride=1
    ):
        super(ConvBlockDownCoordF, self).__init__()
        self.coords = nn.Parameter(
            torch.linspace(-1, 1, ip_height)[None, None, ..., None],
            requires_grad=False,
        )
        self.conv = nn.Conv2d(
            in_chn + 1,
            out_chn,
            kernel_size=k_size,
            padding=pad_size,
            stride=stride,
        )
        self.conv_bn = nn.BatchNorm2d(out_chn)

    def forward(self, x):
        freq_info = self.coords.repeat(x.shape[0], 1, 1, x.shape[3])
        x = torch.cat((x, freq_info), 1)
        x = F.max_pool2d(self.conv(x), 2, 2)
        x = F.relu(self.conv_bn(x), inplace=True)
        return x


class ConvBlockDownStandard(nn.Module):
    def __init__(
        self, in_chn, out_chn, ip_height=None, k_size=3, pad_size=1, stride=1
    ):
        super(ConvBlockDownStandard, self).__init__()
        self.conv = nn.Conv2d(
            in_chn,
            out_chn,
            kernel_size=k_size,
            padding=pad_size,
            stride=stride,
        )
        self.conv_bn = nn.BatchNorm2d(out_chn)

    def forward(self, x):
        x = F.max_pool2d(self.conv(x), 2, 2)
        x = F.relu(self.conv_bn(x), inplace=True)
        return x


class ConvBlockUpF(nn.Module):
    def __init__(
        self,
        in_chn,
        out_chn,
        ip_height,
        k_size=3,
        pad_size=1,
        up_mode="bilinear",
        up_scale=(2, 2),
    ):
        super(ConvBlockUpF, self).__init__()
        self.up_scale = up_scale
        self.up_mode = up_mode
        self.coords = nn.Parameter(
            torch.linspace(-1, 1, ip_height * up_scale[0])[
                None, None, ..., None
            ],
            requires_grad=False,
        )
        self.conv = nn.Conv2d(
            in_chn + 1, out_chn, kernel_size=k_size, padding=pad_size
        )
        self.conv_bn = nn.BatchNorm2d(out_chn)

    def forward(self, x):
        op = F.interpolate(
            x,
            size=(
                x.shape[-2] * self.up_scale[0],
                x.shape[-1] * self.up_scale[1],
            ),
            mode=self.up_mode,
            align_corners=False,
        )
        freq_info = self.coords.repeat(op.shape[0], 1, 1, op.shape[3])
        op = torch.cat((op, freq_info), 1)
        op = self.conv(op)
        op = F.relu(self.conv_bn(op), inplace=True)
        return op


class ConvBlockUpStandard(nn.Module):
    def __init__(
        self,
        in_chn,
        out_chn,
        ip_height=None,
        k_size=3,
        pad_size=1,
        up_mode="bilinear",
        up_scale=(2, 2),
    ):
        super(ConvBlockUpStandard, self).__init__()
        self.up_scale = up_scale
        self.up_mode = up_mode
        self.conv = nn.Conv2d(
            in_chn, out_chn, kernel_size=k_size, padding=pad_size
        )
        self.conv_bn = nn.BatchNorm2d(out_chn)

    def forward(self, x):
        op = F.interpolate(
            x,
            size=(
                x.shape[-2] * self.up_scale[0],
                x.shape[-1] * self.up_scale[1],
            ),
            mode=self.up_mode,
            align_corners=False,
        )
        op = self.conv(op)
        op = F.relu(self.conv_bn(op), inplace=True)
        return op
