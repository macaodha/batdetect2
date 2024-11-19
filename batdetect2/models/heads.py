from typing import NamedTuple

import torch
from torch import nn

__all__ = ["ClassifierHead"]


class Output(NamedTuple):
    detection: torch.Tensor
    classification: torch.Tensor


class ClassifierHead(nn.Module):
    def __init__(self, num_classes: int, in_channels: int):
        super().__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.classifier = nn.Conv2d(
            self.in_channels,
            # Add one to account for the background class
            self.num_classes + 1,
            kernel_size=1,
            padding=0,
        )

    def forward(self, features: torch.Tensor) -> Output:
        logits = self.classifier(features)
        probs = torch.softmax(logits, dim=1)
        detection_probs = probs[:, :-1].sum(dim=1, keepdim=True)
        return Output(
            detection=detection_probs,
            classification=probs[:, :-1],
        )


class BBoxHead(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.bbox = nn.Conv2d(
            self.feature_extractor.out_channels,
            2,
            kernel_size=1,
            padding=0,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.bbox(features)
