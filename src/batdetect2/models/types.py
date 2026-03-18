from abc import ABC, abstractmethod
from typing import NamedTuple, Protocol

import torch

__all__ = [
    "BackboneModel",
    "BlockProtocol",
    "BottleneckProtocol",
    "DecoderProtocol",
    "DetectionModel",
    "EncoderDecoderModel",
    "EncoderProtocol",
    "ModelOutput",
]


class BlockProtocol(Protocol):
    in_channels: int
    out_channels: int

    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...

    def get_output_height(self, input_height: int) -> int: ...


class EncoderProtocol(Protocol):
    in_channels: int
    out_channels: int
    input_height: int
    output_height: int

    def __call__(self, x: torch.Tensor) -> list[torch.Tensor]: ...


class BottleneckProtocol(Protocol):
    in_channels: int
    out_channels: int
    input_height: int

    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...


class DecoderProtocol(Protocol):
    in_channels: int
    out_channels: int
    input_height: int
    output_height: int
    depth: int

    def __call__(
        self,
        x: torch.Tensor,
        residuals: list[torch.Tensor],
    ) -> torch.Tensor: ...


class ModelOutput(NamedTuple):
    detection_probs: torch.Tensor
    size_preds: torch.Tensor
    class_probs: torch.Tensor
    features: torch.Tensor


class BackboneModel(ABC, torch.nn.Module):
    input_height: int
    out_channels: int

    @abstractmethod
    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class EncoderDecoderModel(BackboneModel):
    bottleneck_channels: int

    @abstractmethod
    def encode(self, spec: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def decode(self, encoded: torch.Tensor) -> torch.Tensor: ...


class DetectionModel(ABC, torch.nn.Module):
    @abstractmethod
    def forward(self, spec: torch.Tensor) -> ModelOutput: ...
