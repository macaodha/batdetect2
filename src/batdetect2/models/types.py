from typing import Any, NamedTuple, Protocol

import torch

from batdetect2.postprocess.types import PostprocessorProtocol
from batdetect2.preprocess.types import PreprocessorProtocol

__all__ = [
    "BackboneProtocol",
    "BlockProtocol",
    "BottleneckProtocol",
    "ClassifierHeadProtocol",
    "DecoderProtocol",
    "DetectorProtocol",
    "EncoderProtocol",
    "ModelOutput",
    "ModelProtocol",
    "ModuleProtocol",
    "SizeHeadProtocol",
]


class ModuleProtocol(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

    def train(self, mode: bool = True) -> torch.nn.Module: ...

    def eval(self) -> torch.nn.Module: ...

    def state_dict(
        self, *args: Any, **kwargs: Any
    ) -> dict[str, torch.Tensor]: ...

    def load_state_dict(self, *args: Any, **kwargs: Any) -> Any: ...

    def parameters(self) -> Any: ...


class BlockProtocol(ModuleProtocol, Protocol):
    in_channels: int
    out_channels: int

    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...

    def get_output_height(self, input_height: int) -> int: ...


class EncoderProtocol(ModuleProtocol, Protocol):
    in_channels: int
    out_channels: int
    input_height: int
    output_height: int

    def __call__(self, x: torch.Tensor) -> list[torch.Tensor]: ...


class BottleneckProtocol(ModuleProtocol, Protocol):
    in_channels: int
    out_channels: int
    input_height: int

    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...


class DecoderProtocol(ModuleProtocol, Protocol):
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


class BackboneProtocol(ModuleProtocol, Protocol):
    input_height: int
    out_channels: int

    def forward(self, spec: torch.Tensor) -> torch.Tensor: ...


class ClassifierHeadProtocol(ModuleProtocol, Protocol):
    num_classes: int
    in_channels: int
    class_names: list[str]

    def forward(self, features: torch.Tensor) -> torch.Tensor: ...


class SizeHeadProtocol(ModuleProtocol, Protocol):
    in_channels: int
    num_sizes: int
    dimension_names: list[str]

    def forward(self, features: torch.Tensor) -> torch.Tensor: ...


class DetectorProtocol(ModuleProtocol, Protocol):
    backbone: BackboneProtocol
    classifier_head: ClassifierHeadProtocol
    size_head: SizeHeadProtocol

    def forward(self, spec: torch.Tensor) -> ModelOutput: ...


class ModelProtocol(ModuleProtocol, Protocol):
    detector: DetectorProtocol
    preprocessor: PreprocessorProtocol
    postprocessor: PostprocessorProtocol
    class_names: list[str]
    dimension_names: list[str]

    def get_config(self) -> dict[str, Any]: ...
