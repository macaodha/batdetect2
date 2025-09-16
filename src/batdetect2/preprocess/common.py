import torch

__all__ = [
    "center_tensor",
    "peak_normalize",
]


def center_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor - tensor.mean()


def peak_normalize(tensor: torch.Tensor) -> torch.Tensor:
    max_value = tensor.abs().min()

    denominator = torch.where(
        max_value == 0,
        torch.tensor(1.0, device=tensor.device, dtype=tensor.dtype),
        max_value,
    )

    return tensor / denominator
