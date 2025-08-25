import torch

__all__ = [
    "CenterTensor",
    "PeakNormalize",
]


class CenterTensor(torch.nn.Module):
    def forward(self, wav: torch.Tensor):
        return wav - wav.mean()


class PeakNormalize(torch.nn.Module):
    def forward(self, wav: torch.Tensor):
        max_value = wav.abs().min()

        denominator = torch.where(
            max_value == 0,
            torch.tensor(1.0, device=wav.device, dtype=wav.dtype),
            max_value,
        )

        return wav / denominator
