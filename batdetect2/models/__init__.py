import os
from typing import Tuple, Union

import torch

from batdetect2.models.encoders import (
    Net2DFast,
    Net2DFastNoAttn,
    Net2DFastNoCoordConv,
)
from batdetect2.models.typing import DetectionModel

__all__ = [
    "load_model",
    "Net2DFast",
    "Net2DFastNoAttn",
    "Net2DFastNoCoordConv",
]

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "checkpoints",
    "Net2DFast_UK_same.pth.tar",
)


def load_model(
    model_path: str = DEFAULT_MODEL_PATH,
    load_weights: bool = True,
    device: Union[torch.device, str, None] = None,
) -> Tuple[DetectionModel, dict]:
    """Load model from file.

    Args:
        model_path (str): Path to model file. Defaults to DEFAULT_MODEL_PATH.
        load_weights (bool, optional): Load weights. Defaults to True.

    Returns:
        model, params: Model and parameters.

    Raises:
        FileNotFoundError: Model file not found.
        ValueError: Unknown model name.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isfile(model_path):
        raise FileNotFoundError("Model file not found.")

    net_params = torch.load(model_path, map_location=device)

    params = net_params["params"]

    model: DetectionModel

    if params["model_name"] == "Net2DFast":
        model = Net2DFast(
            params["num_filters"],
            num_classes=len(params["class_names"]),
            emb_dim=params["emb_dim"],
            ip_height=params["ip_height"],
            resize_factor=params["resize_factor"],
        )
    elif params["model_name"] == "Net2DFastNoAttn":
        model = Net2DFastNoAttn(
            params["num_filters"],
            num_classes=len(params["class_names"]),
            emb_dim=params["emb_dim"],
            ip_height=params["ip_height"],
            resize_factor=params["resize_factor"],
        )
    elif params["model_name"] == "Net2DFastNoCoordConv":
        model = Net2DFastNoCoordConv(
            params["num_filters"],
            num_classes=len(params["class_names"]),
            emb_dim=params["emb_dim"],
            ip_height=params["ip_height"],
            resize_factor=params["resize_factor"],
        )
    else:
        raise ValueError("Unknown model.")

    if load_weights:
        model.load_state_dict(net_params["state_dict"])

    model = model.to(device)
    model.eval()

    return model, params
