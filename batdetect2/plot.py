"""Plot functions to visualize detections and spectrograms."""

from typing import List, Optional, Tuple, Union, cast

import numpy as np
import torch
from matplotlib import axes, patches
import matplotlib.ticker as tick
from matplotlib import pyplot as plt

from batdetect2.detector.parameters import DEFAULT_PROCESSING_CONFIGURATIONS
from batdetect2.types import (
    Annotation,
    ProcessingConfiguration,
    SpectrogramParameters,
)

__all__ = [
    "spectrogram_with_detections",
    "detection",
    "detections",
    "spectrogram",
]


def spectrogram(
    spec: Union[torch.Tensor, np.ndarray],
    config: Optional[ProcessingConfiguration] = None,
    ax: Optional[axes.Axes] = None,
    figsize: Optional[Tuple[int, int]] = None,
    cmap: str = "plasma",
    start_time: float = 0,
) -> axes.Axes:
    """Plot a spectrogram.

    Parameters
    ----------
    spec (Union[torch.Tensor, np.ndarray]): Spectrogram to plot.
    config (Optional[ProcessingConfiguration], optional): Configuration
        used to compute the spectrogram. Defaults to None. If None,
        the default configuration will be used.
    ax (Optional[axes.Axes], optional): Matplotlib axes object.
        Defaults to None. if provided, the spectrogram will be plotted
        on this axes.
    figsize (Optional[Tuple[int, int]], optional): Figure size.
        Defaults to None. If `ax` is None, this will be used to create
        a new figure of the given size.
    cmap (str, optional): Colormap to use. Defaults to "plasma".
    start_time (float, optional): Start time of the spectrogram.
        Defaults to 0. This is useful if plotting a spectrogram
        of a segment of a longer audio file.

    Returns
    -------
    axes.Axes: Matplotlib axes object.

    Raises
    ------
    ValueError: If the spectrogram is not of
        shape (1, T, F), (1, 1, T, F) or (T, F)
    """
    # Convert to numpy array if needed
    if isinstance(spec, torch.Tensor):
        spec = spec.detach().cpu().numpy()

    # Remove batch and channel dimensions if present
    spec = spec.squeeze()

    if spec.ndim != 2:
        raise ValueError(
            f"Expected a 2D tensor, got {spec.ndim}D tensor instead."
        )

    # Get config
    if config is None:
        config = DEFAULT_PROCESSING_CONFIGURATIONS.copy()

    # Frequency axis is reversed
    spec = spec[::-1, :]

    if ax is None:
        # Using cast to fix typing. pyplot subplots is not
        # correctly typed.
        ax = cast(axes.Axes, plt.subplots(figsize=figsize)[1])

    # compute extent
    extent = _compute_spec_extent(spec.shape, config)

    # add start time
    extent = (extent[0] + start_time, extent[1] + start_time, *extent[2:])

    ax.imshow(spec, aspect="auto", origin="lower", cmap=cmap, extent=extent)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")

    def y_fmt(x, _):
        return f"{x / 1000:.0f}"

    ax.yaxis.set_major_formatter(tick.FuncFormatter(y_fmt))

    return ax


def spectrogram_with_detections(
    spec: Union[torch.Tensor, np.ndarray],
    dets: List[Annotation],
    config: Optional[ProcessingConfiguration] = None,
    ax: Optional[axes.Axes] = None,
    figsize: Optional[Tuple[int, int]] = None,
    cmap: str = "plasma",
    with_names: bool = True,
    start_time: float = 0,
    **kwargs,
) -> axes.Axes:
    """Plot a spectrogram with detections.

    Parameters
    ----------
    spec (Union[torch.Tensor, np.ndarray]): Spectrogram to plot.
    detections (List[Annotation]): List of detections.
    config (Optional[ProcessingConfiguration], optional): Configuration
        used to compute the spectrogram. Defaults to None. If None,
        the default configuration will be used.
    ax (Optional[axes.Axes], optional): Matplotlib axes object.
        Defaults to None. if provided, the spectrogram will be plotted
        on this axes.
    figsize (Optional[Tuple[int, int]], optional): Figure size.
        Defaults to None. If `ax` is None, this will be used to create
        a new figure of the given size.
    cmap (str, optional): Colormap to use. Defaults to "plasma".
    with_names (bool, optional): Whether to plot the name of the
        predicted class next to the detection. Defaults to True.
    start_time (float, optional): Start time of the spectrogram.
        Defaults to 0. This is useful if plotting a spectrogram
        of a segment of a longer audio file.
    **kwargs: Additional keyword arguments to pass to the
        `plot.detections` function.

    Returns
    -------
    axes.Axes: Matplotlib axes object.

    Raises
    ------
    ValueError: If the spectrogram is not of shape (1, F, T),
        (1, 1, F, T) or (F, T).
    """
    ax = spectrogram(
        spec,
        start_time=start_time,
        config=config,
        cmap=cmap,
        ax=ax,
        figsize=figsize,
    )

    ax = detections(
        dets,
        ax=ax,
        figsize=figsize,
        with_names=with_names,
        **kwargs,
    )

    return ax


def detections(
    dets: List[Annotation],
    ax: Optional[axes.Axes] = None,
    figsize: Optional[Tuple[int, int]] = None,
    with_names: bool = True,
    **kwargs,
) -> axes.Axes:
    """Plot a list of detections.

    Parameters
    ----------
    dets (List[Annotation]): List of detections.
    ax (Optional[axes.Axes], optional): Matplotlib axes object.
        Defaults to None. if provided, the spectrogram will be plotted
        on this axes.
    figsize (Optional[Tuple[int, int]], optional): Figure size.
        Defaults to None. If `ax` is None, this will be used to create
        a new figure of the given size.
    with_names (bool, optional): Whether to plot the name of the
        predicted class next to the detection. Defaults to True.
    **kwargs: Additional keyword arguments to pass to the
        `plot.detection` function.

    Returns
    -------
    axes.Axes: Matplotlib axes object on which the detections
        were plotted.
    """
    if ax is None:
        # Using cast to fix typing. pyplot subplots is not
        # correctly typed.
        ax = cast(axes.Axes, plt.subplots(figsize=figsize)[1])

    for det in dets:
        ax = detection(
            det,
            ax=ax,
            figsize=figsize,
            with_name=with_names,
            **kwargs,
        )

    return ax


def detection(
    det: Annotation,
    ax: Optional[axes.Axes] = None,
    figsize: Optional[Tuple[int, int]] = None,
    linewidth: float = 1,
    edgecolor: str = "w",
    facecolor: str = "none",
    with_name: bool = True,
) -> axes.Axes:
    """Plot a single detection.

    Parameters
    ----------
    det (Annotation): Detection to plot.
    ax (Optional[axes.Axes], optional): Matplotlib axes object. Defaults
        to None. If provided, the spectrogram will be plotted on this axes.
    figsize (Optional[Tuple[int, int]], optional): Figure size. Defaults
        to None. If `ax` is None, this will be used to create a new figure
        of the given size.
    linewidth (float, optional): Line width of the detection. 
        Defaults to 1.
    edgecolor (str, optional): Edge color of the detection. 
        Defaults to "w", i.e. white.
    facecolor (str, optional): Face color of the detection. 
        Defaults to "none", i.e. transparent.
    with_name (bool, optional): Whether to plot the name of the
        predicted class next to the detection. Defaults to True.

    Returns
    -------
    axes.Axes: Matplotlib axes object on which the detection
        was plotted.
    """
    if ax is None:
        # Using cast to fix typing. pyplot subplots is not
        # correctly typed.
        ax = cast(axes.Axes, plt.subplots(figsize=figsize)[1])

    # Plot detection
    rect = patches.Rectangle(
        (det["start_time"], det["low_freq"]),
        det["end_time"] - det["start_time"],
        det["high_freq"] - det["low_freq"],
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        alpha=det.get("det_prob", 1),
    )
    ax.add_patch(rect)

    if with_name:
        # Add class label
        txt = " ".join([sp[:3] for sp in det["class"].split(" ")])
        font_info = {
            "color": edgecolor,
            "size": 10,
            "weight": "bold",
            "alpha": rect.get_alpha(),
        }
        y_pos = rect.get_xy()[1] + rect.get_height()
        ax.text(rect.get_xy()[0], y_pos, txt, fontdict=font_info)

    return ax


def _compute_spec_extent(
    shape: Tuple[int, int],
    params: SpectrogramParameters,
) -> Tuple[float, float, float, float]:
    """Compute the extent of a spectrogram.

    Parameters
    ----------
    shape (Tuple[int, int]): Shape of the spectrogram.
        The first dimension is the frequency axis and the second
        dimension is the time axis.
    params (SpectrogramParameters): Spectrogram parameters.
        Should be the same as the ones used to compute the spectrogram.

    Returns
    -------
    Tuple[float, float, float, float]: Extent of the spectrogram.
        The first two values are the minimum and maximum time values,
        the last two values are the minimum and maximum frequency values.
    """
    fft_win_length = params["fft_win_length"]
    fft_overlap = params["fft_overlap"]
    max_freq = params["max_freq"]
    min_freq = params["min_freq"]

    # compute duration based on spectrogram parameters
    duration = (shape[1] + 1) * (fft_win_length * (1 - fft_overlap))

    # If the spectrogram is not resized, the duration is correct
    # but if it is resized, the duration needs to be adjusted
    resize_factor = params["resize_factor"]
    spec_height = params["spec_height"]
    if spec_height * resize_factor == shape[0]:
        duration = duration / resize_factor

    return 0, duration, min_freq, max_freq
