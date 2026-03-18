"""Assembles the complete BatDetect2 detection model.

This module defines the ``Detector`` class, which combines a backbone
feature extractor with prediction heads for detection, classification, and
bounding-box size regression.

Components
----------
- ``Detector`` – the ``torch.nn.Module`` that wires together a backbone
  (``BackboneModel``) with a ``ClassifierHead`` and a ``BBoxHead`` to
  produce a ``ModelOutput`` tuple from an input spectrogram.
- ``build_detector`` – factory function that builds a ready-to-use
  ``Detector`` from a backbone configuration and a target class count.

Note that ``Detector`` operates purely on spectrogram tensors; raw audio
preprocessing and output postprocessing are handled by
``batdetect2.preprocess`` and ``batdetect2.postprocess`` respectively.
"""

import torch
from loguru import logger

from batdetect2.models.backbones import (
    BackboneConfig,
    UNetBackboneConfig,
    build_backbone,
)
from batdetect2.models.heads import BBoxHead, ClassifierHead
from batdetect2.models.types import BackboneModel, DetectionModel, ModelOutput

__all__ = [
    "Detector",
    "build_detector",
]


class Detector(DetectionModel):
    """Complete BatDetect2 detection and classification model.

    Combines a backbone feature extractor with two prediction heads:

    - ``ClassifierHead``: predicts per-class probabilities at each
      time–frequency location.
    - ``BBoxHead``: predicts call duration and bandwidth at each location.

    The detection probability map is derived from the class probabilities by
    summing across the class dimension (i.e. the probability that *any* class
    is present), rather than from a separate detection head.

    Instances are typically created via ``build_detector``.

    Attributes
    ----------
    backbone : BackboneModel
        The feature extraction backbone.
    num_classes : int
        Number of target classes (inferred from the classifier head).
    classifier_head : ClassifierHead
        Produces per-class probability maps from backbone features.
    bbox_head : BBoxHead
        Produces duration and bandwidth predictions from backbone features.
    """

    backbone: BackboneModel

    def __init__(
        self,
        backbone: BackboneModel,
        classifier_head: ClassifierHead,
        bbox_head: BBoxHead,
    ):
        """Initialise the Detector model.

        This constructor is typically called by the ``build_detector``
        factory function.

        Parameters
        ----------
        backbone : BackboneModel
            An initialised backbone module (e.g. built by
            ``build_backbone``).
        classifier_head : ClassifierHead
            An initialised classification head. The ``num_classes``
            attribute is read from this head.
        bbox_head : BBoxHead
            An initialised bounding-box size prediction head.
        """
        super().__init__()

        self.backbone = backbone
        self.num_classes = classifier_head.num_classes
        self.classifier_head = classifier_head
        self.bbox_head = bbox_head

    def forward(self, spec: torch.Tensor) -> ModelOutput:
        """Run the complete detection model on an input spectrogram.

        Passes the spectrogram through the backbone to produce a feature
        map, then applies the classifier and bounding-box heads. The
        detection probability map is derived by summing the per-class
        probability maps across the class dimension; no separate detection
        head is used.

        Parameters
        ----------
        spec : torch.Tensor
            Input spectrogram tensor, shape
            ``(batch_size, channels, frequency_bins, time_bins)``.

        Returns
        -------
        ModelOutput
            A named tuple with four fields:

            - ``detection_probs`` – ``(B, 1, H, W)`` – probability that a
              call of any class is present at each location. Derived by
              summing ``class_probs`` over the class dimension.
            - ``size_preds`` – ``(B, 2, H, W)`` – scaled duration (channel
              0) and bandwidth (channel 1) predictions at each location.
            - ``class_probs`` – ``(B, num_classes, H, W)`` – per-class
              probabilities at each location.
            - ``features`` – ``(B, C_out, H, W)`` – raw backbone feature
              map.
        """
        features = self.backbone(spec)
        classification = self.classifier_head(features)
        detection = classification.sum(dim=1, keepdim=True)
        size_preds = self.bbox_head(features)
        return ModelOutput(
            detection_probs=detection,
            size_preds=size_preds,
            class_probs=classification,
            features=features,
        )


def build_detector(
    num_classes: int,
    config: BackboneConfig | None = None,
    backbone: BackboneModel | None = None,
) -> DetectionModel:
    """Build a complete BatDetect2 detection model.

    Constructs a backbone from ``config``, attaches a ``ClassifierHead``
    and a ``BBoxHead`` sized to the backbone's output channel count, and
    returns them wrapped in a ``Detector``.

    Parameters
    ----------
    num_classes : int
        Number of target bat species or call types to predict. Must be
        positive.
    config : BackboneConfig, optional
        Backbone architecture configuration. Defaults to
        ``UNetBackboneConfig()`` (the standard BatDetect2 architecture) if
        not provided.

    Returns
    -------
    DetectionModel
        An initialised ``Detector`` instance ready for training or
        inference.

    Raises
    ------
    ValueError
        If ``num_classes`` is not positive, or if the backbone
        configuration is invalid.
    """
    if backbone is None:
        config = config or UNetBackboneConfig()
        logger.opt(lazy=True).debug(
            "Building model with config: \n{}",
            lambda: config.to_yaml_string(),  # type: ignore
        )
        backbone = build_backbone(config=config)

    classifier_head = ClassifierHead(
        num_classes=num_classes,
        in_channels=backbone.out_channels,
    )
    bbox_head = BBoxHead(
        in_channels=backbone.out_channels,
    )
    return Detector(
        backbone=backbone,
        classifier_head=classifier_head,
        bbox_head=bbox_head,
    )
