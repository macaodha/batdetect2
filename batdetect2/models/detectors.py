"""Assembles the complete BatDetect2 Detection Model.

This module defines the concrete `Detector` class, which implements the
`DetectionModel` interface defined in `.types`. It combines a feature
extraction backbone with specific prediction heads to create the end-to-end
neural network used for detecting bat calls, predicting their size, and
classifying them.

The primary components are:
- `Detector`: The `torch.nn.Module` subclass representing the complete model.
- `build_detector`: A factory function to conveniently construct a standard
  `Detector` instance given a backbone and the number of target classes.

This module focuses purely on the neural network architecture definition. The
logic for preprocessing inputs and postprocessing/decoding outputs resides in
the `batdetect2.preprocess` and `batdetect2.postprocess` packages, respectively.
"""

import torch

from batdetect2.models.heads import BBoxHead, ClassifierHead, DetectorHead
from batdetect2.models.types import BackboneModel, DetectionModel, ModelOutput


class Detector(DetectionModel):
    """Concrete implementation of the BatDetect2 Detection Model.

    Assembles a complete detection and classification model by combining a
    feature extraction backbone network with specific prediction heads for
    detection probability, bounding box size regression, and class
    probabilities.

    Attributes
    ----------
    backbone : BackboneModel
        The feature extraction backbone network module.
    num_classes : int
        The number of specific target classes the model predicts (derived from
        the `classifier_head`).
    classifier_head : ClassifierHead
        The prediction head responsible for generating class probabilities.
    detector_head : DetectorHead
        The prediction head responsible for generating detection probabilities.
    bbox_head : BBoxHead
        The prediction head responsible for generating bounding box size
        predictions.
    """

    backbone: BackboneModel

    def __init__(
        self,
        backbone: BackboneModel,
        classifier_head: ClassifierHead,
        detector_head: DetectorHead,
        bbox_head: BBoxHead,
    ):
        """Initialize the Detector model.

        Note: Instances are typically created using the `build_detector`
        factory function.

        Parameters
        ----------
        backbone : BackboneModel
            An initialized feature extraction backbone module (e.g., built by
            `build_backbone` from the `.backbone` module).
        classifier_head : ClassifierHead
            An initialized classification head module. The number of classes
            is inferred from this head.
        detector_head : DetectorHead
            An initialized detection head module.
        bbox_head : BBoxHead
            An initialized bounding box size prediction head module.

        Raises
        ------
        TypeError
            If the provided modules are not of the expected types.
        """
        super().__init__()

        self.backbone = backbone
        self.num_classes = classifier_head.num_classes
        self.classifier_head = classifier_head
        self.detector_head = detector_head
        self.bbox_head = bbox_head

    def forward(self, spec: torch.Tensor) -> ModelOutput:
        """Perform the forward pass of the complete detection model.

        Processes the input spectrogram through the backbone to extract
        features, then passes these features through the separate prediction
        heads to generate detection probabilities, class probabilities, and
        size predictions.

        Parameters
        ----------
        spec : torch.Tensor
            Input spectrogram tensor, typically with shape
            `(batch_size, input_channels, frequency_bins, time_bins)`. The
            shape must be compatible with the `self.backbone` input
            requirements.

        Returns
        -------
        ModelOutput
            A NamedTuple containing the four output tensors:
            - `detection_probs`: Detection probability heatmap `(B, 1, H, W)`.
            - `size_preds`: Predicted scaled size dimensions `(B, 2, H, W)`.
            - `class_probs`: Class probabilities (excluding background)
              `(B, num_classes, H, W)`.
            - `features`: Output feature map from the backbone
              `(B, C_out, H, W)`.
        """
        features = self.backbone(spec)
        detection = self.detector_head(features)
        classification = self.classifier_head(features)
        size_preds = self.bbox_head(features)
        return ModelOutput(
            detection_probs=detection,
            size_preds=size_preds,
            class_probs=classification,
            features=features,
        )


def build_detector(num_classes: int, backbone: BackboneModel) -> Detector:
    """Factory function to build a standard Detector model instance.

    Creates the standard prediction heads (`ClassifierHead`, `DetectorHead`,
    `BBoxHead`) configured appropriately based on the output channels of the
    provided `backbone` and the specified `num_classes`. It then assembles
    these components into a `Detector` model.

    Parameters
    ----------
    num_classes : int
        The number of specific target classes for the classification head
        (excluding any implicit background class). Must be positive.
    backbone : BackboneModel
        An initialized feature extraction backbone module instance. The number
        of output channels from this backbone (`backbone.out_channels`) is used
        to configure the input channels for the prediction heads.

    Returns
    -------
    Detector
        An initialized `Detector` model instance.

    Raises
    ------
    ValueError
        If `num_classes` is not positive.
    AttributeError
        If `backbone` does not have the required `out_channels` attribute.
    """
    classifier_head = ClassifierHead(
        num_classes=num_classes,
        in_channels=backbone.out_channels,
    )
    detector_head = DetectorHead(
        in_channels=backbone.out_channels,
    )
    bbox_head = BBoxHead(
        in_channels=backbone.out_channels,
    )
    return Detector(
        backbone=backbone,
        classifier_head=classifier_head,
        detector_head=detector_head,
        bbox_head=bbox_head,
    )
