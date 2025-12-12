from soundevent import data

from batdetect2.core.configs import BaseConfig, load_config
from batdetect2.models.bottleneck import (
    DEFAULT_BOTTLENECK_CONFIG,
    BottleneckConfig,
)
from batdetect2.models.decoder import (
    DEFAULT_DECODER_CONFIG,
    DecoderConfig,
)
from batdetect2.models.encoder import (
    DEFAULT_ENCODER_CONFIG,
    EncoderConfig,
)

__all__ = [
    "BackboneConfig",
    "load_backbone_config",
]


class BackboneConfig(BaseConfig):
    """Configuration for the Encoder-Decoder Backbone network.

    Aggregates configurations for the encoder, bottleneck, and decoder
    components, along with defining the input and final output dimensions
    for the complete backbone.

    Attributes
    ----------
    input_height : int, default=128
        Expected height (frequency bins) of the input spectrograms to the
        backbone. Must be positive.
    in_channels : int, default=1
        Expected number of channels in the input spectrograms (e.g., 1 for
        mono). Must be positive.
    encoder : EncoderConfig, optional
        Configuration for the encoder. If None or omitted,
        the default encoder configuration (`DEFAULT_ENCODER_CONFIG` from the
        encoder module) will be used.
    bottleneck : BottleneckConfig, optional
        Configuration for the bottleneck layer connecting encoder and decoder.
        If None or omitted, the default bottleneck configuration will be used.
    decoder : DecoderConfig, optional
        Configuration for the decoder. If None or omitted,
        the default decoder configuration (`DEFAULT_DECODER_CONFIG` from the
        decoder module) will be used.
    out_channels : int, default=32
        Desired number of channels in the final feature map output by the
        backbone. Must be positive.
    """

    input_height: int = 128
    in_channels: int = 1
    encoder: EncoderConfig = DEFAULT_ENCODER_CONFIG
    bottleneck: BottleneckConfig = DEFAULT_BOTTLENECK_CONFIG
    decoder: DecoderConfig = DEFAULT_DECODER_CONFIG
    out_channels: int = 32


def load_backbone_config(
    path: data.PathLike,
    field: str | None = None,
) -> BackboneConfig:
    """Load the backbone configuration from a file.

    Reads a configuration file (YAML) and validates it against the
    `BackboneConfig` schema, potentially extracting data from a nested field.

    Parameters
    ----------
    path : PathLike
        Path to the configuration file.
    field : str, optional
        Dot-separated path to a nested section within the file containing the
        backbone configuration (e.g., "model.backbone"). If None, the entire
        file content is used.

    Returns
    -------
    BackboneConfig
        The loaded and validated backbone configuration object.

    Raises
    ------
    FileNotFoundError
        If the config file path does not exist.
    yaml.YAMLError
        If the file content is not valid YAML.
    pydantic.ValidationError
        If the loaded config data does not conform to `BackboneConfig`.
    KeyError, TypeError
        If `field` specifies an invalid path.
    """
    return load_config(path, schema=BackboneConfig, field=field)
