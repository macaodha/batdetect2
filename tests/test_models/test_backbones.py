"""Tests for backbone configuration loading and the backbone registry."""

from pathlib import Path
from typing import Callable

import pytest

from batdetect2.models.backbones import (
    BackboneConfig,
    UNetBackbone,
    UNetBackboneConfig,
    backbone_registry,
    build_backbone,
    load_backbone_config,
)
from batdetect2.models.types import BackboneModel


def test_unet_backbone_config_defaults():
    """Default config has expected field values."""
    config = UNetBackboneConfig()

    assert config.name == "UNetBackbone"
    assert config.input_height == 128
    assert config.in_channels == 1


def test_unet_backbone_config_custom_fields():
    """Custom field values are stored correctly."""
    config = UNetBackboneConfig(in_channels=2, input_height=64)

    assert config.in_channels == 2
    assert config.input_height == 64


def test_unet_backbone_config_extra_fields_ignored():
    """Extra/unknown fields are silently ignored (BaseConfig behaviour)."""
    config = UNetBackboneConfig.model_validate(
        {"name": "UNetBackbone", "unknown_field": 99}
    )

    assert config.name == "UNetBackbone"
    assert not hasattr(config, "unknown_field")


def test_build_backbone_default():
    """Building with no config uses UNetBackbone defaults."""
    backbone = build_backbone()

    assert isinstance(backbone, UNetBackbone)
    assert backbone.input_height == 128


def test_build_backbone_custom_config():
    """Building with a custom config propagates input_height and in_channels."""
    config = UNetBackboneConfig(in_channels=2, input_height=64)
    backbone = build_backbone(config)

    assert isinstance(backbone, UNetBackbone)
    assert backbone.input_height == 64
    assert backbone.encoder.in_channels == 2


def test_build_backbone_returns_backbone_model():
    """build_backbone always returns a BackboneModel instance."""
    backbone = build_backbone()
    assert isinstance(backbone, BackboneModel)


def test_registry_has_unet_backbone():
    """The backbone registry has UNetBackbone registered."""
    config_types = backbone_registry.get_config_types()

    assert UNetBackboneConfig in config_types


def test_registry_config_type_is_unet_backbone_config():
    """The config type stored for UNetBackbone is UNetBackboneConfig."""
    config_type = backbone_registry.get_config_type("UNetBackbone")

    assert config_type is UNetBackboneConfig


def test_registry_build_dispatches_correctly():
    """Registry.build dispatches to UNetBackbone.from_config."""
    config = UNetBackboneConfig(input_height=128)
    backbone = backbone_registry.build(config)

    assert isinstance(backbone, UNetBackbone)


def test_registry_build_unknown_name_raises():
    """Registry.build raises NotImplementedError for an unknown config name."""

    class FakeConfig:
        name = "NonExistentBackbone"

    with pytest.raises(NotImplementedError):
        backbone_registry.build(FakeConfig())  # type: ignore[arg-type]


def test_backbone_config_validates_unet_from_dict():
    """BackboneConfig TypeAdapter resolves to UNetBackboneConfig via name."""
    from pydantic import TypeAdapter

    adapter = TypeAdapter(BackboneConfig)
    config = adapter.validate_python(
        {"name": "UNetBackbone", "input_height": 64}
    )

    assert isinstance(config, UNetBackboneConfig)
    assert config.input_height == 64


def test_backbone_config_invalid_name_raises():
    """BackboneConfig validation raises for an unknown name discriminator."""
    from pydantic import TypeAdapter, ValidationError

    adapter = TypeAdapter(BackboneConfig)
    with pytest.raises(ValidationError):
        adapter.validate_python({"name": "NonExistentBackbone"})


def test_load_backbone_config_from_yaml(
    create_temp_yaml: Callable[[str], Path],
):
    """load_backbone_config loads a UNetBackboneConfig from a YAML file."""
    yaml_content = """\
name: UNetBackbone
input_height: 64
in_channels: 2
"""
    path = create_temp_yaml(yaml_content)
    config = load_backbone_config(path)

    assert isinstance(config, UNetBackboneConfig)
    assert config.input_height == 64
    assert config.in_channels == 2


def test_load_backbone_config_with_field(
    create_temp_yaml: Callable[[str], Path],
):
    """load_backbone_config extracts a nested field before validation."""
    yaml_content = """\
model:
  name: UNetBackbone
  input_height: 32
"""
    path = create_temp_yaml(yaml_content)
    config = load_backbone_config(path, field="model")

    assert isinstance(config, UNetBackboneConfig)
    assert config.input_height == 32


def test_load_backbone_config_defaults_on_minimal_yaml(
    create_temp_yaml: Callable[[str], Path],
):
    """Minimal YAML with only name fills remaining fields with defaults."""
    yaml_content = "name: UNetBackbone\n"
    path = create_temp_yaml(yaml_content)
    config = load_backbone_config(path)

    assert isinstance(config, UNetBackboneConfig)
    assert config.input_height == UNetBackboneConfig().input_height
    assert config.in_channels == UNetBackboneConfig().in_channels


def test_load_backbone_config_extra_fields_ignored(
    create_temp_yaml: Callable[[str], Path],
):
    """Extra YAML fields are silently ignored when loading backbone config."""
    yaml_content = """\
name: UNetBackbone
input_height: 128
deprecated_field: 99
"""
    path = create_temp_yaml(yaml_content)
    config = load_backbone_config(path)

    assert isinstance(config, UNetBackboneConfig)
    assert config.input_height == 128


def test_round_trip_yaml_to_build_backbone(
    create_temp_yaml: Callable[[str], Path],
):
    """A backbone config loaded from YAML can be used directly with build_backbone."""
    yaml_content = """\
name: UNetBackbone
input_height: 128
in_channels: 1
"""
    path = create_temp_yaml(yaml_content)
    config = load_backbone_config(path)
    backbone = build_backbone(config)

    assert isinstance(backbone, UNetBackbone)
    assert backbone.input_height == 128


def test_load_backbone_config_from_example_data(example_data_dir: Path):
    """load_backbone_config loads the real example config correctly."""
    config = load_backbone_config(
        example_data_dir / "config.yaml",
        field="model.architecture",
    )

    assert isinstance(config, UNetBackboneConfig)
    assert config.input_height == 128
    assert config.in_channels == 1
