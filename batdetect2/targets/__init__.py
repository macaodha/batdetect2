"""Module that helps define what the training targets are.

The goal of this module is to configure how raw sound event annotations (tags)
are processed to determine which events are relevant for training and what
specific class label each relevant event should receive. Also, define how
predicted class labels map back to the original tag system for interpretation.
"""

from batdetect2.targets.labels import (
    HeatmapsConfig,
    LabelConfig,
    generate_heatmaps,
    load_label_config,
)
from batdetect2.targets.targets import (
    TargetConfig,
    build_decoder,
    build_target_encoder,
    get_class_names,
    load_target_config,
)
from batdetect2.targets.terms import (
    TagInfo,
    TermInfo,
    call_type,
    get_tag_from_info,
    get_term_from_key,
    individual,
    register_term,
    term_registry,
)
from batdetect2.targets.transform import (
    DerivationRegistry,
    DeriveTagRule,
    MapValueRule,
    ReplaceRule,
    SoundEventTransformation,
    TransformConfig,
    build_transform_from_rule,
    build_transformation_from_config,
    derivation_registry,
    get_derivation,
    load_transformation_config,
    load_transformation_from_config,
)

__all__ = [
    "DerivationRegistry",
    "DeriveTagRule",
    "HeatmapsConfig",
    "LabelConfig",
    "MapValueRule",
    "ReplaceRule",
    "SoundEventTransformation",
    "TagInfo",
    "TargetConfig",
    "TermInfo",
    "TransformConfig",
    "build_decoder",
    "build_target_encoder",
    "build_transform_from_rule",
    "build_transformation_from_config",
    "call_type",
    "derivation_registry",
    "generate_heatmaps",
    "get_class_names",
    "get_derivation",
    "get_tag_from_info",
    "individual",
    "load_label_config",
    "load_target_config",
    "load_transformation_config",
    "load_transformation_from_config",
    "register_term",
    "term_registry",
]
