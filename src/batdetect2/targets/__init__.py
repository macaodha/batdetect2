"""BatDetect2 Target Definition system."""

from batdetect2.targets.classes import (
    TargetClassConfig,
    build_sound_event_decoder,
    build_sound_event_encoder,
    get_class_names_from_config,
)
from batdetect2.targets.config import TargetConfig
from batdetect2.targets.rois import (
    AnchorBBoxMapperConfig,
    ROIMapperConfig,
    build_roi_mapper,
)
from batdetect2.targets.targets import (
    Targets,
    build_targets,
    iterate_encoded_sound_events,
    load_targets,
)
from batdetect2.targets.terms import (
    call_type,
    data_source,
    generic_class,
    individual,
)
from batdetect2.targets.types import (
    Position,
    ROITargetMapper,
    Size,
    SoundEventDecoder,
    SoundEventEncoder,
    TargetProtocol,
)

__all__ = [
    "AnchorBBoxMapperConfig",
    "Position",
    "ROIMapperConfig",
    "ROITargetMapper",
    "Size",
    "SoundEventDecoder",
    "SoundEventEncoder",
    "SoundEventFilter",
    "TargetClassConfig",
    "TargetConfig",
    "TargetProtocol",
    "Targets",
    "build_roi_mapper",
    "build_sound_event_decoder",
    "build_sound_event_encoder",
    "build_targets",
    "call_type",
    "data_source",
    "generic_class",
    "get_class_names_from_config",
    "individual",
    "iterate_encoded_sound_events",
    "load_targets",
]
