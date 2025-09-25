from typing import Dict, List, Optional

from pydantic import Field, PrivateAttr, computed_field, model_validator
from soundevent import data

from batdetect2.core.configs import BaseConfig
from batdetect2.data.conditions import (
    AllOfConfig,
    HasAllTagsConfig,
    HasAnyTagConfig,
    HasTagConfig,
    NotConfig,
    SoundEventCondition,
    SoundEventConditionConfig,
    build_sound_event_condition,
)
from batdetect2.targets.rois import ROIMapperConfig
from batdetect2.targets.terms import call_type, generic_class
from batdetect2.typing.targets import SoundEventDecoder, SoundEventEncoder

__all__ = [
    "build_sound_event_decoder",
    "build_sound_event_encoder",
    "get_class_names_from_config",
]


class TargetClassConfig(BaseConfig):
    """Defines a target class of sound events."""

    name: str

    condition_input: Optional[SoundEventConditionConfig] = Field(
        alias="match_if",
        default=None,
    )

    tags: Optional[List[data.Tag]] = Field(default=None, exclude=True)

    assign_tags: List[data.Tag] = Field(default_factory=list)

    roi: Optional[ROIMapperConfig] = None

    _match_if: SoundEventConditionConfig = PrivateAttr()

    @model_validator(mode="after")
    def _process_tags(self) -> "TargetClassConfig":
        if self.tags and self.condition_input:
            raise ValueError("Use either 'tags' or 'match_if', not both.")

        if self.condition_input is not None:
            self._match_if = self.condition_input
            return self

        if self.tags is None:
            raise ValueError(
                f"Class '{self.name}' must have a 'tags' or 'match_if' rule."
            )

        self._match_if = HasAllTagsConfig(tags=self.tags)

        if not self.assign_tags:
            self.assign_tags = self.tags

        return self

    @computed_field
    @property
    def match_if(self) -> SoundEventConditionConfig:
        return self._match_if


DEFAULT_DETECTION_CLASS = TargetClassConfig(
    name="bat",
    match_if=AllOfConfig(
        conditions=[
            HasTagConfig(tag=data.Tag(term=call_type, value="Echolocation")),
            NotConfig(
                condition=HasAnyTagConfig(
                    tags=[
                        data.Tag(term=call_type, value="Feeding"),
                        data.Tag(term=call_type, value="Social"),
                        data.Tag(term=call_type, value="Unknown"),
                        data.Tag(term=generic_class, value="Unknown"),
                        data.Tag(term=generic_class, value="Not Bat"),
                        data.Tag(term=call_type, value="Not Bat"),
                    ]
                )
            ),
        ]
    ),
    assign_tags=[
        data.Tag(term=call_type, value="Echolocation"),
        data.Tag(key="order", value="Chiroptera"),
    ],
)


DEFAULT_CLASSES = [
    TargetClassConfig(
        name="barbar",
        tags=[data.Tag(key="class", value="Barbastella barbastellus")],
    ),
    TargetClassConfig(
        name="eptser",
        tags=[data.Tag(key="class", value="Eptesicus serotinus")],
    ),
    TargetClassConfig(
        name="myoalc",
        tags=[data.Tag(key="class", value="Myotis alcathoe")],
    ),
    TargetClassConfig(
        name="myobec",
        tags=[data.Tag(key="class", value="Myotis bechsteinii")],
    ),
    TargetClassConfig(
        name="myobra",
        tags=[data.Tag(key="class", value="Myotis brandtii")],
    ),
    TargetClassConfig(
        name="myodau",
        tags=[data.Tag(key="class", value="Myotis daubentonii")],
    ),
    TargetClassConfig(
        name="myomys",
        tags=[data.Tag(key="class", value="Myotis mystacinus")],
    ),
    TargetClassConfig(
        name="myonat",
        tags=[data.Tag(key="class", value="Myotis nattereri")],
    ),
    TargetClassConfig(
        name="nyclei",
        tags=[data.Tag(key="class", value="Nyctalus leisleri")],
    ),
    TargetClassConfig(
        name="nycnoc",
        tags=[data.Tag(key="class", value="Nyctalus noctula")],
    ),
    TargetClassConfig(
        name="pipnat",
        tags=[data.Tag(key="class", value="Pipistrellus nathusii")],
    ),
    TargetClassConfig(
        name="pippip",
        tags=[data.Tag(key="class", value="Pipistrellus pipistrellus")],
    ),
    TargetClassConfig(
        name="pippyg",
        tags=[data.Tag(key="class", value="Pipistrellus pygmaeus")],
    ),
    TargetClassConfig(
        name="pleaur",
        tags=[data.Tag(key="class", value="Plecotus auritus")],
    ),
    TargetClassConfig(
        name="pleaus",
        tags=[data.Tag(key="class", value="Plecotus austriacus")],
    ),
    TargetClassConfig(
        name="rhifer",
        tags=[data.Tag(key="class", value="Rhinolophus ferrumequinum")],
    ),
    TargetClassConfig(
        name="rhihip",
        tags=[data.Tag(key="class", value="Rhinolophus hipposideros")],
    ),
]


def get_class_names_from_config(configs: List[TargetClassConfig]) -> List[str]:
    """Extract the list of class names from a ClassesConfig object.

    Parameters
    ----------
    config : ClassesConfig
        The loaded classes configuration object.

    Returns
    -------
    List[str]
        An ordered list of unique class names defined in the configuration.
    """
    return [class_info.name for class_info in configs]


def build_sound_event_encoder(
    configs: List[TargetClassConfig],
) -> SoundEventEncoder:
    """Build a sound event encoder function from the classes configuration."""
    conditions = {
        class_config.name: build_sound_event_condition(class_config.match_if)
        for class_config in configs
    }

    return SoundEventClassifier(conditions)


class SoundEventClassifier:
    def __init__(self, mapping: Dict[str, SoundEventCondition]):
        self.mapping = mapping

    def __call__(
        self, sound_event_annotation: data.SoundEventAnnotation
    ) -> Optional[str]:
        for name, condition in self.mapping.items():
            if condition(sound_event_annotation):
                return name


def build_sound_event_decoder(
    configs: List[TargetClassConfig],
    raise_on_unmapped: bool = False,
) -> SoundEventDecoder:
    """Build a sound event decoder function from the classes configuration."""
    mapping = {
        class_config.name: class_config.assign_tags for class_config in configs
    }
    return TagDecoder(mapping, raise_on_unknown=raise_on_unmapped)


class TagDecoder:
    def __init__(
        self,
        mapping: Dict[str, List[data.Tag]],
        raise_on_unknown: bool = True,
    ):
        self.mapping = mapping
        self.raise_on_unknown = raise_on_unknown

    def __call__(self, class_name: str) -> List[data.Tag]:
        tags = self.mapping.get(class_name)

        if tags is None:
            if self.raise_on_unknown:
                raise ValueError("Invalid class name")

            tags = []

        return tags
