from functools import partial
from typing import Callable, List, Optional, Set

from pydantic import Field
from soundevent import data

from batdetect2.configs import BaseConfig
from batdetect2.terms import TagInfo, get_tag_from_info


class TargetConfig(BaseConfig):
    """Configuration for target generation."""

    classes: List[TagInfo] = Field(
        default_factory=lambda: [
            TagInfo(key="class", value=value) for value in DEFAULT_SPECIES_LIST
        ]
    )
    generic_class: Optional[TagInfo] = Field(
        default_factory=lambda: TagInfo(key="class", value="Bat")
    )

    include: Optional[List[TagInfo]] = Field(
        default_factory=lambda: [TagInfo(key="event", value="Echolocation")]
    )
    exclude: Optional[List[TagInfo]] = Field(
        default_factory=lambda: [
            TagInfo(key="class", value=""),
            TagInfo(key="class", value=" "),
            TagInfo(key="class", value="Unknown"),
        ]
    )


def build_sound_event_filter(
    include: Optional[List[TagInfo]] = None,
    exclude: Optional[List[TagInfo]] = None,
) -> Callable[[data.SoundEventAnnotation], bool]:
    include_tags = (
        {get_tag_from_info(tag) for tag in include} if include else None
    )
    exclude_tags = (
        {get_tag_from_info(tag) for tag in exclude} if exclude else None
    )
    return partial(
        filter_sound_event,
        include=include_tags,
        exclude=exclude_tags,
    )


def get_tag_label(tag_info: TagInfo) -> str:
    return tag_info.label if tag_info.label else tag_info.value


def get_class_names(classes: List[TagInfo]) -> List[str]:
    return sorted({get_tag_label(tag) for tag in classes})


def build_encoder(
    classes: List[TagInfo],
) -> Callable[[data.SoundEventAnnotation], Optional[str]]:
    target_tags = set([get_tag_from_info(tag) for tag in classes])

    tag_mapping = {
        tag: get_tag_label(tag_info)
        for tag, tag_info in zip(target_tags, classes)
    }

    def encoder(
        sound_event_annotation: data.SoundEventAnnotation,
    ) -> Optional[str]:
        tags = set(sound_event_annotation.tags)

        intersection = tags & target_tags

        if not intersection:
            return None

        first = intersection.pop()
        return tag_mapping[first]

    return encoder


def build_decoder(
    classes: List[TagInfo],
) -> Callable[[str], List[data.Tag]]:
    target_tags = set([get_tag_from_info(tag) for tag in classes])
    tag_mapping = {
        get_tag_label(tag_info): tag
        for tag, tag_info in zip(target_tags, classes)
    }

    def decoder(label: str) -> List[data.Tag]:
        tag = tag_mapping.get(label)
        return [tag] if tag else []

    return decoder


def filter_sound_event(
    sound_event_annotation: data.SoundEventAnnotation,
    include: Optional[Set[data.Tag]] = None,
    exclude: Optional[Set[data.Tag]] = None,
) -> bool:
    tags = set(sound_event_annotation.tags)

    if include is not None and not tags & include:
        return False

    if exclude is not None and tags & exclude:
        return False

    return True


DEFAULT_SPECIES_LIST = [
    "Barbastellus barbastellus",
    "Eptesicus serotinus",
    "Myotis alcathoe",
    "Myotis bechsteinii",
    "Myotis brandtii",
    "Myotis daubentonii",
    "Myotis mystacinus",
    "Myotis nattereri",
    "Nyctalus leisleri",
    "Nyctalus noctula",
    "Pipistrellus nathusii",
    "Pipistrellus pipistrellus",
    "Pipistrellus pygmaeus",
    "Plecotus auritus",
    "Plecotus austriacus",
    "Rhinolophus ferrumequinum",
    "Rhinolophus hipposideros",
]
