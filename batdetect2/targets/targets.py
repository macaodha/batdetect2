from collections.abc import Iterable
from pathlib import Path
from typing import Callable, List, Optional

from pydantic import Field
from soundevent import data

from batdetect2.configs import BaseConfig, load_config
from batdetect2.targets.terms import TagInfo, get_tag_from_info

__all__ = [
    "TargetConfig",
    "load_target_config",
    "build_target_encoder",
    "build_decoder",
]


class ReplaceConfig(BaseConfig):
    """Configuration for replacing tags."""

    original: TagInfo
    replacement: TagInfo


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

    replace: Optional[List[ReplaceConfig]] = None


def get_tag_label(tag_info: TagInfo) -> str:
    # TODO: Review this
    return tag_info.value


def get_class_names(classes: List[TagInfo]) -> List[str]:
    return sorted({get_tag_label(tag) for tag in classes})


def build_replacer(
    rules: List[ReplaceConfig],
) -> Callable[[data.Tag], data.Tag]:
    mapping = {
        get_tag_from_info(rule.original): get_tag_from_info(rule.replacement)
        for rule in rules
    }

    def replacer(tag: data.Tag) -> data.Tag:
        return mapping.get(tag, tag)

    return replacer


def build_target_encoder(
    classes: List[TagInfo],
    replacement_rules: Optional[List[ReplaceConfig]] = None,
) -> Callable[[Iterable[data.Tag]], Optional[str]]:
    target_tags = set([get_tag_from_info(tag) for tag in classes])

    tag_mapping = {
        tag: get_tag_label(tag_info)
        for tag, tag_info in zip(target_tags, classes)
    }

    replacer = (
        build_replacer(replacement_rules) if replacement_rules else lambda x: x
    )

    def encoder(
        tags: Iterable[data.Tag],
    ) -> Optional[str]:
        sanitized_tags = {replacer(tag) for tag in tags}

        intersection = sanitized_tags & target_tags

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


def load_target_config(
    path: Path, field: Optional[str] = None
) -> TargetConfig:
    return load_config(path, schema=TargetConfig, field=field)


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
