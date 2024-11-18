from functools import partial
from typing import Callable, List, Optional, Set

from pydantic import Field
from soundevent import data
from soundevent.types import ClassMapper

from batdetect2.configs import BaseConfig
from batdetect2.terms import TagInfo, get_tag_from_info


class TargetConfig(BaseConfig):
    """Configuration for target generation."""

    classes: List[TagInfo] = Field(default_factory=list)
    generic_class: Optional[TagInfo] = None

    include: Optional[List[TagInfo]] = None
    exclude: Optional[List[TagInfo]] = None


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


def build_class_mapper(classes: List[TagInfo]) -> ClassMapper:
    target_tags = [get_tag_from_info(tag) for tag in classes]
    labels = [tag.label if tag.label else tag.value for tag in classes]
    return GenericMapper(
        classes=target_tags,
        labels=labels,
    )


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


class GenericMapper(ClassMapper):
    """Generic class mapper configuration."""

    def __init__(
        self,
        classes: List[data.Tag],
        labels: List[str],
    ):
        if not len(classes) == len(labels):
            raise ValueError("Number of targets and class labels must match.")

        self.targets = set(classes)
        self.class_labels = list(dict.fromkeys(labels))

        self._mapping = {tag: label for tag, label in zip(classes, labels)}
        self._inverse_mapping = {
            label: tag for tag, label in zip(classes, labels)
        }

    def encode(
        self,
        sound_event_annotation: data.SoundEventAnnotation,
    ) -> Optional[str]:
        tags = set(sound_event_annotation.tags)

        intersection = tags & self.targets
        if not intersection:
            return None

        tag = intersection.pop()
        return self._mapping[tag]

    def decode(self, label: str) -> List[data.Tag]:
        if label not in self._inverse_mapping:
            return []
        return [self._inverse_mapping[label]]
