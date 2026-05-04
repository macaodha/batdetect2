from soundevent import data

from batdetect2.targets import (
    TargetClassConfig,
    TargetConfig,
    build_targets,
    check_target_compatibility,
)


def _target_class(name: str) -> TargetClassConfig:
    return TargetClassConfig(
        name=name,
        tags=[data.Tag(key="class", value=name)],
    )


def test_check_target_compatibility_accepts_superset_targets() -> None:
    config = TargetConfig(
        classification_targets=[
            _target_class("pip35"),
            _target_class("myo"),
            _target_class("extra"),
        ]
    )
    targets = build_targets(config)

    assert check_target_compatibility(targets, ["pip35", "myo"])


def test_check_target_compatibility_rejects_missing_model_classes() -> None:
    config = TargetConfig(
        classification_targets=[
            _target_class("pip35"),
            _target_class("myo"),
        ]
    )
    targets = build_targets(config)

    assert not check_target_compatibility(targets, ["pip35", "nyc"])
