# Targets config workflow reference

This page summarizes the target-definition configuration used by batdetect2.

## `TargetConfig`

Defined in `batdetect2.targets.config`.

Fields:

- `detection_target`: one `TargetClassConfig` defining detection eligibility.
- `classification_targets`: list of `TargetClassConfig` entries for class
  encoding/decoding.
- `roi`: default ROI mapper config.

## `TargetClassConfig`

Defined in `batdetect2.targets.classes`.

Fields:

- `name`: class label name.
- `tags`: tag list used for matching (shortcut for `match_if`).
- `match_if`: explicit condition config (`match_if` is accepted as alias).
- `assign_tags`: tags used when decoding this class.
- `roi`: optional class-specific ROI mapper override.

`tags` and `match_if` are mutually exclusive.

## Supported condition config types

Built from `batdetect2.data.conditions`.

- `has_tag`
- `has_all_tags`
- `has_any_tag`
- `duration`
- `frequency`
- `all_of`
- `any_of`
- `not`

## ROI mapper config

`roi` supports built-in mappers including:

- `anchor_bbox`
- `peak_energy_bbox`

Key `anchor_bbox` fields:

- `anchor`
- `time_scale`
- `frequency_scale`

## Related pages

- Detection target setup: {doc}`../how_to/configure-target-definitions`
- Class setup: {doc}`../how_to/define-target-classes`
- ROI setup: {doc}`../how_to/configure-roi-mapping`
- Concept overview: {doc}`../explanation/target-encoding-and-decoding`
