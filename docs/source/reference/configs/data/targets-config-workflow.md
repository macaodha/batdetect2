# Targets config workflow reference

This page summarizes the target-definition configuration used by batdetect2.

## `TargetConfig`

Defined in `batdetect2.targets.config`.

Fields:

- `detection_target`:
  one `TargetClassConfig` defining detection eligibility.
- `classification_targets`:
  list of `TargetClassConfig` entries for class encoding/decoding.
- `roi`:
  ROI mapping config with `default` mapper and optional per-class `overrides`.

## `TargetClassConfig`

Defined in `batdetect2.targets.classes`.

Fields:

- `name`:
  class label name.
- `tags`:
  tag list used for matching (shortcut for `match_if`).
- `match_if`:
  explicit condition config (`match_if` is accepted as alias).
- `assign_tags`:
  tags used when decoding this class.

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

`roi.default` and each `roi.overrides.<class_name>` entry support built-in
mappers including:

- `anchor_bbox`
- `peak_energy_bbox`

Key `anchor_bbox` fields:

- `anchor`
- `time_scale`
- `frequency_scale`

Top-level ROI mapping shape:

- `default`:
  fallback mapper used for all classes.
- `overrides`:
  optional mapping from class name to mapper config.

## Related pages

- Detection target setup:
  {doc}`../../../how_to/data/configure-target-definitions`
- Class setup:
  {doc}`../../../how_to/data/define-target-classes`
- ROI setup:
  {doc}`../../../how_to/data/configure-roi-mapping`
- Concept overview:
  {doc}`../../../explanation/target-encoding-and-decoding`
