# Data source reference

This page summarizes dataset source formats and their config fields.

## Supported source formats

| Format | Description |
| --- | --- |
| `aoef` | AOEF/soundevent annotation files (`AnnotationSet` or `AnnotationProject`) |
| `batdetect2` | Legacy format with one JSON annotation file per recording |
| `batdetect2_file` | Legacy format with one merged JSON annotation file |

## AOEF (`format: aoef`)

Required fields:

- `name`
- `format`
- `audio_dir`
- `annotations_path`

Optional fields:

- `description`
- `filter`

`filter` is only used when `annotations_path` points to an `AnnotationProject`.

AOEF filter options:

- `only_completed` (default:
  `true`)
- `only_verified` (default:
  `false`)
- `exclude_issues` (default:
  `true`)

Use `filter:
null` to disable project filtering.

## Legacy per-file (`format: batdetect2`)

Required fields:

- `name`
- `format`
- `audio_dir`
- `annotations_dir`

Optional fields:

- `description`
- `filter`

## Legacy merged file (`format: batdetect2_file`)

Required fields:

- `name`
- `format`
- `audio_dir`
- `annotations_path`

Optional fields:

- `description`
- `filter`

Legacy filter options:

- `only_annotated` (default:
  `true`)
- `exclude_issues` (default:
  `true`)

Use `filter:
null` to disable filtering.

## Related guides

- {doc}`../../../how_to/data/configure-aoef-dataset`
- {doc}`../../../how_to/data/import-legacy-batdetect2-annotations`
