# Output formats reference

BatDetect2 currently supports several built-in output formatters.

## `raw`

Defined by `RawOutputConfig`.

Best for rich structured outputs and round-tripping.

Key fields:

- `include_class_scores`
- `include_features`
- `include_geometry`

Writes one NetCDF `.nc` file per clip.

## `parquet`

Defined by `ParquetOutputConfig`.

Best for tabular analysis workflows.

Key fields:

- `include_class_scores`
- `include_features`
- `include_geometry`

Writes a parquet table, typically `predictions.parquet`.

## `soundevent`

Defined by `SoundEventOutputConfig`.

Best when you want a `PredictionSet` JSON workflow.

Key fields:

- `top_k`
- `min_score`

Writes a prediction-set JSON file.

## `batdetect2`

Defined by `BatDetect2OutputConfig`.

This is the legacy-compatible BatDetect2 formatter.

Key fields:

- `event_name`
- `annotation_note`
- `write_detection_csv`
- `write_cnn_features_csv`
- `save_if_empty`
- `preserve_audio_tree`
- `include_file_path`

By default it writes one `.json` file and one detection `.csv` file per
recording, preserving the input audio directory layout under the output root.

It can also write legacy `_cnn_features.csv` sidecars when
`write_cnn_features_csv` is enabled.

## Related pages

- Outputs config:
  {doc}`outputs-config`
- Save predictions in different output formats:
  {doc}`../../../how_to/inference/save-predictions-in-different-output-formats`
- Understanding formatted outputs:
  {doc}`../../../explanation/interpreting-formatted-outputs`
