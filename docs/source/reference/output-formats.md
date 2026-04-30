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

This is the legacy BatDetect2-style JSON output.

Key fields:

- `event_name`
- `annotation_note`

Writes one `.json` file per recording.

## Related pages

- Outputs config: {doc}`outputs-config`
- Save predictions in different output formats: {doc}`../how_to/save-predictions-in-different-output-formats`
- Understanding formatted outputs: {doc}`../explanation/interpreting-formatted-outputs`
