# How to save predictions in different output formats

Use this guide when you need BatDetect2 outputs in a specific representation for
downstream tools.

## Choose the format that matches the job

Current built-in output formats include:

- `raw`:
  one NetCDF file per clip, best for rich structured outputs,
- `parquet`:
  tabular storage for data analysis workflows,
- `soundevent`:
  prediction-set JSON for soundevent-style tooling,
- `batdetect2`:
  legacy per-recording JSON output.

## Select a format from the CLI

Use `--format` for quick experiments.

```bash
batdetect2 process directory \
  path/to/model.ckpt \
  path/to/audio_dir \
  path/to/outputs \
  --format parquet
```

## Use an outputs config for repeatable runs

Use an outputs config when you want reproducible control over format and
transforms.

Example:

```yaml
format:
  name: raw
  include_class_scores: true
  include_features: true
  include_geometry: true
transform:
  detection_transforms: []
  clip_transforms: []
```

Run with:

```bash
batdetect2 process directory \
  path/to/model.ckpt \
  path/to/audio_dir \
  path/to/outputs \
  --outputs-config path/to/outputs.yaml
```

## Pick the simplest useful format

- Use `raw` if you want the richest output surface and easy round-tripping.
- Use `parquet` if you want tabular analysis in Python or data-lake workflows.
- Use `soundevent` if you want prediction-set JSON.
- Use `batdetect2` only when you need the legacy JSON shape.

## Related pages

- Outputs config reference:
  {doc}`../reference/outputs-config`
- Output formats reference:
  {doc}`../reference/output-formats`
- Output transforms reference:
  {doc}`../reference/output-transforms`
