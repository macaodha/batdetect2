# Logging config reference

`AppLoggingConfig` controls which logger backend BatDetect2 uses for training,
evaluation, and inference.

Defined in `batdetect2.logging`.

## Top-level fields

- `train`
  - logger config for training runs.
- `evaluation`
  - logger config for evaluation runs.
- `inference`
  - logger config for inference runs.

## Built-in logger backends

Current built-in logger backends are:

- `csv`
- `tensorboard`
- `mlflow`
- `dvclive`

## Default behaviour

By default:

- training uses `csv`,
- evaluation uses `csv`,
- inference uses `csv`.

With the CSV logger, training writes a `metrics.csv` file in the log folder.

Example files live under `example_data/configs/`, including
`example_data/configs/logging.yaml`.

## Related pages

- Train command reference:
  {doc}`cli/train`
- Evaluate command reference:
  {doc}`cli/evaluate`
- Run inference on a folder:
  {doc}`../tutorials/run-inference-on-folder`
