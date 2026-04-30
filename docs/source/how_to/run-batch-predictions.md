# How to run batch predictions

This guide shows practical command patterns for directory-based and file-list
prediction runs.

Use it after you already know which input mode you want and need concrete command templates for a repeatable batch run.

## Predict from a directory

```bash
batdetect2 predict directory \
  path/to/model.ckpt \
  path/to/audio_dir \
  path/to/outputs
```

Use this when BatDetect2 should discover the audio files for you.

## Predict from a file list

```bash
batdetect2 predict file_list \
  path/to/model.ckpt \
  path/to/audio_files.txt \
  path/to/outputs
```

Use this when another part of your workflow already produced the exact recording list to process.

## Predict from a dataset config

```bash
batdetect2 predict dataset \
  path/to/model.ckpt \
  path/to/annotation_set.json \
  path/to/outputs
```

Use this when your project already has a `soundevent` annotation set and you want to extract unique recording paths from it.

## Useful options

- `--batch-size` to control throughput.
- `--workers` to set data-loading parallelism.
- `--format` to select output format.
- `--inference-config` to control clipping and loader behavior.
- `--outputs-config` to control serialization and output transforms.
- `--detection-threshold` to override the detection threshold for a run.

## Practical workflow

For large runs:

1. test the command on a small reviewed subset,
2. lock the config files and command shape,
3. write outputs to a dedicated directory per run,
4. record the checkpoint, config paths, and thresholds used.

For complete option details, see {doc}`../reference/cli/predict`.
