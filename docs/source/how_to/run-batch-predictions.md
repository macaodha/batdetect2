# How to run batch predictions

This guide shows practical command patterns for directory-based and file-list
prediction runs.

## Predict from a directory

```bash
batdetect2 predict directory \
  path/to/model.ckpt \
  path/to/audio_dir \
  path/to/outputs
```

## Predict from a file list

```bash
batdetect2 predict file_list \
  path/to/model.ckpt \
  path/to/audio_files.txt \
  path/to/outputs
```

## Useful options

- `--batch-size` to control throughput.
- `--workers` to set data-loading parallelism.
- `--format` to select output format.

For complete option details, see {doc}`../reference/cli`.
