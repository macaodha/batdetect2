# How to choose an inference input mode

Use this guide to decide whether `process directory`, `process file_list`, or
`process dataset` is the right entry point for your run.

## Use `process directory` when the recordings already live together

This is the simplest choice.

Use it when:

- your recordings are already organized in one directory tree,
- you want BatDetect2 to discover audio files for you,
- you are doing a first pass over a folder of recordings.

```bash
batdetect2 process directory \
  path/to/model.ckpt \
  path/to/audio_dir \
  path/to/outputs
```

## Use `process file_list` when you need explicit control over the file set

Use it when:

- you want to run only a selected subset,
- your files are spread across directories,
- another tool has already produced the exact list of recordings to process.

The list file should contain one path per line.

```bash
batdetect2 process file_list \
  path/to/model.ckpt \
  path/to/audio_files.txt \
  path/to/outputs
```

## Use `process dataset` when your workflow is already annotation-set driven

Use it when:

- your project already has a `soundevent` annotation set,
- you want prediction runs aligned with that annotation metadata,
- you want BatDetect2 to resolve recording paths from the annotation set.

```bash
batdetect2 process dataset \
  path/to/model.ckpt \
  path/to/annotation_set.json \
  path/to/outputs
```

The dataset command reads a `soundevent` annotation set and extracts unique
recording paths before inference.

## Rule of thumb

- Start with `directory` for the easiest first run.
- Use `file_list` when selection matters.
- Use `dataset` when the rest of your workflow is already dataset-based.

## Related pages

- Run batch predictions:
  {doc}`run-batch-predictions`
- Tune inference clipping:
  {doc}`tune-inference-clipping`
- Process command reference:
  {doc}`../reference/cli/predict`
