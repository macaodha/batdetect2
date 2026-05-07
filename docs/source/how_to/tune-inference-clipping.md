# How to tune inference clipping

Use this guide when long recordings need to be split into smaller clips during
inference.

## What clipping controls

`InferenceConfig.clipping` controls how recordings are split before batching.

Key fields are:

- `duration`:
  clip duration in seconds,
- `overlap`:
  overlap between adjacent clips,
- `max_empty`:
  how much empty padding is allowed,
- `discard_empty`:
  whether empty clips are dropped.

## Start from the defaults

Use the built-in clipping behavior first unless you already know you need
something else.

Only tune clipping when:

- recordings are much longer than your normal working set,
- you are seeing edge effects around calls,
- you need tighter control over throughput or padding behavior.

## Override clipping with an inference config

Create an inference config file and pass it to `process` or `evaluate`.

Example:

```yaml
clipping:
  enabled: true
  duration: 0.5
  overlap: 0.1
  max_empty: 0.0
  discard_empty: true
loader:
  batch_size: 8
```

Run with:

```bash
batdetect2 process directory \
  path/to/model.ckpt \
  path/to/audio_dir \
  path/to/outputs \
  --inference-config path/to/inference.yaml
```

## Validate clipping changes on a small reviewed subset

Changing clipping changes what the model sees per batch and can change how
events near clip boundaries behave.

Check a reviewed subset before applying clipping changes to a full project.

## Related pages

- Inference config reference:
  {doc}`../reference/inference-config`
- Run batch predictions:
  {doc}`run-batch-predictions`
- Understanding the pipeline:
  {doc}`../explanation/pipeline-overview`
