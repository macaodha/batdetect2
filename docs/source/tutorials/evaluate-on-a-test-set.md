# Tutorial: Evaluate on a test set

This tutorial shows how to evaluate a trained checkpoint on a held-out dataset
and inspect the output metrics.

## Before you start

- A trained model checkpoint.
- A test dataset config file.
- (Optional) Targets, audio, inference, and evaluation config overrides.

## Tutorial steps

1. Select a checkpoint and a test dataset.
2. Run `batdetect2 evaluate`.
3. Inspect output metrics and prediction artifacts.
4. Record evaluation settings for reproducibility.

## Example command

```bash
batdetect2 evaluate \
  path/to/model.ckpt \
  path/to/test_dataset.yaml \
  --output-dir path/to/eval_outputs
```

## What to do next

- Compare thresholds on representative files:
  {doc}`../how_to/tune-detection-threshold`
- Check full evaluate options: {doc}`../reference/cli/evaluate`

This page is a starter scaffold and will be expanded with a full worked
example.
