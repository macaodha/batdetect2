# Tutorial: Train a custom model

This tutorial walks through a first custom training run using your own
annotations.

## Before you start

- BatDetect2 installed.
- A training dataset config file.
- (Optional) A validation dataset config file.

## Tutorial steps

1. Prepare training and validation dataset config files.
2. Choose target definitions and model/training config files.
3. Run `batdetect2 train`.
4. Check that checkpoints and logs are written.
5. Run a quick sanity inference on a small audio subset.

## Example command

```bash
batdetect2 train \
  path/to/train_dataset.yaml \
  --val-dataset path/to/val_dataset.yaml \
  --targets path/to/targets.yaml \
  --model-config path/to/model.yaml \
  --training-config path/to/training.yaml
```

## What to do next

- Evaluate the trained checkpoint: {doc}`evaluate-on-a-test-set`
- Check full train options: {doc}`../reference/cli/train`

This page is a starter scaffold and will be expanded with a full worked
example.
