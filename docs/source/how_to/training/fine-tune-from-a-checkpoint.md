# How to fine-tune from a checkpoint

Use this guide when you want to continue from an existing checkpoint instead of training a fresh model config.

## Use `--model` for checkpoint-based training

Pass a checkpoint with `--model`.

Do not combine `--model` with `--model-config`.

```bash
batdetect2 train \
  path/to/train_dataset.yaml \
  --val-dataset path/to/val_dataset.yaml \
  --model path/to/model.ckpt \
  --training-config path/to/training.yaml
```

## Keep targets and preprocessing aligned

If you override targets or audio-related settings while fine-tuning, validate that they still match the checkpoint and your dataset.

Mismatches here can produce confusing failures or invalid comparisons.

## Decide what question the fine-tune should answer

Common fine-tuning goals are:

- adapting to local recording conditions,
- adapting to a new label set,
- improving performance on a narrower deployment context.

Make that goal explicit before comparing results.

## Evaluate after fine-tuning

Always compare the fine-tuned checkpoint against a held-out dataset.

Use the same evaluation setup when comparing before and after.

## Related pages

- Training tutorial: {doc}`../tutorials/train-a-custom-model`
- Evaluate a test set: {doc}`../tutorials/evaluate-on-a-test-set`
- Train command reference: {doc}`../reference/cli/train`
