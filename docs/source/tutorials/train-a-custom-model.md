# Tutorial: Train a custom model

This tutorial walks through a first custom training run using your own
annotations.

This tutorial is for advanced users who already have dataset files and want to train a model on their own annotated data.

## Before you start

- BatDetect2 installed.
- A training dataset config file.
- (Optional) A validation dataset config file.
- A targets config file if you are not using the default target setup.
- A model config file if you are not training from the built-in defaults.

```{note}
This is not the first page to start with if you only want to run the existing model on recordings.
Use {doc}`run-inference-on-folder` for that.
```

## Outcome

By the end of this tutorial you will have:

- started a training run,
- written checkpoints and logs,
- understood the minimum settings involved,
- identified the next pages for fine-tuning and evaluation.

## 1. Gather the minimum required inputs

At minimum, a custom training run needs:

- a training dataset config,
- optional validation dataset config,
- either a model config for a fresh run or a checkpoint for continued training,
- optional settings files for targets, audio, training, evaluation, inference, outputs, and logging.

The most important point is that the dataset file, target definitions, and preprocessing choices need to agree with each other.

## 2. Run a first training command

Use a command like this for a fresh run:

```bash
batdetect2 train \
  path/to/train_dataset.yaml \
  --val-dataset path/to/val_dataset.yaml \
  --targets path/to/targets.yaml \
  --model-config path/to/model.yaml \
  --training-config path/to/training.yaml
```

Use `--model` instead of `--model-config` when you want to continue from an existing checkpoint.

## 3. Check that outputs are being written

After the command starts, verify that:

- the run initializes without configuration errors,
- checkpoints are written to the checkpoint directory,
- logs are written to the log directory or configured logger backend,
- the training and validation datasets load as expected.

## 4. Run a sanity inference pass after training

Do not wait until full evaluation to confirm that the trained checkpoint behaves sensibly.

Take a small reviewed subset of recordings and run a quick prediction pass with the new checkpoint.

That catches setup mismatches early, especially around targets and preprocessing.

## 5. Evaluate on held-out data

Once the checkpoint looks sensible on a small sanity subset, run the formal evaluation workflow on a held-out test set.

That is where you should compare models, thresholds, and task-level performance metrics.

## What to do next

- Evaluate the trained checkpoint: {doc}`evaluate-on-a-test-set`
- Fine-tune from a checkpoint: {doc}`../how_to/fine-tune-from-a-checkpoint`
- Configure targets: {doc}`../how_to/configure-target-definitions`
- Configure preprocessing: {doc}`../how_to/configure-audio-preprocessing`
- Check full train options: {doc}`../reference/cli/train`
