# Tutorial: Evaluate on a test set

This tutorial shows how to evaluate a trained checkpoint on a held-out dataset
and inspect the output metrics.

This tutorial is for advanced users who want to compare one trained model against a separate test dataset.

## Before you start

- A trained model checkpoint.
- A test dataset config file.
- (Optional) Targets, audio, inference, and evaluation config overrides.

```{note}
This page is for model evaluation.
If you only want to run BatDetect2 on recordings,
start with {doc}`run-inference-on-folder` instead.
```

## Outcome

By the end of this tutorial you will have:

- run `batdetect2 evaluate`,
- written evaluation metrics and result files,
- understood what to inspect first,
- identified the next pages for evaluation concepts and configuration.

## 1. Start with a held-out dataset

Use a dataset that was not used for training or tuning.

A held-out dataset is simply a separate dataset kept aside for evaluation.

If you tune thresholds or configs on the same dataset that you report as final evaluation, the results will be optimistic.

## 2. Run evaluation

```bash
batdetect2 evaluate \
  path/to/model.ckpt \
  path/to/test_dataset.yaml \
  --base-dir path/to/project_root \
  --output-dir path/to/eval_outputs
```

This command loads the checkpoint,
runs prediction on the test dataset,
applies the chosen evaluation tasks,
and writes metrics and result files to the output directory.

Use `--base-dir` whenever the dataset config contains relative paths.

That is the common case for project-local dataset files.

## 3. Inspect the output directory

Look for:

- summary metrics,
- generated plots,
- saved prediction files if they were enabled,
- enough metadata to reproduce the run later.

The exact set depends on the configured evaluation tasks and plots.

## 4. Interpret the results in context

Do not reduce evaluation to a single number.

Check:

- which task the metric belongs to,
- which thresholding or matching assumptions were used,
- whether class-level behavior matches your use case,
- whether the failures are concentrated in specific taxa, sites, or recording conditions.

## 5. Record the evaluation setup

Keep the command, config files, checkpoint path, and dataset version together.

That matters for reproducibility and for later model comparisons.

## What to do next

- Compare thresholds on representative files:
  {doc}`../how_to/tune-detection-threshold`
- Configure evaluation tasks: {doc}`../how_to/choose-and-configure-evaluation-tasks`
- Interpret evaluation artifacts: {doc}`../how_to/interpret-evaluation-outputs`
- Learn the evaluation concepts: {doc}`../explanation/evaluation-concepts-and-matching`
- Check full evaluate options: {doc}`../reference/cli/evaluate`
