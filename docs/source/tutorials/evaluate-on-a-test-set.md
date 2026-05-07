# Evaluate on a test set

This tutorial shows how to evaluate a trained checkpoint on a held-out dataset
and inspect the output metrics.

Use it when you want to measure how a model performs on labelled data that was
kept aside for testing.

## Before you start

You need:

- a test dataset config,
- a trained checkpoint or model alias.

```{note}
This page is for model evaluation.
If you only want to run BatDetect2 on recordings, start with
{doc}`run-inference-on-folder` instead.
```

## What you will do

By the end of this tutorial you will have:

- prepared a test dataset config,
- run `batdetect2 evaluate`,
- written evaluation metrics and result files,
- identified the next pages for model choice and evaluation configuration.

## 1. Create a test dataset config

Evaluation needs a dataset config that points to the labelled data you want to
use for testing.

This is the same kind of dataset config used for training.
It explicitly declares which data sources BatDetect2 should read, including the
audio files and their annotations.

For an example, see `example_data/dataset.yaml`.

If you need help creating the dataset config, follow the dataset section in
{doc}`train-a-custom-model`.
For more detail on dataset source formats, see {doc}`../reference/data-sources`.

Use a dataset that was not used for training or tuning.

## 2. Run evaluation

For a simple run, use:

```bash
batdetect2 evaluate \
  path/to/test_dataset.yaml
```

If you do not pass `--model`, BatDetect2 uses the built-in default UK model.
If you want to choose a different checkpoint, alias, or Hugging Face model, see
{doc}`../how_to/choose-a-model`.

If you want to save the results somewhere else, add `--output-dir`:

```bash
batdetect2 evaluate \
  path/to/test_dataset.yaml \
  --model path/to/model.ckpt \
  --output-dir path/to/eval_outputs
```

This command loads the model, runs prediction on the test dataset, applies the
evaluation tasks, and writes the results to the output directory.

## 3. Check the output files

By default, the CLI writes evaluation outputs to `outputs/evaluation`.

With the default evaluation config, a run will usually create a folder like
this:

```text
outputs/evaluation/
  version_0/
    metrics.csv
    hparams.yaml
```

The most important file is `metrics.csv`.
It contains the metric values computed for the evaluation run.

A file like this might start like:

```csv
classification/average_precision/barbar,classification/average_precision/cneser,...,detection/average_precision
0.898695170879364,0.9408193826675415,...,0.851219117641449
```

The exact columns depend on the evaluation tasks you run.

The `hparams.yaml` file records the config used for the evaluation run.

## 4. Expect extra plots and files when configs enable them

You may also see extra outputs such as plots and saved predictions.

For example, if you run evaluation with `example_data/configs/evaluation.yaml`,
you should expect a richer output folder with:

- `metrics.csv`
- `hparams.yaml`
- a `plots/` directory
- a `predictions/` directory

That config enables more evaluation tasks and plots than the default setup.

So, depending on your evaluation config, you may see files such as:

- precision-recall plots,
- ROC curves,
- confusion matrices,
- example detection plots,
- saved prediction files.

If you want to control which tasks run and which plots are generated, see
{doc}`../reference/evaluation-config` and
{doc}`../how_to/choose-and-configure-evaluation-tasks`.

## Common next steps

- Choose a different model:
  {doc}`../how_to/choose-a-model`
- Configure evaluation tasks:
  {doc}`../how_to/choose-and-configure-evaluation-tasks`
- Interpret evaluation artifacts:
  {doc}`../how_to/interpret-evaluation-outputs`
- Learn the evaluation concepts:
  {doc}`../explanation/evaluation-concepts-and-matching`
- Check full evaluate options:
  {doc}`../reference/cli/evaluate`
