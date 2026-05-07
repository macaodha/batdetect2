# Train a custom model

This tutorial walks through a first custom training run using your own annotations.

Use it when you already have labelled recordings and want to train a model for your own data.

## Before you start

You need:

- BatDetect2 installed.
- labelled recordings and annotations.

```{note}
This is not the first page to start with if you only want to run the existing
model on recordings.
Use {doc}`run-inference-on-folder` for that.
```

## Optional: use the repository example files

If you want to follow the steps with the same files shown here, clone the repository and move into it:

```bash
git clone https://github.com/macaodha/batdetect2.git
cd batdetect2
```

## What you will do

By the end of this tutorial you will have:

- created a dataset config,
- defined a targets config,
- started a training run,
- checked the checkpoint and log outputs,
- identified the next pages for evaluation and customisation.

## 1. Create a dataset config

The dataset config explicitly declares what data you want to use for training.
It is a YAML file.
If YAML is new to you, see [Learn YAML in Y Minutes](https://learnxinyminutes.com/yaml/).

In the dataset config, you list one or more data sources.
Each source tells `batdetect2` where the audio recordings live and where the matching annotations are stored.

BatDetect2 can read annotations from different source formats.
In this example, we use the example data in the `batdetect2` format.

Use `example_data/dataset.yaml` as a reference:

```yaml
name: example dataset
description: Only for demonstration purposes
sources:
  - format: batdetect2
    name: Example Data
    description: Examples included for testing batdetect2
    annotations_dir: example_data/anns
    audio_dir: example_data/audio
```

For your own project, the main thing to change is the file paths.
If you have several collections of recordings, you can add more than one source to the same dataset config.
That lets you describe the full training data you want to use in one place.

If you need more detail on dataset source formats, see {doc}`../reference/data-sources`.

## 2. Define a targets config

The targets config tells BatDetect2 how to turn your annotations into training targets.

It defines two main things:

- what should count as a detection,
- which classes the model should learn to predict.

In practice, this means the targets config maps the labels in your annotations to the detection and classification outputs used during training.

Use `example_data/targets.yaml` as a reference:

```yaml
detection_target:
  name: bat
  match_if:
    name: all_of
    conditions:
      - name: has_tag
        tag: { key: event, value: Echolocation }
      - name: not
        condition:
          name: has_tag
          tag: { key: class, value: Unknown }
  assign_tags:
    - key: class
      value: Bat

classification_targets:
  - name: myomys
    tags:
      - key: class
        value: Myotis mystacinus
  - name: pippip
    tags:
      - key: class
        value: Pipistrellus pipistrellus
```

For your own project, update the matching rules and class definitions so they fit your labels.

In this example:

- `detection_target` says that echolocation calls should be treated as detections,
- `classification_targets` define the classes the model should predict,

It is worth taking a bit of time over this file, because your targets config decides what the model is actually being asked to learn.

If you need help with that, see {doc}`../how_to/configure-target-definitions` and {doc}`../reference/targets-config-workflow`.

## 3. Run a first training command

For a first run, keep the command simple:

```bash
batdetect2 train \
  path/to/train_dataset.yaml \
  --val-dataset path/to/val_dataset.yaml \
  --targets path/to/targets.yaml
```

If you are using the repository example files, run:

```bash
batdetect2 train \
  example_data/dataset.yaml \
  --val-dataset example_data/dataset.yaml \
  --targets example_data/targets.yaml
```

This uses the same dataset for training and validation only to keep the example simple.
For real training runs, you usually want separate training and validation datasets.

This uses the built-in default model and training settings.
If you want to change the model architecture later, see {doc}`../reference/model-config`.
If you want to change optimiser settings, batch size, epochs, or checkpoint behaviour, see {doc}`../reference/training-config`.

## 4. Check the training outputs

After the run starts, `batdetect2` should write checkpoints and logs.

By default, training logs are written with the CSV logger.
That means you should see a log folder with a `metrics.csv` file.

A typical layout looks like this:

```text
outputs/
  checkpoints/
    epoch=19-step=20.ckpt
  logs/
    version_0/
      metrics.csv
      hparams.yaml
    training_artifacts/
      train_dataset.yaml
      val_dataset.yaml
      targets.yaml
      train_class_summary.csv
      val_class_summary.csv
```

The checkpoint is the trained model you can use later for inference, evaluation, or sharing with someone else.

The files in `training_artifacts/` record which datasets and targets were used for the run.
The `hparams.yaml` file records the full training setup, including the configs used for the model, training, and other parts of the run.

The `metrics.csv` file stores one row per validation epoch.
It includes training losses as well as validation losses and metrics such as:

```csv
classification/mean_average_precision,detection/average_precision,epoch,total_loss/val
0.10041624307632446,0.3697187900543213,0,4070.3515625
0.11328697204589844,0.346899151802063,1,3941.6455078125
0.1388484090566635,0.36171725392341614,2,3776.323974609375
```

You may also see class-specific metrics in extra columns.

The more detailed metrics are computed from the validation set.
If you do not provide `--val-dataset`, those validation metrics will not appear.

Other logger backends are also supported, including TensorBoard, MLflow, and DVCLive.
See {doc}`../reference/logging-config` if you want to change that.

## Use the trained model

You can now use the trained checkpoint in BatDetect2, or share it with someone else to use in their own runs.
If you want to load it for inference or evaluation, see {doc}`../how_to/choose-a-model`.

## Common next steps

- Evaluate the trained checkpoint: {doc}`evaluate-on-a-test-set`
- Fine-tune from a checkpoint: {doc}`../how_to/fine-tune-from-a-checkpoint`
- Configure targets in more detail: {doc}`../how_to/configure-target-definitions`
- Configure audio preprocessing: {doc}`../how_to/configure-audio-preprocessing`
- Configure spectrogram preprocessing: {doc}`../how_to/configure-spectrogram-preprocessing`
- Check full train options: {doc}`../reference/cli/train`
