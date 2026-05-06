# Tutorial: Run BatDetect2 on a folder of audio files

This tutorial walks through a first end-to-end inference run with the CLI.

It is the default starting point for new users.

Use it when you want to run an existing model on a folder of recordings and
quickly check what BatDetect2 found.

## Before you start

- BatDetect2 installed in your environment.
- A folder containing `.wav` files.
- A model checkpoint path.

A checkpoint is the saved model file that BatDetect2 uses to make predictions.

If you are working from this repository checkout, you can use:

```text
src/batdetect2/models/checkpoints/Net2DFast_UK_same.pth.tar
```

## Outcome

By the end of this tutorial you will have:

- run `batdetect2 process directory`,
- saved predictions to disk,
- checked that BatDetect2 wrote output files,
- identified the next pages to use for tuning or customization.

## 1. Choose your input and output paths

Pick three paths:

- the checkpoint to use,
- the directory containing your audio files,
- an output directory where BatDetect2 will save its results.

Example layout:

```text
project/
  model.pth.tar
  audio/
    file_001.wav
    file_002.wav
  outputs/
```

## 2. Run processing on the directory

Use this command when you want BatDetect2 to scan a folder of recordings
automatically.

```bash
batdetect2 process directory \
  path/to/model.pth.tar \
  path/to/audio_dir \
  path/to/outputs
```

What this does:

- loads the checkpoint,
- finds audio files in `audio_dir`,
- splits recordings into smaller pieces internally when needed,
- saves result files to `outputs`.

## 3. Verify that outputs were written

After the command completes, inspect the output directory.

For a first run, the important check is simple:

- did BatDetect2 create result files,
- are they in the output directory you expected,
- did it process the recordings you meant to analyze.

Different workflows can save results in different file formats.

You do not need to learn those details for the first run.

If you later need to choose a specific output format, go to
{doc}`../how_to/save-predictions-in-different-output-formats`.

## 4. Inspect predictions

Start with a small subset of representative files.

Check:

- whether detections were written for the expected recordings,
- whether output counts are plausible,
- whether the model is obviously too sensitive or too conservative,
- whether the predicted classes look broadly reasonable for your data.

Do not treat the first run as validated ecological output.

The first run is a workflow check.

Validation comes next.

## 5. Tune only after you have a baseline

If the first run is too noisy or misses obvious calls, tune thresholds on a
reviewed subset rather than changing settings blindly across the full dataset.

Use {doc}`../how_to/tune-detection-threshold` for that process.

## What to do next

- If you need a different input mode, use
  {doc}`../how_to/choose-an-inference-input-mode`.
- If you want to tune sensitivity, use
  {doc}`../how_to/tune-detection-threshold`.
- If you already write code and want more control from Python, use
  {doc}`integrate-with-a-python-pipeline`.
- If you need full command details, use {doc}`../reference/cli/predict`.
