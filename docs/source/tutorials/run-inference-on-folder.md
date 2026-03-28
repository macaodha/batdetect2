# Tutorial: Run inference on a folder of audio files

## Prerequisites

- BatDetect2 installed in your environment.
- A folder containing `.wav` files.
- A model checkpoint path.

## Steps

1. Choose your input and output directories.
2. Run prediction with the CLI.
3. Verify output files were written.
4. Inspect predictions and confidence scores.

## Example command

```bash
batdetect2 predict directory \
  path/to/model.ckpt \
  path/to/audio_dir \
  path/to/outputs
```

## What to do next

- Use {doc}`../how_to/tune-detection-threshold` to tune sensitivity.
- Use {doc}`../reference/cli` for full command options.

Note: this is the initial Phase 1 scaffold and will be expanded with a full,
validated end-to-end walkthrough.
