# How to tune detection threshold

Use this guide to compare detection outputs at different threshold values.

The goal is not to find a universal threshold.

The goal is to choose a threshold that fits your reviewed local data and the project trade-off between missed calls and false positives.

## 1) Start with a baseline run

Run an initial prediction workflow and keep outputs in a dedicated folder.

## 2) Sweep threshold values

Run `predict` multiple times with different thresholds (for example `0.1`,
`0.3`, `0.5`) and compare output counts and quality on the same validation
subset.

```bash
batdetect2 predict directory \
  path/to/model.ckpt \
  path/to/audio_dir \
  path/to/outputs_thr_03 \
  --detection-threshold 0.3
```

Keep each threshold run in a separate output directory.

That makes it easier to compare counts and inspect example files without mixing results.

## 3) Validate against known calls

Use files with trusted annotations or expert review to select a threshold that
fits your project goals.

Check both:

- obvious false positives,
- obvious missed calls.

If class interpretation matters downstream, inspect class ranking behavior as well, not just detection counts.

## 4) Record your chosen setting

Write down the chosen threshold and rationale so analyses are reproducible.

For conceptual trade-offs, see
{doc}`../explanation/model-output-and-validation`.
