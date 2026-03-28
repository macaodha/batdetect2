# How to tune detection threshold

Use this guide to compare detection outputs at different threshold values.

## 1) Start with a baseline run

Run an initial prediction workflow and keep outputs in a dedicated folder.

## 2) Sweep threshold values

If you use the legacy `detect` command, run multiple thresholds (for example,
`0.1`, `0.3`, `0.5`) and compare output counts and quality on a validation
subset.

## 3) Validate against known calls

Use files with trusted annotations or expert review to select a threshold that
fits your project goals.

## 4) Record your chosen setting

Write down the chosen threshold and rationale so analyses are reproducible.

For conceptual trade-offs, see
{doc}`../explanation/model-output-and-validation`.
