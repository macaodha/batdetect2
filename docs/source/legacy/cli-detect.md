# Legacy CLI workflow: `batdetect2 detect`

This page documents the previous CLI workflow based on `batdetect2 detect`.

```{warning}
This is legacy documentation.
For new workflows, use `batdetect2 predict directory` instead.
If you are migrating, start with {doc}`migration-guide`.
```

## Legacy command shape

```bash
batdetect2 detect AUDIO_DIR ANN_DIR DETECTION_THRESHOLD
```

Common legacy options included:

- `--cnn_features`
- `--spec_features`
- `--time_expansion_factor`
- `--save_preds_if_empty`
- `--model_path`

## Current replacement

The closest current CLI entry point is:

```bash
batdetect2 predict directory \
  path/to/model.ckpt \
  path/to/audio_dir \
  path/to/outputs
```

## Related pages

- Migration guide: {doc}`migration-guide`
- Current predict docs: {doc}`../reference/cli/predict`
