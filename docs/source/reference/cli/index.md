# CLI reference

Use this section to find the right command quickly, then open the command page
for the full option list.

## Command map

| Command | Use it for | Required positional args |
| --- | --- | --- |
| `batdetect2 predict` | Run prediction on audio | Depends on subcommand (`directory`, `file_list`, `dataset`) |
| `batdetect2 data` | Inspect and convert dataset configs | Depends on subcommand (`summary`, `convert`) |
| `batdetect2 train` | Train or fine-tune models | `TRAIN_DATASET` |
| `batdetect2 finetune` | Fine-tune a checkpoint on new targets | `TRAIN_DATASET` plus `--targets` |
| `batdetect2 evaluate` | Evaluate a checkpoint on a test dataset | `MODEL_PATH`, `TEST_DATASET` |
| `batdetect2 detect` | Legacy compatibility workflow | `AUDIO_DIR`, `ANN_DIR`, `DETECTION_THRESHOLD` |

## Notes

- Global CLI options are documented in {doc}`base`.
- Paths with spaces should be wrapped in quotes.
- Input audio is expected to be mono.
- `predict` uses the optional `--detection-threshold` override.
- `finetune` defaults to the bundled `uk_same` checkpoint if `--model` is not
  provided.

```{warning}
`batdetect2 detect` is a legacy command.
Prefer `batdetect2 predict directory` for new workflows.
```

## Related pages

- {doc}`../../tutorials/run-inference-on-folder`
- {doc}`../../how_to/run-batch-predictions`
- {doc}`../../how_to/tune-detection-threshold`
- {doc}`../configs`

```{toctree}
:maxdepth: 1

Base command and global options <base>
Predict command group <predict>
Data command group <data>
Train command <train>
Finetune command <finetune>
Evaluate command <evaluate>
Legacy detect command <detect_legacy>
```
