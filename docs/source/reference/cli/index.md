# CLI reference

Use this section to find the right command quickly, then open the command page
for the full option list.

## Command map

| Command | Use it for | Required positional args |
| --- | --- | --- |
| `batdetect2 process` | Run inference on audio | Depends on subcommand (`directory`, `file_list`, `dataset`) |
| `batdetect2 data` | Inspect and convert dataset configs | Depends on subcommand (`summary`, `convert`) |
| `batdetect2 train` | Train or fine-tune models | `TRAIN_DATASET` |
| `batdetect2 finetune` | Fine-tune a checkpoint on new targets | `TRAIN_DATASET` plus `--targets` |
| `batdetect2 evaluate` | Evaluate a checkpoint on a test dataset | `TEST_DATASET` |
| `batdetect2 detect` | Legacy compatibility workflow | `AUDIO_DIR`, `ANN_DIR`, `DETECTION_THRESHOLD` |

## Notes

- Global CLI options are documented in {doc}`base`.
- Use `--log-file path/to/cli.log` to save CLI logs to a file while still
  showing them in the terminal.
- Paths with spaces should be wrapped in quotes.
- Input audio is expected to be mono.
- `process` uses the optional `--detection-threshold` override.
- `evaluate` takes `TEST_DATASET` as a positional argument and uses `--model`
  for the checkpoint override.
- `finetune` defaults to the bundled `uk_same` checkpoint if `--model` is not
  provided.

```{warning}
`batdetect2 detect` is a legacy command.
Prefer `batdetect2 process directory` for new workflows.
```

## Related pages

- {doc}`../../tutorials/run-inference-on-folder`
- {doc}`../../how_to/inference/run-batch-predictions`
- {doc}`../../how_to/inference/tune-detection-threshold`
- {doc}`../configs`

```{toctree}
:maxdepth: 1

Base command and global options <base>
Process command group <predict>
Data command group <data>
Train command <train>
Finetune command <finetune>
Evaluate command <evaluate>
Legacy detect command <detect_legacy>
```
