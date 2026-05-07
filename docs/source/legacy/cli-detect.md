# CLI workflow: `batdetect2 detect`

This page documents the previous CLI workflow based on `batdetect2 detect`.

```{warning}
This is documentation for a previous version of batdetect2.
For new workflows, use `batdetect2 process directory` instead.
If you are migrating, start with {doc}`migration-guide`.
```

## Processing a folder of audio files

```bash
batdetect2 detect AUDIO_DIR ANN_DIR DETECTION_THRESHOLD
```

Example:

```bash
batdetect2 detect example_data/audio/ example_data/anns/ 0.3
```

This command scans a directory of audio files, runs the BatDetect2 detector on
each file, and writes BatDetect2-style outputs into `ANN_DIR`.
Those outputs usually include one JSON file and one CSV file per recording, and
can optionally include extra feature CSVs.

`AUDIO_DIR` is the folder containing the input `.wav` files.
`ANN_DIR` is the folder where model outputs are written.

`DETECTION_THRESHOLD` controls which detections are kept.
Predictions below this score are discarded.
Smaller values keep more detections, but usually also increase mistakes.

Common options:

- `--cnn_features` Write extra CNN feature CSV files for each recording.
- `--spec_features` Extract and write traditional acoustic spectrogram feature
  CSV files.
  These are saved as `*_spec_features.csv` files.
- `--time_expansion_factor` Set the time expansion factor used for all files in
  the run.
- `--save_preds_if_empty` Save output files even when no detections are found.
- `--model_path` Use a specific checkpoint instead of the included default
  model.
  If omitted, the command uses the default model trained on UK data.

## Related pages

- Migration guide:
  {doc}`migration-guide`
- Current process docs:
  {doc}`../reference/cli/predict`
