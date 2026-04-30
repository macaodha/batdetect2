# Migration guide: legacy to current workflows

Use this guide when moving from the previous BatDetect2 workflow to the current CLI and API.

## Who should migrate now

You should migrate if:

- you are starting a new workflow,
- you want the current docs path,
- you want the newer CLI and API surface,
- you are maintaining code that does not depend on the exact legacy JSON or feature outputs.

You may need the legacy workflow a bit longer if:

- downstream tooling depends on the exact old output structure,
- you rely on older notebooks built around `batdetect2.api`,
- you depend on legacy feature extraction outputs without a validated replacement yet.

## CLI mapping

- `batdetect2 detect AUDIO_DIR ANN_DIR DETECTION_THRESHOLD`
  -> `batdetect2 predict directory MODEL_PATH AUDIO_DIR OUTPUT_PATH --detection-threshold ...`

Main changes:

- the model path is now a positional argument on the `predict` subcommand,
- the current workflow expects an explicit checkpoint path rather than silently relying on the old default CLI behavior,
- output formatting is configurable,
- threshold override is an option rather than a required positional argument,
- there are separate subcommands for directory, file-list, and dataset-driven inference.

## Python API mapping

- old: `import batdetect2.api as api`
- current: `from batdetect2.api_v2 import BatDetect2API`

Typical migration shape:

```python
from pathlib import Path

from batdetect2.api_v2 import BatDetect2API

api = BatDetect2API.from_checkpoint(Path("path/to/model.ckpt"))
prediction = api.process_file(Path("path/to/audio.wav"))
```

Useful replacements:

- legacy `process_file` -> current `BatDetect2API.process_file`
- legacy `process_audio` -> current `BatDetect2API.process_audio`
- legacy `process_spectrogram` -> current `BatDetect2API.process_spectrogram`
- legacy one-off batch loops -> current `process_files` or CLI `predict`

## Output and terminology changes

Legacy workflows often centered on:

- BatDetect2-style JSON output,
- `cnn_feats`,
- `spec_features`,
- `spec_slices`.

Current workflows center on:

- `ClipDetections` and `Detection` objects,
- per-detection `detection_score`,
- per-detection `class_scores`,
- per-detection `features`,
- configurable output formatters.

## What to validate after migration

Before replacing a legacy workflow in production or research analysis, validate:

- that thresholds are still appropriate,
- that outputs are being saved in the right format,
- that downstream code reads the new outputs correctly,
- that feature-related assumptions still hold,
- that evaluation and ecological interpretation are unchanged only where you have actually verified that.

## Migration checklist

1. Identify the old entry points you use.
2. Replace them with the current CLI or `BatDetect2API` equivalents.
3. Choose an output format explicitly.
4. Re-run on a small reviewed subset.
5. Compare outputs and downstream behavior.
6. Update any notebooks or scripts that assume legacy field names.

## Related pages

- Current getting started: {doc}`../getting_started`
- Current tutorials: {doc}`../tutorials/index`
- Current API reference: {doc}`../reference/api`
