# BatDetect2 2.0 migration guide

Use this guide when moving from BatDetect2 1.x workflows to the CLI and API in
2.x.

## Why migrate

You get access to newer features.
The codebase changed quite a bit and now gives you much more control over the
workflow through config files, improved training and fine-tuning code, and a
more flexible sound target definition system.

You can also run newer or improved models.
That includes updated versions of the UK model, plus other models trained with
the newer codebase.

We are no longer actively supporting version 1.
No new enhancements are planned there, and only major bug fixes may still be
considered.
Future work is focused on version 2, including compatibility with newer Python
versions.

## Deprecation plan

We have kept the `batdetect2.api` module and the `batdetect2 detect` CLI command
in place for now.
You can keep using them without changing your current workflow.
However, many of the internal functions were relocated, removed or modified.
If your code relied on anything outside of the `api` module, it may break.
It is worth checking the new docs first, since there may already be a newer
feature that covers your use case.
If not, please open an issue.

Because the old `api` and CLI command are now redundant with the newer stack, we
plan to remove them in about a year.
If you want to keep pipelines up to date and long-running, it is a good idea to
migrate to version 2.

## How to migrate

If you are only using the `batdetect2 detect` CLI command or the
`batdetect2.api` module, the migration should be fairly simple.
This guide only covers these two entry points.

### CLI mapping

- `batdetect2 detect AUDIO_DIR ANN_DIR DETECTION_THRESHOLD` -> `batdetect2
  process directory AUDIO_DIR OUTPUT_PATH --detection-threshold
  DETECTION_THRESHOLD ...`

Main changes:

- outputs can be written in different formats.
  See the output format reference for the available options.
- the detection threshold is now an option instead of a required positional
  argument.
- options like saving CNN features are now controlled through config rather than
  command flags.
- there are separate subcommands for processing a directory, file list, or
  dataset.

### Python API mapping

- old:
  `import batdetect2.api as api`
- current:
  `from batdetect2 import BatDetect2API`

Typical migration shape:

```python
from pathlib import Path

from batdetect2 import BatDetect2API

# If no checkpoint is provided, the default UK model is loaded
api = BatDetect2API.from_checkpoint()
prediction = api.process_file(Path("path/to/audio.wav"))
```

Useful replacements:

- legacy `process_file` -> current `BatDetect2API.process_file`
- legacy `process_audio` -> current `BatDetect2API.process_audio`
- legacy `process_spectrogram` -> current `BatDetect2API.process_spectrogram`
- legacy one-off batch loops -> current `process_files` or CLI `process`

### Model changes

The default checkpoint used by the new CLI `process` commands and by
`BatDetect2API` is a newer model trained from scratch using the updated training
code, but the same model architecture, training procedure, and data.
Performance did not change substantially, but some differences are still
expected.

### Species names

For the default UK model there are two naming changes:

1. The original model had a typo and instead of `Barbastella barbastellus` it
   used `Barbastellus barbastellus`.
   This has now been corrected.
2. There has been a recent change in name for `Eptesicus serotinus` to
   `Cnephaeus serotinus`.

## Stay on version 1

If you prefer not to migrate to version 2 yet, you can keep using version 1.
In that case, it is a good idea to pin your dependency:

```bash
pip install "batdetect2>=1.3.1,<2"
```

## Related pages

- Getting started:
  {doc}`../getting_started`
- Tutorials:
  {doc}`../tutorials/index`
- API reference:
  {doc}`../reference/api`
