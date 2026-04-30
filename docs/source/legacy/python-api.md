# Legacy Python API: `batdetect2.api`

This page documents the previous Python API workflow based on `batdetect2.api`.

```{warning}
This is legacy documentation.
For new workflows, use `batdetect2.api_v2.BatDetect2API`.
If you are migrating, start with {doc}`migration-guide`.
```

## Legacy entry points

Common legacy functions included:

- `process_file`
- `process_audio`
- `process_spectrogram`
- `load_audio`
- `generate_spectrogram`
- `postprocess`

The legacy API also exposed the default model and default config more directly.

## Current replacement

The current Python path is:

```python
from pathlib import Path

from batdetect2.api_v2 import BatDetect2API

api = BatDetect2API.from_checkpoint(Path("path/to/model.ckpt"))
prediction = api.process_file(Path("path/to/audio.wav"))
```

## Related pages

- Migration guide: {doc}`migration-guide`
- Current API reference: {doc}`../reference/api`
