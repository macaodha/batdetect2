# Legacy Python API: `batdetect2.api`

This page documents the previous Python API workflow based on `batdetect2.api`.

```{warning}
This is documentation for a previous version of batdetect2.
For new workflows, use `batdetect2.BatDetect2API`.
If you are migrating, start with {doc}`migration-guide`.
```

## Using BatDetect2 in Python

If you prefer to process data inside a Python script, you can use the `batdetect2.api` module.

This interface gives you a simple entry point for running the built-in BatDetect2 model and also exposes the default model and default configuration more directly than the current API.

You can process a whole file in one step, or load audio, generate a spectrogram, and work with lower-level functions yourself.

Common functions:

- `process_file` Load an audio file, run the model, and return BatDetect2-style results for that recording.
- `process_audio` Run inference on an audio array that is already loaded in memory.
- `process_spectrogram` Run inference starting from a spectrogram tensor instead of raw audio.
- `load_audio` Load and resample audio using the legacy preprocessing path.
- `generate_spectrogram` Convert audio into the spectrogram representation expected by the model.
- `postprocess` Convert raw model outputs into detections and extracted features.

Typical usage:

```python
import batdetect2.api as api

AUDIO_FILE = "example_data/audio/20170701_213954-MYOMYS-LR_0_0.5.wav"

# Process a whole file
results = api.process_file(AUDIO_FILE)
annotations = results["pred_dict"]["annotation"]

# Or, load audio and compute spectrograms
audio = api.load_audio(AUDIO_FILE)
spec = api.generate_spectrogram(audio)

# And process the audio or the spectrogram with the model
detections, features, spec = api.process_audio(audio)
detections, features = api.process_spectrogram(spec)

# Integrate the detections or extracted features into your own analysis
```

This interface is most useful when you want to work directly with detections, features, spectrograms, or intermediate arrays inside your own code.

## Related pages

- Migration guide: {doc}`migration-guide`
- Current API reference: {doc}`../reference/api`
