# `BatDetect2API` reference

`BatDetect2API` is the main entry point for the current Python workflow.

It wraps model loading, inference, evaluation, output formatting, and
training-related entry points behind one object.

Defined in `batdetect2.api_v2`.

## Create an API instance

- `BatDetect2API.from_checkpoint(path, ...)`
  - load a trained checkpoint and optional config overrides.
- `BatDetect2API.from_config(model_config=..., targets_config=..., ...)`
  - build a full stack from separate config objects.

## Inference methods

- `process_file(audio_file, ...)`
  - run inference for one recording.
- `process_files(audio_files, ...)`
  - run batch inference across a sequence of file paths.
- `process_directory(audio_dir, ...)`
  - run inference across the audio files found in one directory.
- `process_clips(clips, ...)`
  - run inference on an explicit sequence of clip objects.
- `process_audio(audio, ...)`
  - run inference starting from a waveform array.
- `process_spectrogram(spec, ...)`
  - run inference starting from a spectrogram tensor.

## Prediction inspection helpers

- `get_top_class_name(detection)`
  - return the highest-scoring class name for one detection.
- `get_class_scores(detection, include_top_class=True, sort_descending=True)`
  - return ranked `(class_name, score)` pairs.
- `get_detection_features(detection)`
  - return the per-detection feature vector.

## Audio loading helpers

- `load_audio(path)`
- `load_recording(recording)`
- `load_clip(clip)`
- `generate_spectrogram(audio)`

## Output persistence helpers

- `save_predictions(predictions, path, audio_dir=None, format=None,
  config=None)`
- `load_predictions(path, format=None, config=None)`

Use these when you want to save programmatic predictions without going through
the CLI.

## Training and evaluation entry points

- `train(...)`
- `finetune(...)`
- `evaluate(...)`
- `evaluate_predictions(...)`

## Related pages

- Python tutorial:
  {doc}`../tutorials/integrate-with-a-python-pipeline`
- Outputs config reference:
  {doc}`outputs-config`
- Output formats reference:
  {doc}`output-formats`
