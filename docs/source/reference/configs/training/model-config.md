# Model config reference

`ModelConfig` defines the model stack used for training or fresh model
construction.

Defined in `batdetect2.models`.

## Top-level fields

- `samplerate`
  - expected input sample rate.
- `architecture`
  - backbone network settings.
- `preprocess`
  - spectrogram preprocessing settings.
- `postprocess`
  - decoding and output filtering settings.

## What this config controls

Use `ModelConfig` when you want to change things like:

- the backbone architecture,
- the spectrogram settings used by the model,
- postprocessing settings stored with the model.

Example files live under `example_data/configs/`, including
`example_data/configs/model.yaml`.

## Related pages

- Preprocessing config:
  {doc}`../data/preprocessing-config`
- Postprocess config:
  {doc}`../data/postprocess-config`
- Train command reference:
  {doc}`../../cli/train`
