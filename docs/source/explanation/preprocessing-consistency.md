# Preprocessing consistency

Preprocessing consistency is one of the biggest factors behind stable model
performance.

## Why consistency matters

The detector is trained on spectrograms produced by a specific preprocessing
pipeline.
If inference uses different settings, the model can see a shifted input
distribution and performance may drop.

Typical mismatch sources:

- sample-rate differences,
- changed frequency crop,
- changed STFT window/hop,
- changed spectrogram transforms.

## Practical implication

When possible, keep preprocessing settings aligned between:

- training,
- evaluation,
- deployment inference.

If you intentionally change preprocessing, treat this as a new experiment and
re-validate on reviewed local data.

## Related pages

- Configure audio preprocessing:
  {doc}`../how_to/data/configure-audio-preprocessing`
- Configure spectrogram preprocessing:
  {doc}`../how_to/data/configure-spectrogram-preprocessing`
- Preprocessing config reference:
  {doc}`../reference/configs/data/preprocessing-config`
