# Preprocessing config reference

This page summarizes preprocessing-related config objects used by batdetect2.

## Audio loader config (`AudioConfig`)

Defined in `batdetect2.audio.loader`.

Fields:

- `samplerate` (int): target audio sample rate in Hz.
- `resample.enabled` (bool): whether to resample loaded audio.
- `resample.method` (`poly` or `fourier`): resampling method.

## Model preprocessing config (`PreprocessingConfig`)

Defined in `batdetect2.preprocess.config`.

Top-level fields:

- `audio_transforms`: ordered waveform transforms.
- `stft`: STFT parameters.
- `frequencies`: spectrogram frequency range.
- `spectrogram_transforms`: ordered spectrogram transforms.
- `size`: final resize settings.

### `audio_transforms` built-ins

- `center_audio`
- `scale_audio`
- `fix_duration` (`duration` in seconds)

### `stft` fields

- `window_duration`
- `window_overlap`
- `window_fn`

### `frequencies` fields

- `min_freq`
- `max_freq`

### `spectrogram_transforms` built-ins

- `pcen`
- `scale_amplitude` (`scale: db|power`)
- `spectral_mean_subtraction`
- `peak_normalize`

### `size` fields

- `height`
- `resize_factor`

## Related pages

- Audio preprocessing how-to: {doc}`../how_to/configure-audio-preprocessing`
- Spectrogram preprocessing how-to:
  {doc}`../how_to/configure-spectrogram-preprocessing`
- Why consistency matters: {doc}`../explanation/preprocessing-consistency`
