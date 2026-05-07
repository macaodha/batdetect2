# How to configure spectrogram preprocessing

Use this guide to set STFT, frequency range, and spectrogram transforms.

## 1) Configure STFT and frequency range

```yaml
preprocess:
  stft:
    window_duration: 0.002
    window_overlap: 0.75
    window_fn: hann
  frequencies:
    min_freq: 10000
    max_freq: 120000
```

## 2) Configure spectrogram transforms

`spectrogram_transforms` are applied in order.

```yaml
preprocess:
  spectrogram_transforms:
    - name: pcen
      time_constant: 0.4
      gain: 0.98
      bias: 2.0
      power: 0.5
    - name: spectral_mean_subtraction
    - name: scale_amplitude
      scale: db
```

Common built-ins:

- `pcen`
- `spectral_mean_subtraction`
- `scale_amplitude` (`db` or `power`)
- `peak_normalize`

## 3) Configure output size

```yaml
preprocess:
  size:
    height: 128
    resize_factor: 0.5
```

## 4) Keep train and inference settings aligned

Use the same preprocessing setup for training and prediction whenever possible.
Large mismatches can degrade model performance.

## Related pages

- Why consistency matters: {doc}`../explanation/preprocessing-consistency`
- Preprocessing config reference: {doc}`../reference/preprocessing-config`
