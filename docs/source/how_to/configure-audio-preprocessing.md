# How to configure audio preprocessing

Use this guide to set sample-rate and waveform-level preprocessing behaviour.

## 1) Set audio loader settings

The audio loader config controls resampling.

```yaml
samplerate: 256000
resample:
  enabled: true
  method: poly
```

If your recordings are already at the expected sample rate, you can disable
resampling.

```yaml
samplerate: 256000
resample:
  enabled: false
```

## 2) Set waveform transforms in preprocessing config

Waveform transforms are configured in `preprocess.audio_transforms`.

```yaml
preprocess:
  audio_transforms:
    - name: center_audio
    - name: scale_audio
    - name: fix_duration
      duration: 0.5
```

Available built-ins:

- `center_audio`
- `scale_audio`
- `fix_duration`

## 3) Use the config in your workflow

For CLI inference/evaluation, use `--audio-config`.

```bash
batdetect2 predict directory \
  path/to/model.ckpt \
  path/to/audio_dir \
  path/to/outputs \
  --audio-config path/to/audio.yaml
```

## 4) Verify quickly on a small subset

Run on a small folder first and confirm that outputs and runtime are as
expected before full-batch runs.

## Related pages

- Spectrogram settings: {doc}`configure-spectrogram-preprocessing`
- Preprocessing config reference: {doc}`../reference/preprocessing-config`
