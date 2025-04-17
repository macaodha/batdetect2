# Preprocessing Audio for BatDetect2

## What is Preprocessing?

Preprocessing refers to the steps taken to transform your raw audio recordings into a standardized format suitable for analysis by the BatDetect2 deep learning model.
This module (`batdetect2.preprocessing`) provides the tools to perform these transformations.

## Why is Preprocessing Important?

Applying a consistent preprocessing pipeline is important for several reasons:

1.  **Standardization:** Audio recordings vary significantly depending on the equipment used, recording conditions, and settings (e.g., different sample rates, varying loudness levels, background noise).
    Preprocessing helps standardize these aspects, making the data more uniform and allowing the model to learn relevant patterns more effectively.
2.  **Model Requirements:** Deep learning models, particularly those like BatDetect2 that analyze 2D-patterns in spectrograms, are designed to work with specific input characteristics.
    They often expect spectrograms of a certain size (time x frequency bins), with values represented on a particular scale (e.g., logarithmic/dB), and within a defined frequency range.
    Preprocessing ensures the data meets these requirements.
3.  **Consistency is Key:** Perhaps most importantly, the **exact same preprocessing steps** must be applied both when _training_ the model and when _using the trained model to make predictions_ (inference) on new data.
    Any discrepancy between the preprocessing used during training and inference can significantly degrade the model's performance and lead to unreliable results.
    BatDetect2's configurable pipeline ensures this consistency.

## How Preprocessing is Done in BatDetect2

BatDetect2 handles preprocessing through a configurable, two-stage pipeline:

1.  **Audio Loading & Preparation:** This first stage deals with the raw audio waveform.
    It involves loading the specified audio segment (from a file or clip), selecting a single channel (mono), optionally resampling it to a consistent sample rate, optionally adjusting its duration, and applying basic waveform conditioning like centering (DC offset removal) and amplitude scaling.
    (Details in the {doc}`audio` section).
2.  **Spectrogram Generation:** The prepared audio waveform is then converted into a spectrogram.
    This involves calculating the Short-Time Fourier Transform (STFT) and then applying a series of configurable steps like cropping the frequency range, applying amplitude representations (like dB scale or PCEN), optional denoising, optional resizing to the model's required dimensions, and optional final normalization.
    (Details in the {doc}`spectrogram` section).

The entire pipeline is controlled via settings in your main configuration file (typically a YAML file), usually grouped under a `preprocessing:` section which contains subsections like `audio:` and `spectrogram:`.
This allows you to easily define, share, and reproduce the exact preprocessing used for a specific model or experiment.

## Next Steps

Explore the following sections for detailed explanations of how to configure each stage of the preprocessing pipeline and how to use the resulting preprocessor:

```{toctree}
:maxdepth: 1
:caption: Preprocessing Steps:

audio
spectrogram
usage
```
