# Spectrogram Generation

## Purpose

After loading and performing initial processing on the audio waveform (as described in the Audio Loading section), the next crucial step in the `preprocessing` pipeline is to convert that waveform into a **spectrogram**.
A spectrogram is a visual representation of sound, showing frequency content over time, and it's the primary input format for many deep learning models, including BatDetect2.

This module handles the calculation and subsequent processing of the spectrogram.
Just like the audio processing, these steps need to be applied **consistently** during both model training and later use (inference) to ensure the model performs reliably.
You control this entire process through the configuration file.

## The Spectrogram Generation Pipeline

Once BatDetect2 has a prepared audio waveform, it follows these steps to create the final spectrogram input for the model:

1.  **Calculate STFT (Short-Time Fourier Transform):** This is the fundamental step that converts the 1D audio waveform into a 2D time-frequency representation.
    It calculates the frequency content within short, overlapping time windows.
    The output is typically a **magnitude spectrogram**, showing the intensity (amplitude) of different frequencies at different times.
    Key parameters here are the `window_duration` and `window_overlap`, which affect the trade-off between time and frequency resolution.
2.  **Crop Frequencies:** The STFT often produces frequency information over a very wide range (e.g., 0 Hz up to half the sample rate).
    This step crops the spectrogram to focus only on the frequency range relevant to your target sounds (e.g., 10 kHz to 120 kHz for typical bat echolocation).
3.  **Apply PCEN (Optional):** If configured, Per-Channel Energy Normalization is applied.
    PCEN is an adaptive technique that adjusts the gain (loudness) in each frequency channel based on its recent history.
    It can help suppress stationary background noise and enhance the prominence of transient sounds like echolocation pulses.
    This step is optional.
4.  **Set Amplitude Scale / Representation:** The values in the spectrogram (either raw magnitude or post-PCEN values) need to be represented on a suitable scale.
    You choose one of the following:
    - `"amplitude"`: Use the linear magnitude values directly.
      (Default)
    - `"power"`: Use the squared magnitude values (representing energy).
    - `"dB"`: Apply a logarithmic transformation (specifically `log(1 + C*Magnitude)`).
      This compresses the range of values, often making variations in quieter sounds more apparent, similar to how humans perceive loudness.
5.  **Denoise (Optional):** If configured (and usually **on** by default), a simple noise reduction technique is applied.
    This method subtracts the average value of each frequency bin (calculated across time) from that bin, assuming the average represents steady background noise.
    Negative values after subtraction are clipped to zero.
6.  **Resize (Optional):** If configured, the dimensions (height/frequency bins and width/time bins) of the spectrogram are adjusted using interpolation to match the exact input size expected by the neural network architecture.
7.  **Peak Normalize (Optional):** If configured (typically **off** by default), the entire final spectrogram is scaled so that its highest value is exactly 1.0.
    This ensures all spectrograms fed to the model have a consistent maximum value, which can sometimes aid training stability.

## Configuring Spectrogram Generation

You control all these steps via settings in your main configuration file (e.g., `config.yaml`), within the `spectrogram:` section (usually located under the main `preprocessing:` section).

Here are the key configuration options:

- **STFT Settings (`stft`)**:

  - `window_duration`: (Number, seconds, e.g., `0.002`) Length of the analysis window.
  - `window_overlap`: (Number, 0.0 to <1.0, e.g., `0.75`) Fractional overlap between windows.
  - `window_fn`: (Text, e.g., `"hann"`) Name of the windowing function.

- **Frequency Cropping (`frequencies`)**:

  - `min_freq`: (Integer, Hz, e.g., `10000`) Minimum frequency to keep.
  - `max_freq`: (Integer, Hz, e.g., `120000`) Maximum frequency to keep.

- **PCEN (`pcen`)**:

  - This entire section is **optional**.
    Include it only if you want to apply PCEN.
    If omitted or set to `null`, PCEN is skipped.
  - `time_constant`: (Number, seconds, e.g., `0.4`) Controls adaptation speed.
  - `gain`: (Number, e.g., `0.98`) Gain factor.
  - `bias`: (Number, e.g., `2.0`) Bias factor.
  - `power`: (Number, e.g., `0.5`) Compression exponent.

- **Amplitude Scale (`scale`)**:

  - (Text: `"dB"`, `"power"`, or `"amplitude"`) Selects the final representation of the spectrogram values.
    Default is `"amplitude"`.

- **Denoising (`spectral_mean_substraction`)**:

  - (Boolean: `true` or `false`) Enables/disables the spectral mean subtraction denoising step.
    Default is usually `true`.

- **Resizing (`size`)**:

  - This entire section is **optional**.
    Include it only if you need to resize the spectrogram to specific dimensions required by the model.
    If omitted or set to `null`, no resizing occurs after frequency cropping.
  - `height`: (Integer, e.g., `128`) Target number of frequency bins.
  - `resize_factor`: (Number or `null`, e.g., `0.5`) Factor to scale the time dimension by.
    `0.5` halves the width, `null` or `1.0` keeps the original width.

- **Peak Normalization (`peak_normalize`)**:
  - (Boolean: `true` or `false`) Enables/disables final scaling of the entire spectrogram so the maximum value is 1.0.
    Default is usually `false`.

**Example YAML Configuration:**

```yaml
# Inside your main configuration file

preprocessing:
  audio:
    # ... (your audio configuration settings) ...
    resample:
      samplerate: 256000 # Ensure this matches model needs

  spectrogram:
    # --- STFT Parameters ---
    stft:
      window_duration: 0.002 # 2ms window
      window_overlap: 0.75 # 75% overlap
      window_fn: hann

    # --- Frequency Range ---
    frequencies:
      min_freq: 10000 # 10 kHz
      max_freq: 120000 # 120 kHz

    # --- PCEN (Optional) ---
    # Include this block to enable PCEN, omit or set to null to disable.
    pcen:
      time_constant: 0.4
      gain: 0.98
      bias: 2.0
      power: 0.5

    # --- Final Amplitude Representation ---
    scale: dB # Choose 'dB', 'power', or 'amplitude'

    # --- Denoising ---
    spectral_mean_substraction: true # Enable spectral mean subtraction

    # --- Resizing (Optional) ---
    # Include this block to resize, omit or set to null to disable.
    size:
      height: 128 # Target height in frequency bins
      resize_factor: 0.5 # Halve the number of time bins

    # --- Final Normalization ---
    peak_normalize: false # Do not scale max value to 1.0
```

## Outcome

The output of this module is the final, processed spectrogram (as a 2D numerical array with time and frequency information).
This spectrogram is now in the precise format expected by the BatDetect2 neural network, ready to be used for training the model or for making predictions on new data.
Remember, using the exact same `spectrogram` configuration settings during training and inference is essential for correct model performance.
