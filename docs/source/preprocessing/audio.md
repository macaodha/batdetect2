# Audio Loading and Preprocessing

## Purpose

Before BatDetect2 can analyze the sounds in your recordings, the raw audio data needs to be loaded from the file and prepared.
This initial preparation involves several standard waveform processing steps.
This `audio` module handles this first stage of preprocessing.

It's crucial to understand that the _exact same_ preprocessing steps must be applied both when **training** a model and when **using** that trained model later to make predictions (inference).
Consistent preprocessing ensures the model receives data in the format it expects.

BatDetect2 allows you to control these audio preprocessing steps through settings in your main configuration file.

## The Audio Processing Pipeline

When BatDetect2 needs to process an audio segment (either a full recording or a specific clip), it follows a defined sequence of steps:

1.  **Load Audio Segment:** The system first reads the specified time segment from the audio file.
    - **Note:** BatDetect2 typically works with **mono** audio.
      By default, if your file has multiple channels (e.g., stereo), only the **first channel** is loaded and used for subsequent processing.
2.  **Adjust Duration (Optional):** If you've specified a target duration in your configuration, the loaded audio segment is either shortened (by cropping from the start) or lengthened (by adding silence, i.e., zeros, at the end) to match that exact duration.
    This is sometimes required by specific model architectures that expect fixed-size inputs.
    By default, this step is **off**, and the original clip duration is used.
3.  **Resample (Optional):** If configured (and usually **on** by default), the audio's sample rate is changed to a specific target value (e.g., 256,000 Hz).
    This is vital for standardizing the data, as different recording devices capture audio at different rates.
    The model needs to be trained and run on data with a consistent sample rate.
4.  **Center Waveform (Optional):** If configured (and typically **on** by default), the system removes any constant shift away from zero in the waveform (known as DC offset).
    This is a standard practice that can sometimes improve the quality of later signal processing steps.
5.  **Scale Amplitude (Optional):** If configured (typically **off** by default), the waveform's amplitude (loudness) is adjusted.
    The standard method used here is "peak normalization," which scales the entire clip so that the loudest point has an absolute value of 1.0.
    This can help standardize volume levels across different recordings, although it's not always necessary or desirable depending on your analysis goals.

## Configuring Audio Processing

You can control these steps via settings in your main configuration file (e.g., `config.yaml`), usually within a dedicated `audio:` section (which might itself be under a broader `preprocessing:` section).

Here are the key options you can set:

- **Resampling (`resample`)**:

  - To enable resampling (recommended and usually default), include a `resample:` block.
    To disable it completely, you might set `resample: null` or omit the block.
  - `samplerate`: (Number) The target sample rate in Hertz (Hz) that all audio will be converted to.
    This **must** match the sample rate expected by the BatDetect2 model you are using or training (e.g., `samplerate: 256000`).
  - `mode`: (Text, `"poly"` or `"fourier"`) The underlying algorithm used for resampling.
    The default `"poly"` is generally a good choice.
    You typically don't need to change this unless you have specific reasons.

- **Duration (`duration`)**:

  - (Number or `null`) Sets a fixed duration for all audio clips in **seconds**.
    If set (e.g., `duration: 4.0`), shorter clips are padded with silence, and longer clips are cropped.
    If `null` (default), the original clip duration is used.

- **Centering (`center`)**:

  - (Boolean, `true` or `false`) Controls DC offset removal.
    Default is usually `true`.
    Set to `false` to disable.

- **Scaling (`scale`)**:
  - (Boolean, `true` or `false`) Controls peak amplitude normalization.
    Default is usually `false`.
    Set to `true` to enable scaling so the maximum absolute amplitude becomes 1.0.

**Example YAML Configuration:**

```yaml
# Inside your main configuration file (e.g., training_config.yaml)

preprocessing: # Or this might be at the top level
  audio:
    # --- Resampling Settings ---
    resample: # Settings block to control resampling
      samplerate: 256000 # Target sample rate in Hz (Required if resampling)
      mode: poly # Algorithm ('poly' or 'fourier', optional, defaults to 'poly')
      # To disable resampling entirely, you might use:
      # resample: null

    # --- Other Settings ---
    duration: null # Keep original clip duration (e.g., use 4.0 for 4 seconds)
    center: true # Remove DC offset (default is often true)
    scale: false # Do not normalize peak amplitude (default is often false)

# ... other configuration sections (like model, dataset, targets) ...
```

## Outcome

After these steps, the output is a standardized audio waveform (represented as a numerical array with time information).
This processed waveform is now ready for the next stage of preprocessing, which typically involves calculating the spectrogram (covered in the next module/section).
Ensuring these audio preprocessing settings are consistent is fundamental for achieving reliable results in both training and inference.
