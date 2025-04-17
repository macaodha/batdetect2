# Using Preprocessors in BatDetect2

## Overview

In the previous sections ({doc}`audio`and {doc}`spectrogram`), we discussed the individual steps involved in converting raw audio into a processed spectrogram suitable for BatDetect2 models, and how to configure these steps using YAML files (specifically the `audio:` and `spectrogram:` sections within a main `preprocessing:` configuration block).

This page focuses on how this configured pipeline is represented and used within BatDetect2, primarily through the concept of a **`Preprocessor`** object.
This object bundles together your chosen audio loading settings and spectrogram generation settings into a single component that can perform the end-to-end processing.

## Do I Need to Interact with Preprocessors Directly?

**Usually, no.** For standard model training or running inference with BatDetect2 using the provided scripts, the system will automatically:

1.  Read your main configuration file (e.g., `config.yaml`).
2.  Find the `preprocessing:` section (containing `audio:` and `spectrogram:` settings).
3.  Build the appropriate `Preprocessor` object internally based on your settings.
4.  Use that internal `Preprocessor` object automatically whenever audio needs to be loaded and converted to a spectrogram.

**However**, understanding the `Preprocessor` object is useful if you want to:

- **Customize:** Go beyond the standard configuration options by interacting with parts of the pipeline programmatically.
- **Integrate:** Use BatDetect2's preprocessing steps within your own custom Python analysis scripts.
- **Inspect/Debug:** Manually apply preprocessing steps to specific files or clips to examine intermediate outputs (like the processed waveform) or the final spectrogram.

## Getting a Preprocessor Object

If you _do_ want to work with the preprocessor programmatically, you first need to get an instance of it.
This is typically done based on a configuration:

1.  **Define Configuration:** Create your `preprocessing:` configuration, usually in a YAML file (let's call it `preprocess_config.yaml`), detailing your desired `audio` and `spectrogram` settings.

    ```yaml
    # preprocess_config.yaml
    audio:
      resample:
        samplerate: 256000
      # ... other audio settings ...
    spectrogram:
      frequencies:
        min_freq: 15000
        max_freq: 120000
      scale: dB
      # ... other spectrogram settings ...
    ```

2.  **Load Configuration & Build Preprocessor (in Python):**

    ```python
    from batdetect2.preprocessing import load_preprocessing_config, build_preprocessor
    from batdetect2.preprocess.types import Preprocessor # Import the type

    # Load the configuration from the file
    config_path = "path/to/your/preprocess_config.yaml"
    preprocessing_config = load_preprocessing_config(config_path)

    # Build the actual preprocessor object using the loaded config
    preprocessor: Preprocessor = build_preprocessor(preprocessing_config)

    # 'preprocessor' is now ready to use!
    ```

3.  **Using Defaults:** If you just want the standard BatDetect2 default preprocessing settings, you can build one without loading a config file:

    ```python
    from batdetect2.preprocessing import build_preprocessor
    from batdetect2.preprocess.types import Preprocessor

    # Build with default settings
    default_preprocessor: Preprocessor = build_preprocessor()
    ```

## Applying Preprocessing

Once you have a `preprocessor` object, you can use its methods to process audio data:

**1.
End-to-End Processing (Common Use Case):**

These methods take an audio source identifier (file path, Recording object, or Clip object) and return the final, processed spectrogram.

- `preprocessor.preprocess_file(path)`: Processes an entire audio file.
- `preprocessor.preprocess_recording(recording_obj)`: Processes the entire audio associated with a `soundevent.data.Recording` object.
- `preprocessor.preprocess_clip(clip_obj)`: Processes only the specific time segment defined by a `soundevent.data.Clip` object.
  - **Efficiency Note:** Using `preprocess_clip` is **highly recommended** when you are only interested in analyzing a small portion of a potentially long recording.
    It avoids loading the entire audio file into memory, making it much more efficient.

```python
from soundevent import data

# Assume 'preprocessor' is built as shown before
# Assume 'my_clip' is a soundevent.data.Clip object defining a segment

# Process an entire file
spectrogram_from_file = preprocessor.preprocess_file("my_recording.wav")

# Process only a specific clip (more efficient for segments)
spectrogram_from_clip = preprocessor.preprocess_clip(my_clip)

# The results (spectrogram_from_file, spectrogram_from_clip) are xr.DataArrays
print(type(spectrogram_from_clip))
# Output: <class 'xarray.core.dataarray.DataArray'>
```

**2.
Intermediate Steps (Advanced Use Cases):**

The preprocessor also allows access to intermediate stages if needed:

- `preprocessor.load_clip_audio(clip_obj)` (and similar for file/recording): Loads the audio and applies _only_ the waveform processing steps (resampling, centering, etc.) defined in the `audio` config.
  Returns the processed waveform as an `xr.DataArray`.
  This is useful if you want to analyze or manipulate the waveform itself before spectrogram generation.
- `preprocessor.compute_spectrogram(waveform)`: Takes an _already loaded_ waveform (either `np.ndarray` or `xr.DataArray`) and applies _only_ the spectrogram generation steps defined in the `spectrogram` config.
  - If you provide an `xr.DataArray` (e.g., from `load_clip_audio`), it uses the sample rate from the array's coordinates.
  - If you provide a raw `np.ndarray`, it **must assume a sample rate**.
    It uses the `default_samplerate` that was determined when the `preprocessor` was built (based on your `audio` config's resample settings or the global default).
    Be cautious when using NumPy arrays to ensure the sample rate assumption is correct for your data!

```python
# Example: Get waveform first, then spectrogram
waveform = preprocessor.load_clip_audio(my_clip)
# waveform is an xr.DataArray

# ...potentially do other things with the waveform...

# Compute spectrogram from the loaded waveform
spectrogram = preprocessor.compute_spectrogram(waveform)

# Example: Process external numpy array (use with caution re: sample rate)
# import soundfile as sf # Requires installing soundfile
# numpy_waveform, original_sr = sf.read("some_other_audio.wav")
# # MUST ensure numpy_waveform's actual sample rate matches
# # preprocessor.default_samplerate for correct results here!
# spec_from_numpy = preprocessor.compute_spectrogram(numpy_waveform)
```

## Understanding the Output: `xarray.DataArray`

All preprocessing methods return the final spectrogram (or the intermediate waveform) as an **`xarray.DataArray`**.

**What is it?** Think of it like a standard NumPy array (holding the numerical data of the spectrogram) but with added "superpowers":

- **Labeled Dimensions:** Instead of just having axis 0 and axis 1, the dimensions have names, typically `"frequency"` and `"time"`.
- **Coordinates:** It stores the actual frequency values (e.g., in Hz) corresponding to each row and the actual time values (e.g., in seconds) corresponding to each column along the dimensions.

**Why is it used?**

- **Clarity:** The data is self-describing.
  You don't need to remember which axis is time and which is frequency, or what the units are – it's stored with the data.
- **Convenience:** You can select, slice, or plot data using the real-world coordinate values (times, frequencies) instead of just numerical indices.
  This makes analysis code easier to write and less prone to errors.
- **Metadata:** It can hold additional metadata about the processing steps in its `attrs` (attributes) dictionary.

**Using the Output:**

- **Input to Model:** For standard training or inference, you typically pass this `xr.DataArray` spectrogram directly to the BatDetect2 model functions.
- **Inspection/Analysis:** If you're working programmatically, you can use xarray's powerful features.
  For example (these are just illustrations of xarray):

  ```python
  # Get the shape (frequency_bins, time_bins)
  # print(spectrogram.shape)

  # Get the frequency coordinate values
  # print(spectrogram['frequency'].values)

  # Select data near a specific time and frequency
  # value_at_point = spectrogram.sel(time=0.5, frequency=50000, method="nearest")
  # print(value_at_point)

  # Select a time slice between 0.2 and 0.3 seconds
  # time_slice = spectrogram.sel(time=slice(0.2, 0.3))
  # print(time_slice.shape)
  ```

In summary, while BatDetect2 often handles preprocessing automatically based on your configuration, the underlying `Preprocessor` object provides a flexible interface for applying these steps programmatically if needed, returning results in the convenient and informative `xarray.DataArray` format.
