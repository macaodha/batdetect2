## Defining Target Geometry: Mapping Sound Event Regions

### Introduction

In the previous steps of defining targets, we focused on determining _which_ sound events are relevant (`filtering`), _what_ descriptive tags they should have (`transform`), and _which category_ they belong to (`classes`).
However, for the model to learn effectively, it also needs to know **where** in the spectrogram each sound event is located and approximately **how large** it is.

Your annotations typically define the location and extent of a sound event using a **Region of Interest (ROI)**, most commonly a **bounding box** drawn around the call on the spectrogram.
This ROI contains detailed spatial information (start/end time, low/high frequency).

This section explains how BatDetect2 converts the geometric ROI from your annotations into the specific positional and size information used as targets during model training.

### From ROI to Model Targets: Position & Size

BatDetect2 does not directly predict a full bounding box.
Instead, it is trained to predict:

1.  **A Reference Point:** A single point `(time, frequency)` that represents the primary location of the detected sound event within the spectrogram.
2.  **Size Dimensions:** Numerical values representing the event's size relative to that reference point, typically its `width` (duration in time) and `height` (bandwidth in frequency).

This step defines _how_ BatDetect2 calculates this specific reference point and these numerical size values from the original annotation's bounding box.
It also handles the reverse process – converting predicted positions and sizes back into bounding boxes for visualization or analysis.

### Configuring the ROI Mapping

You can control how this conversion happens through settings in your configuration file (e.g., your main `.yaml` file).
These settings are usually placed within the main `targets:` configuration block, under a specific `roi:` key.

Here are the key settings:

- **`position`**:

  - **What it does:** Determines which specific point on the annotation's bounding box is used as the single **Reference Point** for training (e.g., `"center"`, `"bottom-left"`).
  - **Why configure it?** This affects where the peak signal appears in the target heatmaps used for training.
    Different choices might slightly influence model learning.
    The default (`"bottom-left"`) is often a good starting point.
  - **Example Value:** `position: "center"`

- **`time_scale`**:

  - **What it does:** This is a numerical scaling factor that converts the _actual duration_ (width, measured in seconds) of the bounding box into the numerical 'width' value the model learns to predict (and which is stored in the Size Heatmap).
  - **Why configure it?** The model predicts raw numbers for size; this scale gives those numbers meaning.
    For example, setting `time_scale: 1000.0` means the model will be trained to predict the duration in **milliseconds** instead of seconds.
  - **Important Considerations:**
    - You can often set this value based on the units you prefer the model to work with internally.
      However, having target numerical values roughly centered around 1 (e.g., typically between 0.1 and 10) can sometimes improve numerical stability during model training.
    - The default value in BatDetect2 (e.g., `1000.0`) has been chosen to scale the duration relative to the spectrogram width under default STFT settings.
      Be aware that if you significantly change STFT parameters (window size or overlap), the relationship between the default scale and the spectrogram dimensions might change.
    - Crucially, whatever scale you use during training **must** be used when decoding the model's predictions back into real-world time units (seconds).
      BatDetect2 generally handles this consistency for you automatically when using the full pipeline.
  - **Example Value:** `time_scale: 1000.0`

- **`frequency_scale`**:
  - **What it does:** Similar to `time_scale`, this numerical scaling factor converts the _actual frequency bandwidth_ (height, typically measured in Hz or kHz) of the bounding box into the numerical 'height' value the model learns to predict.
  - **Why configure it?** It gives physical meaning to the model's raw numerical prediction for bandwidth and allows you to choose the internal units or scale.
  - **Important Considerations:**
    - Same as for `time_scale`.
  - **Example Value:** `frequency_scale: 0.00116`

**Example YAML Configuration:**

```yaml
# Inside your main configuration file (e.g., training_config.yaml)

targets: # Top-level key for target definition
  # ... filtering settings ...
  # ... transforms settings ...
  # ... classes settings ...

  # --- ROI Mapping Settings ---
  roi:
    position: "bottom-left" # Reference point (e.g., "center", "bottom-left")
    time_scale: 1000.0 # e.g., Model predicts width in ms
    frequency_scale: 0.00116 # e.g., Model predicts height relative to ~860Hz (or other model-specific scaling)
```

### Decoding Size Predictions

These scaling factors (`time_scale`, `frequency_scale`) are also essential for interpreting the model's output correctly.
When the model predicts numerical values for width and height, BatDetect2 uses these same scales (in reverse) to convert those numbers back into physically meaningful durations (seconds) and bandwidths (Hz/kHz) when reconstructing bounding boxes from predictions.

### Outcome

By configuring the `roi` settings, you ensure that BatDetect2 consistently translates the geometric information from your annotations into the specific reference points and scaled size values required for training the model.
Using consistent scales that are appropriate for your data and potentially beneficial for training stability allows the model to effectively learn not just _what_ sound is present, but also _where_ it is located and _how large_ it is, and enables meaningful interpretation of the model's spatial and size predictions.
