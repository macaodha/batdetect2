## Step 5: Generating Training Targets (Heatmaps)

### Purpose and Context

Following the previous steps of defining terms, filtering events, transforming tags, and defining specific class rules, this final stage focuses on **generating the ground truth data** used directly for training the BatDetect2 model.
This involves converting the refined annotation information for each audio clip into specific **heatmap formats** required by the underlying neural network architecture.

This step essentially translates your structured annotations into the precise "answer key" the model learns to replicate during training.

### What are Heatmaps?

Heatmaps, in this context, are multi-dimensional arrays, often visualized as images aligned with the input spectrogram, where the values at different time-frequency coordinates represent specific information about the sound events.
For BatDetect2 training, three primary heatmaps are generated:

1.  **Detection Heatmap:**

    - **Represents:** The presence or likelihood of relevant sound events across the spectrogram.
    - **Structure:** A 2D array matching the spectrogram's time-frequency dimensions.
      Peaks (typically smoothed) are generated at the reference locations of all sound events that passed the filtering stage (including both specifically classified events and those falling into the generic "Bat" category).

2.  **Class Heatmap:**

    - **Represents:** The location and class identity for sounds belonging to the _specific_ target classes you defined in Step 4.
    - **Structure:** A 3D array with dimensions for category, time, and frequency.
      It contains a separate 2D layer (channel) for each target class name (e.g., 'pippip', 'myodau').
      A smoothed peak appears on a specific class layer only if a sound event assigned to that class exists at that location.
      Events assigned only to the generic class do not produce peaks here.

3.  **Size Heatmap:**
    - **Represents:** The target dimensions (duration/width and bandwidth/height) of detected sound events.
    - **Structure:** A 3D array with dimensions for size-dimension ('width', 'height'), time, and frequency.
      At the reference location of each detected sound event, this heatmap stores two numerical values corresponding to the scaled width and height derived from the event's bounding box.

### How Heatmaps are Created (The Process)

The generation of these heatmaps is an automated process within `batdetect2`, driven by your configurations from all previous steps.
For each audio clip and its corresponding spectrogram in the training dataset:

1.  The system retrieves the associated sound event annotations.
2.  Configured **filtering rules** (Step 2) are applied to select relevant annotations.
3.  Configured **tag transformation rules** (Step 3) are applied to modify the tags of the selected annotations.
4.  Configured **class definition rules** (Step 4) are used to assign a specific class name or determine generic "Bat" status for each processed annotation.
5.  These final annotations are then mapped onto initialized heatmap arrays:
    - A signal (initially a single point) is placed on the **Detection Heatmap** at the reference location for each relevant annotation.
    - The scaled width and height values are placed on the **Size Heatmap** at the reference location.
    - If an annotation received a specific class name, a signal is placed on the corresponding layer of the **Class Heatmap** at the reference location.
6.  Finally, Gaussian smoothing (a blurring effect) is typically applied to the Detection and Class heatmaps to create spatially smoother targets, which often aids model training stability and performance.

### Configurable Settings for Heatmap Generation

While the content of the heatmaps is primarily determined by the previous configuration steps, a few parameters specific to the heatmap drawing process itself can be adjusted.
These are usually set in your main configuration file under a section like `labelling`:

- `sigma`: (Number, e.g., `3.0`) Defines the standard deviation, in pixels or bins, of the Gaussian kernel used for smoothing the Detection and Class heatmaps.
  Larger values result in more diffused heatmap peaks.
- `position`: (Text, e.g., `"bottom-left"`, `"center"`) Specifies the geometric reference point within each sound event's bounding box that anchors its representation on the heatmaps.
- `time_scale` & `frequency_scale`: (Numbers) These crucial scaling factors convert the physical duration (in seconds) and frequency bandwidth (in Hz) of annotation bounding boxes into the numerical values stored in the 'width' and 'height' channels of the Size Heatmap.
  - **Important Note:** The appropriate values for these scales are dictated by the requirements of the specific BatDetect2 model architecture being trained.
    They ensure the size information is presented in the units or relative scale the model expects.
    **Consult the documentation or tutorials for your specific model to determine the correct `time_scale` and `frequency_scale` values.** Mismatched scales can hinder the model's ability to learn size regression accurately.

**Example YAML Configuration for Labelling Settings:**

```yaml
# In your main configuration file
labelling:
  sigma: 3.0 # Std. dev. for Gaussian smoothing (pixels/bins)
  position: "bottom-left" # Bounding box reference point
  time_scale: 1000.0 # Example: Scales seconds to milliseconds
  frequency_scale: 0.00116 # Example: Scales Hz relative to ~860 Hz (model specific!)
```

### Outcome: Final Training Targets

Executing this step for all training data yields the complete set of target heatmaps (Detection, Class, Size) for each corresponding input spectrogram.
These arrays constitute the ground truth data that the BatDetect2 model directly compares its predictions against during the training phase, guiding its learning process.
