# Postprocessing: From Model Output to Predictions

## What is Postprocessing?

After the BatDetect2 neural network analyzes a spectrogram, it doesn't directly output a neat list of bat calls.
Instead, it produces raw numerical data, usually in the form of multi-dimensional arrays or "heatmaps".
These arrays contain information like:

- The probability of a sound event being present at each time-frequency location.
- The probability of each possible target class (e.g., species) at each location.
- Predicted size characteristics (like duration and bandwidth) at each location.
- Internal learned features at each location.

**Postprocessing** is the sequence of steps that takes these numerical model outputs and translates them into a structured list of detected sound events, complete with predicted tags, bounding boxes, and confidence scores.
The {py:mod}`batdetect2.postprocess` mode handles this entire workflow.

## Why is Postprocessing Necessary?

1.  **Interpretation:** Raw heatmap outputs need interpretation to identify distinct sound events (detections).
    A high probability score might spread across several adjacent time-frequency bins, all related to the same call.
2.  **Refinement:** Model outputs can be noisy or contain redundancies.
    Postprocessing steps like Non-Maximum Suppression (NMS) clean this up, ensuring (ideally) only one detection is reported for each actual sound event.
3.  **Contextualization:** Raw outputs lack real-world units.
    Postprocessing adds back time (seconds) and frequency (Hz) coordinates, converts predicted sizes to physical units using configured scales, and decodes predicted class indices back into meaningful tags based on your target definitions.
4.  **User Control:** Postprocessing includes tunable parameters, most importantly **thresholds**.
    By adjusting these, you can control the trade-off between finding more potential calls (sensitivity) versus reducing false positives (specificity) _without needing to retrain the model_.

## The Postprocessing Pipeline

BatDetect2 applies a series of steps to convert the raw model output into final predictions.
Understanding these steps helps interpret the results and configure the process effectively:

1.  **Non-Maximum Suppression (NMS):**

    - **Goal:** Reduce redundant detections.
      If the model outputs high scores for several nearby points corresponding to the same call, NMS selects the single highest peak in a local neighbourhood and suppresses the others (sets their score to zero).
    - **Configurable:** The size of the neighbourhood (`nms_kernel_size`) can be adjusted.

2.  **Coordinate Remapping:**

    - **Goal:** Add coordinate (time/frequency) information.
      This step takes the grid-based model outputs (which just have row/column indices) and associates them with actual time (seconds) and frequency (Hz) coordinates based on the input spectrogram's properties.
      The result is coordinate-aware arrays (using {py:class}`xarray.DataArray`}).

3.  **Detection Extraction:**

    - **Goal:** Identify the specific points representing detected events.
    - **Process:** Looks for peaks in the NMS-processed detection heatmap that are above a certain confidence level (`detection_threshold`).
      It also often limits the maximum number of detections based on a rate (`top_k_per_sec`) to avoid excessive outputs in very busy files.
    - **Configurable:** `detection_threshold`, `top_k_per_sec`.

4.  **Data Extraction:**

    - **Goal:** Gather all relevant information for each detected point.
    - **Process:** For each time-frequency location identified in Step 3, this step looks up the corresponding values in the _other_ remapped model output arrays (class probabilities, predicted sizes, internal features).
    - **Intermediate Output 1:** The result of this stage (containing aligned scores, positions, sizes, class probabilities, and features for all detections in a clip) is often accessible programmatically as an {py:class}`xarray.Dataset`}.
      This can be useful for advanced users needing direct access to the numerical outputs.

5.  **Decoding & Formatting:**

    - **Goal:** Convert the extracted numerical data into interpretable, standard formats.
    - **Process:**
      - **ROI Recovery:** Uses the predicted position and size values, along with the ROI mapping configuration defined in the `targets` module, to reconstruct an estimated bounding box ({py:class}`soundevent.data.BoundingBox`}).
      - **Class Decoding:** Translates the numerical class probability vector into meaningful {py:class}`soundevent.data.PredictedTag` objects.
        This involves:
        - Applying the `classification_threshold` to ignore low-confidence class scores.
        - Using the class decoding rules (from the `targets` module) to map the name(s) of the high-scoring class(es) back to standard tags (like `species: Myotis daubentonii`).
        - Optionally selecting only the top-scoring class or multiple classes above the threshold.
        - Including the generic "Bat" tags if no specific class meets the threshold.
      - **Feature Conversion:** Converts raw feature vectors into {py:class}`soundevent.data.Feature` objects.
    - **Intermediate Output 2:** This step might internally create a list of simplified `RawPrediction` objects containing the bounding box, scores, and features.
      This intermediate list might also be accessible programmatically for users who prefer a simpler structure than the final {py:mod}`soundevent` objects.

6.  **Final Output (`ClipPrediction`):**
    - **Goal:** Package everything into a standard format.
    - **Process:** Collects all the fully processed `SoundEventPrediction` objects (each containing a sound event with geometry, features, overall score, and predicted tags with scores) for a given audio clip into a final {py:class}`soundevent.data.ClipPrediction` object.
      This is the standard output format representing the model's findings for that clip.

## Configuring Postprocessing

You can control key aspects of this pipeline, especially the thresholds and NMS settings, via a `postprocess:` section in your main configuration YAML file.
Adjusting these **allows you to fine-tune the detection results without retraining the model**.

**Key Configurable Parameters:**

- `detection_threshold`: (Number >= 0, e.g., `0.1`) Minimum score for a peak to be considered a detection.
  **Lowering this increases sensitivity (more detections, potentially more false positives); raising it increases specificity (fewer detections, potentially missing faint calls).**
- `classification_threshold`: (Number >= 0, e.g., `0.3`) Minimum score for a _specific class_ prediction to be assigned as a tag.
  Affects how confidently the model must identify the class.
- `top_k_per_sec`: (Integer > 0, e.g., `200`) Limits the maximum density of detections reported per second.
  Helps manage extremely dense recordings.
- `nms_kernel_size`: (Integer > 0, e.g., `9`) Size of the NMS window in pixels/bins.
  Affects how close two distinct peaks can be before one suppresses the other.

**Example YAML Configuration:**

```yaml
# Inside your main configuration file (e.g., config.yaml)

postprocess:
  nms_kernel_size: 9
  detection_threshold: 0.1 # Lower threshold -> more sensitive
  classification_threshold: 0.3 # Higher threshold -> more confident classifications
  top_k_per_sec: 200
# ... other sections preprocessing, targets ...
```

**Note:** These parameters can often also be adjusted via Command Line Interface (CLI) arguments when running predictions, or through function arguments if using the Python API, providing flexibility for experimentation.

## Accessing Intermediate Results

While the final `ClipPrediction` objects are the standard output, the `Postprocessor` object used internally provides methods to access results from intermediate stages (like the `xr.Dataset` after Step 4, or the list of `RawPrediction` objects after Step 5).

This can be valuable for:

- Debugging the pipeline.
- Performing custom analyses on the numerical outputs before final decoding.
- **Transfer Learning / Feature Extraction:** Directly accessing the extracted `features` (from Step 4 or 5a) associated with detected events can be highly useful for training other models or further analysis.

Consult the API documentation for details on how to access these intermediate results programmatically if needed.

## Summary

Postprocessing is the conversion between neural network outputs and meaningful, interpretable sound event detections.
BatDetect2 provides a configurable pipeline including NMS, coordinate remapping, peak detection with thresholding, data extraction, and class/geometry decoding.
Researchers can easily tune key parameters like thresholds via configuration files or arguments to adjust the final set of predictions without altering the trained model itself, and advanced users can access intermediate results for custom analyses or feature reuse.
