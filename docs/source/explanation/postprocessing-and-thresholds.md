# Postprocessing and thresholds

After the detector runs on a spectrogram, the model output is still a set of
dense prediction tensors. Postprocessing turns that into a final list of call
detections with positions, sizes, and class scores.

## What postprocessing does

In broad terms, the pipeline:

1. suppresses nearby duplicate peaks,
2. extracts candidate detections,
3. reads size and class values at each detected location,
4. decodes outputs into call-level predictions.

This is where score thresholds and output density limits are applied.

## Why thresholds matter

Thresholds control the balance between sensitivity and precision.

- Lower thresholds keep more detections, including weaker calls, but may add
  false positives.
- Higher thresholds remove low-confidence detections, but may miss faint calls.

You can tune this behavior per run without retraining the model.

## Two common threshold controls

- `detection_threshold`: minimum score required to keep a detection.
- `classification_threshold`: minimum class score used when assigning class
  labels.

Both settings shape the final output and should be validated on reviewed local
data.

## Practical workflow

Tune thresholds on a representative subset first, then lock settings for the
full analysis run.

- How-to: {doc}`../how_to/tune-detection-threshold`
- CLI reference: {doc}`../reference/cli/predict`
