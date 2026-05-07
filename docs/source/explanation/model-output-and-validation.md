# Model output and validation

BatDetect2 outputs model predictions, not ground truth. The same configuration
can behave differently across recording conditions, species compositions, and
acoustic environments.

## Why threshold choice matters

- Lower detection thresholds increase sensitivity but can increase false
  positives.
- Higher thresholds reduce false positives but can miss faint calls.

No threshold is universally correct. The right setting depends on your survey
objectives and tolerance for false positives versus missed detections.

## Why local validation is required

Model performance depends on how similar your data are to training data.
Before ecological interpretation, validate predictions on a representative,
locally reviewed subset.

Recommended validation checks:

1. Compare detection counts against expert-reviewed clips.
2. Inspect species-level predictions for plausible confusion patterns.
3. Repeat checks across sites, seasons, and recorder setups.

For practical threshold workflows, see
{doc}`../how_to/tune-detection-threshold`.
