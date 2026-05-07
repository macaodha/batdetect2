# What BatDetect2 predicts

BatDetect2 predicts call-level events, not recording-level truth.

For each retained detection, the current stack can expose:

- a geometry describing where the event sits in time-frequency space,
- a detection score,
- a class-score vector,
- an internal feature vector.

## Detection score versus class scores

These are different outputs and should not be interpreted as the same thing.

- The detection score is about whether the event is kept as a detection.
- The class-score vector ranks classes for that detected event.

A detection can be kept while still having uncertain class identity.

## Predictions are conditional on the workflow

The final output also depends on:

- preprocessing,
- postprocessing,
- thresholds,
- target definitions,
- output transforms.

That is why two runs can differ even when they use the same checkpoint.

## What BatDetect2 does not predict

BatDetect2 does not directly output ecological truth.

It also does not eliminate the need for local validation.

Use reviewed local data before making ecological claims.

## Related pages

- Model output and validation: {doc}`model-output-and-validation`
- Postprocessing and thresholds: {doc}`postprocessing-and-thresholds`
- Interpreting formatted outputs: {doc}`interpreting-formatted-outputs`
