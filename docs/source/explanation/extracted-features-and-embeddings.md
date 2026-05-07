# Extracted features and embeddings

The current API exposes a per-detection `features` vector.

Older BatDetect2 workflows also exposed concepts such as `cnn_feats`,
`spec_features`, and `spec_slices`.

## What the current feature vector is

In the current stack, each retained detection can carry an internal feature
representation produced by the model output pipeline.

This is useful for downstream exploration, comparison, and custom analysis.

## What these features are not

They are not automatically human-interpretable ecological variables.

They are also not a substitute for careful validation.

## Why people refer to them as embeddings

In practice, users often treat these feature vectors as embeddings because they
can be used as dense learned representations of detections.

That usage is reasonable, but you should still treat them as model-derived
internal representations whose meaning depends on the training setup.

## Legacy terminology versus current terminology

- legacy `cnn_feats` referred to CNN feature outputs in the older workflow,
- legacy `spec_features` referred to lower-level extracted call features,
- current `features` are the per-detection vectors attached to `Detection`
  objects.

These are related ideas, but not necessarily one-to-one replacements.

## Related pages

- Inspect detection features in Python:
  {doc}`../how_to/inspect-detection-features-in-python`
- Legacy migration guide:
  {doc}`../legacy/migration-guide`
