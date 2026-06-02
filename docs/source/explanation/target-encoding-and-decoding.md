# Target encoding and decoding

batdetect2 turns annotated sound events into training targets, then maps model
outputs back into interpretable predictions.

## Encoding path (annotations -> model targets)

At training time, the target system:

1. checks whether an event belongs to the configured detection target,
2. assigns a classification label (or none for non-specific class matches),
3. maps event geometry into position and size targets.

This behaviour is configured through `TargetConfig`, `TargetClassConfig`, and
ROI mapper settings.

## Decoding path (model outputs -> tags and geometry)

At inference time, class labels and ROI parameters are decoded back into
annotation tags and geometry.

This makes outputs interpretable in the same conceptual space as your original
annotations.

## Why this matters

Target definitions are not just metadata.
They directly shape:

- what events are treated as positive examples,
- which class names the model learns,
- how geometry is represented and reconstructed.

Small changes here can alter both training outcomes and prediction semantics.

## Related pages

- Configure detection target logic:
  {doc}`../how_to/data/configure-target-definitions`
- Configure class mapping:
  {doc}`../how_to/data/define-target-classes`
- Configure ROI mapping:
  {doc}`../how_to/data/configure-roi-mapping`
- Target config reference:
  {doc}`../reference/configs/data/targets-config-workflow`
