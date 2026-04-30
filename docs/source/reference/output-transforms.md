# Output transforms reference

Output transforms operate after decoding and before formatting.

Defined in `batdetect2.outputs.transforms`.

## Top-level config

`OutputTransformConfig` contains:

- `detection_transforms`
- `clip_transforms`

## Detection transforms

Detection transforms operate on one detection at a time.

Built-in examples include:

- filtering by frequency,
- filtering by duration.

These can remove detections entirely if they fail the transform.

## Clip transforms

Clip transforms operate on the list of detections for one clip.

Built-in examples include:

- removing detections above Nyquist,
- removing detections at clip edges.

## Related pages

- Outputs config: {doc}`outputs-config`
- Understanding outputs: {doc}`../explanation/interpreting-formatted-outputs`
