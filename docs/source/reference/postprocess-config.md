# Postprocess config reference

`PostprocessConfig` controls how raw detector outputs are converted into final
detections.

Defined in `batdetect2.postprocess.config`.

## Fields

- `nms_kernel_size` (int > 0)
  - neighborhood size for non-maximum suppression.
- `detection_threshold` (float >= 0)
  - minimum detection score to keep a candidate event.
- `classification_threshold` (float >= 0)
  - minimum class score used when assigning class tags.
- `top_k_per_sec` (int > 0)
  - maximum detection density per second.

## Defaults

- `detection_threshold`: `0.01`
- `classification_threshold`: `0.1`
- `top_k_per_sec`: `100`

`nms_kernel_size` defaults to the library constant used by the NMS module.

## Related pages

- Threshold behaviour: {doc}`../explanation/postprocessing-and-thresholds`
- Threshold tuning workflow: {doc}`../how_to/tune-detection-threshold`
- CLI predict options: {doc}`cli/predict`
