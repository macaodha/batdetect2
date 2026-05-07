# Inference config reference

`InferenceConfig` controls how files are clipped and batched during prediction-time workflows.

Defined in `batdetect2.inference.config`.

## Top-level fields

- `loader`
  - data-loader settings for inference.
- `clipping`
  - controls how recordings are split into clips before batching.

## `loader`

Current built-in loader field:

- `batch_size` (int, default `8`)

## `clipping`

Fields:

- `enabled` (bool)
- `duration` (float, seconds)
- `overlap` (float, seconds)
- `max_empty` (float)
- `discard_empty` (bool)

## When to override this config

Override `InferenceConfig` when:

- long recordings need different clipping behavior,
- you want to tune batch size for your hardware,
- you need reproducible prediction settings across runs.

## Related pages

- Tune inference clipping: {doc}`../how_to/tune-inference-clipping`
- Predict CLI reference: {doc}`cli/predict`
