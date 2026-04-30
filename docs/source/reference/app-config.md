# Top-level app config reference

The top-level config object is `BatDetect2Config`.

Defined in `batdetect2.config`.

It combines the main configuration surfaces used across training, inference, evaluation, outputs, and logging.

## Fields

- `config_version`
- `train`
  - training-specific config.
- `evaluation`
  - evaluation task and plot config.
- `model`
  - model architecture, preprocessing, postprocessing, and targets.
- `audio`
  - audio loading and resampling config.
- `inference`
  - clipping and loader config for prediction-time workflows.
- `outputs`
  - output format and output transform config.
- `logging`
  - logging backend and formatting config.

## Mental model

Think of `BatDetect2Config` as the complete application wiring for the current stack.

Use it when you want one reproducible config that describes the whole workflow.

## Related pages

- Inference config: {doc}`inference-config`
- Evaluation config: {doc}`evaluation-config`
- Outputs config: {doc}`outputs-config`
- General config reference: {doc}`configs`
