# `BatDetect2API` reference

`BatDetect2API` is the main Python entry point for BatDetect2.

Use it when you want to load a model, run prediction, inspect detections,
evaluate results, or train from Python.

Defined in `batdetect2.api_v2`.

## Main ways to create it

- `BatDetect2API.from_checkpoint(path, ...)`
  - load a trained checkpoint, a bundled checkpoint alias, or a Hugging Face
    checkpoint.
- `BatDetect2API.from_config(model_config=..., targets_config=..., ...)`
  - build a full model stack from config objects.

## Common tasks

- Load a checkpoint and run prediction on one file.
- Run prediction on many files or clips.
- Save predictions in one of the supported output formats.
- Evaluate a model on labelled data.
- Fine-tune an existing checkpoint on new targets.

## Generated reference

```{eval-rst}
.. autoclass:: batdetect2.api_v2.BatDetect2API
```

## Related pages

- Python tutorial:
  {doc}`../tutorials/integrate-with-a-python-pipeline`
- Outputs config reference:
  {doc}`outputs-config`
- Output formats reference:
  {doc}`output-formats`
