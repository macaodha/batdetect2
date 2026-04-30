# Outputs config reference

`OutputsConfig` controls two layers of prediction handling:

- how detections are transformed before formatting,
- how formatted outputs are written to disk.

Defined in `batdetect2.outputs.config`.

## Fields

- `format`
  - output format config.
- `transform`
  - output transform config.

## Mental model

The output workflow is:

1. model outputs are decoded into detections,
2. optional output transforms filter or adjust those detections,
3. a formatter serializes them to disk.

## Default behavior

By default, the current stack uses the raw output formatter unless you override it.

## Related pages

- Output formats: {doc}`output-formats`
- Output transforms: {doc}`output-transforms`
- Save predictions in different output formats: {doc}`../how_to/save-predictions-in-different-output-formats`
