# Interpreting formatted outputs

BatDetect2 can write predictions in several output formats.

Those formats are different views of the same underlying detections, not different model behaviors.

## Separate the underlying detection from the serialized file

Internally, the current stack works with clip-level detections containing geometry, detection score, class scores, and features.

Output formatters then serialize those detections in different ways.

## Raw outputs are richest

The `raw` format preserves the broadest structured view of detections and is a good default when you want to inspect or reload predictions later.

## Tabular outputs are for analysis convenience

The `parquet` format is convenient for data analysis workflows, but the tabular representation is only one projection of the underlying detection object.

## Legacy-shaped outputs are mainly for compatibility

The `batdetect2` formatter writes the older BatDetect2-style JSON shape.

Use it when you need compatibility with older downstream tools or workflows.

## The meaning does not come from the file extension

Do not assume that a `.json`, `.parquet`, or `.nc` file changes what the model predicted.

It changes how the prediction is packaged and how much detail is retained.

## Related pages

- Output formats reference: {doc}`../reference/output-formats`
- Outputs config reference: {doc}`../reference/outputs-config`
