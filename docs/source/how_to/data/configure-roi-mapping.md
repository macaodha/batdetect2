# How to configure ROI mapping

Use this guide to control how annotation geometry is encoded into training
targets and decoded back into boxes.

## 1) Set the default ROI mapper

The default mapper is `anchor_bbox`.

```yaml
roi:
  default:
    name: anchor_bbox
    anchor: bottom-left
    time_scale: 1000.0
    frequency_scale: 0.001163
```

## 2) Choose an anchor strategy

Typical options include `bottom-left` and `center`.

- `bottom-left` is the current default.
- `center` can be easier to reason about in some workflows.

## 3) Set scale factors intentionally

- `time_scale` controls width scaling.
- `frequency_scale` controls height scaling.

Use values that are consistent with your model setup and keep them fixed when
comparing experiments.

## 4) (Optional) override ROI mapping for specific classes

Add class-specific mappers under `roi.overrides`.

```yaml
roi:
  default:
    name: anchor_bbox
    anchor: bottom-left
    time_scale: 1000.0
    frequency_scale: 0.001163
  overrides:
    species_x:
      name: anchor_bbox
      anchor: center
      time_scale: 1000.0
      frequency_scale: 0.001163
```

## Related pages

- Target definitions:
  {doc}`configure-target-definitions`
- Class definitions:
  {doc}`define-target-classes`
- Target encoding overview:
  {doc}`../../explanation/target-encoding-and-decoding`
