# How to define target classes

Use this guide to map annotations to classification labels used during training.

## 1) Add classification target entries

Each entry defines a class name and matching tags.

```yaml
classification_targets:
  - name: pippip
    tags:
      - key: class
        value: Pipistrellus pipistrellus
  - name: pippyg
    tags:
      - key: class
        value: Pipistrellus pygmaeus
```

## 2) Use `assign_tags` to control decoded output tags

If you want prediction output tags to differ from matching tags, set
`assign_tags` explicitly.

```yaml
classification_targets:
  - name: pipistrelle_group
    tags:
      - key: class
        value: Pipistrellus pipistrellus
    assign_tags:
      - key: genus
        value: Pipistrellus
```

## 3) Use `match_if` for complex class rules

For advanced conditions, use `match_if` instead of `tags`.

```yaml
classification_targets:
  - name: long_call
    match_if:
      name: duration
      operator: gt
      seconds: 0.02
```

## 4) Confirm class names are unique

`classification_targets.name` values must be unique.

## Related pages

- Detection-target filtering:
  {doc}`configure-target-definitions`
- ROI mapping:
  {doc}`configure-roi-mapping`
- Targets config reference:
  {doc}`../../reference/configs/data/targets-config-workflow`
