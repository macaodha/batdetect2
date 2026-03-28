# How to configure target definitions

Use this guide to define which annotated sound events are considered valid
detection targets.

## 1) Start from a targets config file

```yaml
detection_target:
  name: bat
  match_if:
    name: has_tag
    tag:
      key: call_type
      value: Echolocation
  assign_tags:
    - key: call_type
      value: Echolocation
    - key: order
      value: Chiroptera
```

`match_if` decides whether an annotation is included in the detection target.

## 2) Use condition combinators when needed

You can combine conditions with `all_of`, `any_of`, and `not`.

```yaml
detection_target:
  name: bat
  match_if:
    name: all_of
    conditions:
      - name: has_tag
        tag:
          key: call_type
          value: Echolocation
      - name: not
        condition:
          name: has_any_tag
          tags:
            - key: call_type
              value: Social
            - key: class
              value: Not Bat
```

## 3) Verify with a small sample first

Before full training, inspect a small annotation subset and confirm that the
selection logic keeps the events you expect.

## Related pages

- Class mapping: {doc}`define-target-classes`
- ROI mapping: {doc}`configure-roi-mapping`
- Targets reference: {doc}`../reference/targets-config-workflow`
