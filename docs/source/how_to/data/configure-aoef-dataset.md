# How to configure an AOEF dataset source

Use this guide when your annotations are stored in AOEF/soundevent JSON files,
including exports from Whombat.

## 1) Add an AOEF source entry

In your dataset config, add a source with `format:
aoef`.

```yaml
sources:
  - name: my_aoef_source
    format: aoef
    audio_dir: /path/to/audio
    annotations_path: /path/to/annotations.soundevent.json
```

## 2) Choose filtering behavior for annotation projects

If `annotations_path` is an `AnnotationProject`, you can filter by task state.

```yaml
sources:
  - name: whombat_verified
    format: aoef
    audio_dir: /path/to/audio
    annotations_path: /path/to/project_export.aoef
    filter:
      only_completed: true
      only_verified: true
      exclude_issues: true
```

If you omit `filter`, default project filtering is applied.

To disable filtering for project files:

```yaml
filter: null
```

## 3) Check that the source loads

Run a summary on your dataset config:

```bash
batdetect2 data summary path/to/dataset.yaml
```

## 4) Continue to training or evaluation

- For training:
  {doc}`../../tutorials/train-a-custom-model`
- For field-level reference:
  {doc}`../../reference/configs/data/data-sources`
