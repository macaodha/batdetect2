# How to import legacy batdetect2 annotations

Use this guide if your annotations are in older batdetect2 JSON formats.

Two legacy formats are supported:

- `batdetect2`:
  one annotation JSON file per recording
- `batdetect2_file`:
  one merged JSON file for many recordings

## 1) Choose the correct source format

Directory-based annotations (`format:
batdetect2`):

```yaml
sources:
  - name: legacy_per_file
    format: batdetect2
    audio_dir: /path/to/audio
    annotations_dir: /path/to/annotation_json_dir
```

Merged annotation file (`format:
batdetect2_file`):

```yaml
sources:
  - name: legacy_merged
    format: batdetect2_file
    audio_dir: /path/to/audio
    annotations_path: /path/to/merged_annotations.json
```

## 2) Set optional legacy filters

Legacy filters are based on `annotated` and `issues` flags.

```yaml
filter:
  only_annotated: true
  exclude_issues: true
```

To load all entries regardless of flags:

```yaml
filter: null
```

## 3) Validate and convert if needed

Check loaded records:

```bash
batdetect2 data summary path/to/dataset.yaml
```

Convert to annotation-set output for downstream tooling:

```bash
batdetect2 data convert path/to/dataset.yaml --output path/to/output.json
```

## 4) Continue with current workflows

- Run predictions:
  {doc}`../inference/run-batch-predictions`
- Train on imported data:
  {doc}`../../tutorials/train-a-custom-model`
- Field-level reference:
  {doc}`../../reference/configs/data/data-sources`
