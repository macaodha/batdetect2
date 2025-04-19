# Using Legacy BatDetect2 Annotation Formats

## Introduction

If you have annotation data created using older BatDetect2 annotation tools, BatDetect2 provides tools to load these datasets.
These older formats typically use JSON files to store annotation information, including bounding boxes and labels for sound events within recordings.

There are two main variations of this legacy format that BatDetect2 can load:

1.  **Directory-Based (`format: "batdetect2"`):** Annotations for each audio recording are stored in a _separate_ JSON file within a dedicated directory.
    There's a naming convention linking the JSON file to its corresponding audio file (e.g., `my_recording.wav` annotations are stored in `my_recording.wav.json`).
2.  **Single Merged File (`format: "batdetect2_file"`):** Annotations for _multiple_ recordings are aggregated into a _single_ JSON file.
    This file contains a list, where each item represents the annotations for one recording, following the same internal structure as the directory-based format.

When you configure BatDetect2 to use these formats, it will read the legacy data and convert it internally into the standard `soundevent` data structures used by the rest of the pipeline.

## Configuration

You specify which legacy format to use within the `sources` list of your main `DatasetConfig` (usually in your primary YAML configuration file).

### Format 1: Directory-Based

Use this when you have a folder containing many individual JSON annotation files, one for each audio file.

**Configuration Fields:**

- `format: "batdetect2"`: **(Required)** Identifies this specific legacy format loader.
- `name: your_source_name`: **(Required)** A unique name for this data source.
- `audio_dir: path/to/audio/files`: **(Required)** Path to the directory containing the `.wav` audio files.
- `annotations_dir: path/to/annotation/jsons`: **(Required)** Path to the directory containing the individual `.json` annotation files.
- `description: "Details..."`: (Optional) Description of this source.
- `filter: ...`: (Optional) Settings to filter which JSON files are processed based on flags within them (see "Filtering Legacy Annotations" below).

**YAML Example:**

```yaml
# In your main DatasetConfig YAML file
sources:
  - name: "OldProject_SiteA_Files"
    format: "batdetect2" # Use the directory-based loader
    audio_dir: "/data/SiteA/Audio/"
    annotations_dir: "/data/SiteA/Annotations_JSON/"
    description: "Legacy annotations stored as individual JSONs per recording."
    # filter: ... # Optional filter settings can be added here
```

### Format 2: Single Merged File

Use this when you have a single JSON file that contains a list of annotations for multiple recordings.

**Configuration Fields:**

- `format: "batdetect2_file"`: **(Required)** Identifies this specific legacy format loader.
- `name: your_source_name`: **(Required)** A unique name for this data source.
- `audio_dir: path/to/audio/files`: **(Required)** Path to the directory containing the `.wav` audio files referenced _within_ the merged JSON file.
- `annotations_path: path/to/your/merged_annotations.json`: **(Required)** Path to the single `.json` file containing the list of annotations.
- `description: "Details..."`: (Optional) Description of this source.
- `filter: ...`: (Optional) Settings to filter which records _within_ the merged file are processed (see "Filtering Legacy Annotations" below).

**YAML Example:**

```yaml
# In your main DatasetConfig YAML file
sources:
  - name: "OldProject_Merged"
    format: "batdetect2_file" # Use the merged file loader
    audio_dir: "/data/AllAudio/"
    annotations_path: "/data/CombinedAnnotations/old_project_merged.json"
    description: "Legacy annotations aggregated into a single JSON file."
    # filter: ... # Optional filter settings can be added here
```

## Filtering Legacy Annotations

The legacy JSON annotation structure (for both formats) included boolean flags indicating the status of the annotation work for each recording:

- `annotated`: Typically `true` if a human had reviewed or created annotations for the file.
- `issues`: Typically `true` if problems were noted during annotation or review.

You can optionally filter the data based on these flags using a `filter:` block within the source configuration.
This applies whether you use `"batdetect2"` or `"batdetect2_file"`.

**Filter Options:**

- `only_annotated: <true_or_false>`:
  - `true` (**Default**): Only process entries where the `annotated` flag in the JSON is `true`.
  - `false`: Process entries regardless of the `annotated` flag.
- `exclude_issues: <true_or_false>`:
  - `true` (**Default**): Skip processing entries where the `issues` flag in the JSON is `true`.
  - `false`: Process entries even if they are flagged with `issues`.

**Default Filtering:** If you **omit** the `filter:` block entirely, the default settings (`only_annotated: true`, `exclude_issues: true`) are applied automatically.
This means only entries marked as annotated and not having issues will be loaded.

**Disabling Filtering:** To load _all_ entries from the legacy source regardless of the `annotated` or `issues` flags, explicitly disable the filter:

```yaml
filter: null
```

**YAML Example (Custom Filter):** Only load entries marked as annotated, but _include_ those with issues.

```yaml
sources:
  - name: "LegacyData_WithIssues"
    format: "batdetect2" # Or "batdetect2_file"
    audio_dir: "path/to/audio"
    annotations_dir: "path/to/annotations" # Or annotations_path for merged
    filter:
      only_annotated: true
      exclude_issues: false # Include entries even if issues flag is true
```

## Summary

BatDetect2 allows you to incorporate datasets stored in older "BatDetect2" JSON formats.

- Use `format: "batdetect2"` and provide `annotations_dir` if you have one JSON file per recording in a directory.
- Use `format: "batdetect2_file"` and provide `annotations_path` if you have a single JSON file containing annotations for multiple recordings.
- Optionally use the `filter:` block with `only_annotated` and `exclude_issues` to select data based on flags present in the legacy JSON structure.

The system will handle loading, filtering (if configured), and converting this legacy data into the standard `soundevent` format used internally.
