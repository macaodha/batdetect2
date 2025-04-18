# Using AOEF / Soundevent Data Sources (`format: "aoef"`)

## Introduction

The **AOEF (Acoustic Open Event Format)**, stored as `.json` files, is the annotation format used by the underlying `soundevent` library and is compatible with annotation tools like **Whombat**.
BatDetect2 can directly load annotation data stored in this format.

This format can represent two main types of annotation collections:

1.  `AnnotationSet`: A straightforward collection of annotations for various audio clips.
2.  `AnnotationProject`: A more structured format often exported by annotation tools (like Whombat).
    It includes not only the annotations but also information about annotation _tasks_ (work assigned to annotators) and their status (e.g., in-progress, completed, verified, rejected).

This section explains how to configure a data source in your `DatasetConfig` to load data from either type of AOEF file.

## Configuration

To define a data source using the AOEF format, you add an entry to the `sources` list in your main `DatasetConfig` (usually within your primary YAML configuration file) and set the `format` field to `"aoef"`.

Here are the key fields you need to specify for an AOEF source:

- `format: "aoef"`: **(Required)** Tells BatDetect2 to use the AOEF loader for this source.
- `name: your_source_name`: **(Required)** A unique name you choose for this data source (e.g., `"whombat_project_export"`, `"final_annotations"`).
- `audio_dir: path/to/audio/files`: **(Required)** The path to the directory where the actual audio `.wav` files referenced in the annotations are located.
- `annotations_path: path/to/your/annotations.aoef`: **(Required)** The path to the single `.aoef` or `.json` file containing the annotation data (either an `AnnotationSet` or an `AnnotationProject`).
- `description: "Details about this source..."`: (Optional) A brief description of the data source.
- `filter: ...`: **(Optional)** Specific settings used _only if_ the `annotations_path` file contains an `AnnotationProject`.
  See details below.

## Filtering Annotation Projects (Optional)

When working with annotation projects, especially collaborative ones or those still in progress (like exports from Whombat), you often want to train only on annotations that are considered complete and reliable.
The optional `filter:` section allows you to specify criteria based on the status of the annotation _tasks_ within the project.

**If `annotations_path` points to a simple `AnnotationSet` file, the `filter:` section is ignored.**

If `annotations_path` points to an `AnnotationProject`, you can add a `filter:` block with the following options:

- `only_completed: <true_or_false>`:
  - `true` (Default): Only include annotations from tasks that have been marked as "completed".
  - `false`: Include annotations regardless of task completion status.
- `only_verified: <true_or_false>`:
  - `false` (Default): Verification status is not considered.
  - `true`: Only include annotations from tasks that have _also_ been marked as "verified" (typically meaning they passed a review step).
- `exclude_issues: <true_or_false>`:
  - `true` (Default): Exclude annotations from any task that has been marked as "rejected" or flagged with issues.
  - `false`: Include annotations even if their task was marked as having issues (use with caution).

**Default Filtering:** If you include the `filter:` block but omit some options, or if you _omit the entire `filter:` block_, the default settings are applied to `AnnotationProject` files: `only_completed: true`, `only_verified: false`, `exclude_issues: true`.
This common default selects annotations from completed tasks that haven't been rejected, without requiring separate verification.

**Disabling Filtering:** If you want to load _all_ annotations from an `AnnotationProject` regardless of task status, you can explicitly disable filtering by setting `filter: null` in your YAML configuration.

## YAML Configuration Examples

**Example 1: Loading a standard AnnotationSet (or a Project with default filtering)**

```yaml
# In your main DatasetConfig YAML file

sources:
  - name: "MyFinishedAnnotations"
    format: "aoef" # Specifies the loader
    audio_dir: "/path/to/my/audio/"
    annotations_path: "/path/to/my/dataset.soundevent.json" # Path to the AOEF file
    description: "Finalized annotations set."
    # No 'filter:' block means default filtering applied IF it's an AnnotationProject,
    # or no filtering applied if it's an AnnotationSet.
```

**Example 2: Loading an AnnotationProject, requiring verification**

```yaml
# In your main DatasetConfig YAML file

sources:
  - name: "WhombatVerifiedExport"
    format: "aoef"
    audio_dir: "relative/path/to/audio/" # Relative to where BatDetect2 runs or a base_dir
    annotations_path: "exports/whombat_project.aoef" # Path to the project file
    description: "Annotations from Whombat project, only using verified tasks."
    filter: # Customize the filter
      only_completed: true # Still require completion
      only_verified: true # *Also* require verification
      exclude_issues: true # Still exclude rejected tasks
```

**Example 3: Loading an AnnotationProject, disabling all filtering**

```yaml
# In your main DatasetConfig YAML file

sources:
  - name: "WhombatRawExport"
    format: "aoef"
    audio_dir: "data/audio_pool/"
    annotations_path: "exports/whombat_project_all.aoef"
    description: "All annotations from Whombat, regardless of task status."
    filter: null # Explicitly disable task filtering
```

## Summary

To load standard `soundevent` annotations (including Whombat exports), set `format: "aoef"` for your data source in the `DatasetConfig`.
Provide the `audio_dir` and the path to the single `annotations_path` file.
If dealing with `AnnotationProject` files, you can optionally use the `filter:` block to select annotations based on task completion, verification, or issue status.
