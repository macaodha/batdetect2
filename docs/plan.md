# Documentation Plan

## Goal

Build documentation around the main user stories:

1. Run inference with the CLI on one folder of audio.
2. Use the Python API for inference with fine-grained control over outputs,
   including per-file workflows, class scores, features, and batch processing.
3. Train or fine-tune a custom model.
4. Evaluate a model and understand what the metrics mean.
5. Understand the concepts needed to use BatDetect2 correctly.

The docs should provide:

- a simple happy path in tutorials,
- richer task-oriented guidance in how-to guides,
- complete lookup material in reference,
- deep conceptual coverage in understanding.

Note: the current docs tree uses `explanation/`. For Diataxis consistency, this
plan uses `understanding/` as the target name for that conceptual section.

## Current State Review

### Looks reasonably complete

- `docs/source/index.md`: good top-level orientation and navigation.
- `docs/source/getting_started.md`: solid install and entry-point guidance.
- `docs/source/explanation/*.md`: the conceptual pages are currently the
  strongest part of the docs, especially pipeline overview, thresholds,
  preprocessing consistency, and targets.
- `docs/source/how_to/configure-*.md` and related target/data pages: practical
  support docs for preprocessing, targets, ROI mapping, and dataset formats are
  in decent shape.
- `docs/source/reference/cli/*.rst`: CLI reference wiring exists and should
  render useful option-level documentation from the Click commands.

### Partially complete

- `docs/source/how_to/run-batch-predictions.md`: useful, but thin.
- `docs/source/how_to/tune-detection-threshold.md`: useful, but too brief for
  a key workflow.
- `docs/source/reference/preprocessing-config.md`
- `docs/source/reference/postprocess-config.md`
- `docs/source/reference/targets-config-workflow.md`

These are good summaries, but they do not yet feel like complete references for
all the customization surfaces available in the code.

### Clearly incomplete or scaffolded

- `docs/source/tutorials/run-inference-on-folder.md`
- `docs/source/tutorials/integrate-with-a-python-pipeline.md`
- `docs/source/tutorials/train-a-custom-model.md`
- `docs/source/tutorials/evaluate-on-a-test-set.md`

All four main tutorials are still starter scaffolds. This is the biggest gap in
the current user story.

### Major mismatch to resolve

- `README.md` still tells an older story built around `batdetect2 detect` and
  `batdetect2.api`.
- The docs site tells the newer story built around `batdetect2 predict` and
  `batdetect2.api_v2`.

This creates avoidable confusion for users and should be treated as a priority
documentation alignment issue.

### Legacy documentation is not yet placed clearly

The repo still contains meaningful legacy documentation material, but it is not
yet presented as a clearly marked legacy path inside the docs.

Users need two things:

- a clear message that these docs exist for the previous BatDetect2 workflow,
- a clear recommendation that new users should prefer the newer CLI/API
  workflows and migrate where possible.

## Legacy Documentation Plan

### Goals

1. Preserve access to the old workflow documentation.
2. Prevent new users from accidentally following legacy guidance.
3. Give current users a clear migration path from legacy to current workflows.

### Proposed location

Add a dedicated legacy area inside the docs, for example:

- `docs/source/legacy/index.md`
- `docs/source/legacy/cli-detect.md`
- `docs/source/legacy/python-api.md`
- `docs/source/legacy/feature-extraction.md`
- `docs/source/legacy/migration-guide.md`

This keeps the material available without mixing it into the main happy-path
docs.

### User-facing messaging

Add clear notices in all relevant navigation entry points.

Suggested message pattern:

"If you want to use the previous version of BatDetect2, see the legacy
documentation. For new workflows, we recommend using the current `predict`
CLI and `BatDetect2API` interfaces."

Places that should link to the legacy docs:

- `docs/source/index.md`
- `docs/source/getting_started.md`
- `README.md`
- tutorial landing pages where users may be coming from older workflows
- any page that mentions the old `detect` command or old Python API

### Migration guide plan

Add a dedicated migration guide that explains:

1. who should migrate now and who may need to stay on the legacy workflow,
2. the mapping from old CLI commands to new CLI commands,
3. the mapping from old Python API calls to new `api_v2` / `BatDetect2API`
   patterns,
4. what changed in outputs, terminology, and configuration,
5. how legacy feature extraction concepts map to the new API surfaces,
6. what behavior differences users should validate before switching,
7. a short migration checklist.

High-priority migration mappings to document:

- `batdetect2 detect` -> `batdetect2 predict directory`
- old `batdetect2.api` file processing -> `BatDetect2API.from_checkpoint(... )`
  plus `process_file`, `process_files`, `process_audio`, or
  `process_spectrogram`
- legacy `cnn_feats`, `spec_features`, and `spec_slices` -> current output and
  feature access patterns, with explicit notes where there is no direct
  one-to-one replacement

### Legacy content handling plan

For each legacy page or legacy concept:

1. Decide whether it should be preserved as-is, rewritten as a legacy page, or
   replaced by the migration guide.
2. Add a prominent warning banner saying it describes the previous workflow.
3. Link forward to the current equivalent page when one exists.

### Definition of done for legacy handling

Legacy documentation work is done when:

1. a reader can clearly distinguish legacy from current docs,
2. old users can still find the previous workflow documentation,
3. new users are consistently directed to the new docs,
4. there is a practical migration guide covering the main CLI and Python API
   transitions.

## Main Gaps By User Story

### 1. CLI inference

Current coverage exists, but the happy path is not truly documented yet.

Missing:

- a full worked tutorial from input audio to saved outputs,
- clear guidance on what outputs are written and how to inspect them,
- stronger documentation for `predict dataset`,
- a clearer story for default model vs custom checkpoint,
- practical guidance for selecting output formats and thresholds.

### 2. Python API inference

This is currently the weakest major story.

The code exposes much more than the docs explain, including:

- `BatDetect2API.from_checkpoint` and `from_config`,
- `process_file`, `process_files`, `process_directory`, `process_clips`,
- `process_audio`, `process_spectrogram`,
- `get_top_class_name`, `get_class_scores`, `get_detection_features`,
- `save_predictions` and `load_predictions`.

Missing docs:

- an API-first tutorial with a simple path,
- a how-to for file-by-file inspection and custom post-processing,
- a how-to for batch API inference,
- a reference page for `BatDetect2API`,
- an explanation of what the feature vectors are and how users should think
  about them.

Important terminology note:

- the old API/docs talk about `cnn_feats`, `spec_features`, and `spec_slices`,
- the new API exposes per-detection `features`,
- users interested in embeddings / downstream exploration will need a clear,
  explicit doc that connects these ideas.

### 3. Batch inference

Batch prediction exists in both CLI and API workflows, but the docs do not yet
explain the design space well.

Missing:

- when to use `directory` vs `file_list` vs `dataset`,
- how clipping works during inference,
- what `InferenceConfig` controls,
- how batch size, workers, and output format choices affect runs,
- how to organize large runs reproducibly.

### 4. Training a custom model

Supporting pages exist, but the end-to-end story is not yet there.

Missing:

- one complete tutorial from dataset config to checkpoints and sanity check,
- a "minimum viable training setup" page,
- clearer explanation of how model, targets, audio, training, inference,
  outputs, and logging configs fit together,
- a fine-tuning story versus training from scratch.

### 5. Evaluation

Evaluation is significantly under-documented relative to the code.

Missing:

- what evaluation tasks exist,
- what metrics and plots are produced,
- how predictions are matched to annotations,
- how to interpret failures and trade-offs,
- how to configure evaluation for different research questions.

### 6. Understanding / concepts

This is the best-developed section today, but it still needs expansion.

Concepts that should be covered more fully:

- what the model predicts,
- what the raw and formatted outputs represent,
- how to interpret detection scores and class scores,
- what targets are and how they shape training and decoding,
- how preprocessing choices affect model behavior,
- what the extracted features represent and when they are useful,
- what evaluation metrics actually measure,
- why local validation is required before ecological inference.

## Proposed Documentation Architecture

## Target Table of Contents

### Home

- Home
- Getting started
- FAQ
- Legacy docs

### Tutorials

These should be the default path for most users.

- Tutorial: Run inference on a folder of audio
- Tutorial: Explore predictions in Python for one file
- Tutorial: Train a custom model
- Tutorial: Evaluate a trained model

### How-to Guides

These cover practical tasks once the user is past the happy path.

- How to choose an inference input mode
- How to run batch predictions from a directory
- How to run batch predictions from a file list
- How to run predictions from a dataset config
- How to tune detection thresholds
- How to inspect class scores in Python
- How to inspect detection features in Python
- How to save predictions in different output formats
- How to configure inference clipping
- How to configure audio preprocessing
- How to configure spectrogram preprocessing
- How to configure target definitions
- How to define target classes
- How to configure ROI mapping
- How to configure an AOEF dataset
- How to import legacy BatDetect2 annotations
- How to fine-tune from a checkpoint
- How to choose and configure evaluation tasks
- How to interpret evaluation outputs

### Reference

This should be the complete lookup layer.

- CLI reference
- CLI reference: base command and global options
- CLI reference: predict
- CLI reference: data
- CLI reference: train
- CLI reference: evaluate
- CLI reference: legacy detect
- API reference: `BatDetect2API`
- Config reference: top-level app config
- Config reference: inference config
- Config reference: evaluation config
- Config reference: outputs config
- Config reference: output formats
- Config reference: output transforms
- Config reference: preprocessing config
- Config reference: postprocess config
- Config reference: targets config workflow
- Reference: data sources
- Reference: targets module

### Understanding

This is the conceptual layer and should carry the deeper Diataxis
"understanding" material.

- What BatDetect2 predicts
- How the pipeline fits together
- How to interpret detection scores and class scores
- How to interpret formatted outputs
- What extracted features / embeddings are and are not
- Postprocessing and thresholds
- Preprocessing consistency and domain shift
- Target encoding and decoding
- Evaluation concepts and matching behavior
- Model output, validation, and ecological interpretation

### Legacy

This is a clearly signposted area for the previous workflow only.

- Legacy overview
- Legacy CLI workflow with `batdetect2 detect`
- Legacy Python API with `batdetect2.api`
- Legacy feature extraction outputs
- Migration guide: legacy to current workflows

### Tutorials

Keep tutorials opinionated and minimal. Each one should show the default happy
path with the fewest possible choices.

Planned tutorial set:

1. Run inference on a folder of audio.
2. Explore predictions in Python for one file.
3. Train a custom model.
4. Evaluate a trained model.

### How-to Guides

Use how-to guides for branching tasks and customization.

Planned additions or expansions:

- Choose an inference input mode: directory, file list, or dataset.
- Run large batch inference reproducibly.
- Save predictions in different output formats.
- Inspect class scores and features in Python.
- Explore detection features / embeddings downstream.
- Tune clipping and inference settings.
- Fine-tune from a checkpoint.
- Choose and configure evaluation tasks.
- Interpret evaluation artifacts.

### Reference

Reference should become the complete map of all configurable surfaces.

High-priority additions:

- `BatDetect2API` reference.
- `InferenceConfig` reference.
- `EvaluationConfig` reference.
- `OutputsConfig` and output format reference.
- Output transform reference.
- clearer config composition reference for the full app config.

### Understanding

This is where the deeper conceptual material should live.

High-priority pages:

1. What BatDetect2 predicts.
2. How to interpret outputs, scores, and uncertainty.
3. What extracted features / embeddings are and are not.
4. Targets, labels, and decoded outputs.
5. Preprocessing consistency and domain shift.
6. Postprocessing, thresholds, and output density.
7. How evaluation works and what the metrics mean.
8. Why local validation is required before ecological interpretation.

## Priority Order

### Phase 1: Fix the primary user journey

1. Expand the four scaffold tutorials into real end-to-end guides.
2. Add a proper Python/API inference story.
3. Document outputs and how to inspect them.
4. Align `README.md` with the newer CLI/API documentation story.
5. Create the legacy docs section and add clear signposting to it.

### Phase 2: Cover the customization surface

1. Add how-to guides for batch inference, output formats, and API inspection.
2. Add reference pages for inference, outputs, evaluation, and API surfaces.
3. Add fine-tuning and advanced training guidance.
4. Write the migration guide from legacy to current workflows.

### Phase 3: Deepen understanding

1. Expand the conceptual section into a true understanding section.
2. Add pages for output interpretation, features/embeddings, and evaluation
   concepts.
3. Reader-test the docs against realistic user questions.

## Immediate Next Steps

1. Decide whether to rename `explanation/` to `understanding/` or keep the
   current directory name and just treat it as the Diataxis understanding
   section.
2. Draft the target table of contents for Tutorials, How-to, Reference, and
   Understanding.
3. Draft the legacy docs section and migration-guide table of contents.
4. Rewrite the four scaffold tutorials first.
5. Add the missing API, outputs, evaluation, and migration documentation
   immediately after.
