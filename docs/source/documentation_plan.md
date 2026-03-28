---
orphan: true
---

# Documentation Architecture and Migration Plan (Phase 0)

This page defines the Phase 0 documentation architecture and inventory for
reorganizing `batdetect2` documentation using the Diataxis framework.

## Scope and goals

Phase 0 focuses on architecture and prioritization only. It does not attempt
to write all new docs yet.

Primary goals:

1. Define a target docs architecture by Diataxis type.
2. Map current pages to target documentation types.
3. Identify what to keep, split, rewrite, or deprecate.
4. Set priorities for implementation phases.

## Audiences

Two primary audiences are in scope.

1. Ecologists who prefer minimal coding, focused on practical workflows:
   run inference, inspect outputs, and possibly train with custom data.
2. Ecologists or bioacousticians who are Python-savvy and want to customize
   workflows, training, and analysis.

## Target information architecture

The target architecture uses four top-level documentation sections.

1. Tutorials
   - Learning-oriented, single-path, reproducible walkthroughs.
2. How-to guides
   - Task-oriented procedures for common real goals.
3. Reference
   - Factual descriptions of CLI, configs, APIs, and formats.
4. Explanation
   - Conceptual material that explains why design and workflow decisions
     matter.

Cross-cutting navigation conventions:

- Every page starts with audience, prerequisites, and outcome.
- Every page serves one Diataxis type only.
- Beginner-first path is prioritized, with clear links to advanced pages.

## Phase 0 inventory: current docs mapped to Diataxis

Legend:

- Keep: useful as-is with minor edits.
- Split: contains mixed documentation types and should be separated.
- Rewrite: major changes needed to fit target audience/type.
- Move: content is valid but belongs under another section.

| Current page | Current role | Target type | Audience | Action | Priority |
| --- | --- | --- | --- | --- | --- |
| `README.md` | Mixed quickstart + CLI + API + warning | Tutorial + How-to + Explanation (split) | 1 + 2 | Split | P0 |
| `docs/source/index.md` | Sparse landing page | Navigation hub | 1 + 2 | Rewrite | P0 |
| `docs/source/architecture.md` | Internal architecture deep dive | Explanation + developer reference | 2 | Move/trim | P2 |
| `docs/source/postprocessing.md` | Concept + config + internals + usage | Explanation + How-to + Reference (split) | 1 + 2 | Split | P1 |
| `docs/source/preprocessing/index.md` | Conceptual overview with some procedural flow | Explanation | 2 (and 1 optional) | Keep/trim | P2 |
| `docs/source/preprocessing/audio.md` | Detailed configuration and behavior | Reference + How-to fragments | 2 | Split | P2 |
| `docs/source/preprocessing/spectrogram.md` | Detailed configuration and behavior | Reference + How-to fragments | 2 | Split | P2 |
| `docs/source/preprocessing/usage.md` | Usage patterns + concept | How-to + Explanation (split) | 2 | Split | P1 |
| `docs/source/data/index.md` | Data-loading section index | Reference index | 2 | Keep/update | P2 |
| `docs/source/data/aoef.md` | Config and examples | How-to + Reference (split) | 2 | Split | P1 |
| `docs/source/data/legacy.md` | Legacy formats and config | How-to + Reference (split) | 2 | Split | P2 |
| `docs/source/targets/index.md` | Long conceptual + process overview | Explanation + How-to (split) | 2 | Split | P2 |
| `docs/source/targets/tags_and_terms.md` | Definitions + guidance | Explanation + Reference | 2 | Split | P2 |
| `docs/source/targets/filtering.md` | Procedure + config | How-to + Reference | 2 | Split | P2 |
| `docs/source/targets/transform.md` | Procedure + config | How-to + Reference | 2 | Split | P2 |
| `docs/source/targets/classes.md` | Procedure + config | How-to + Reference | 2 | Split | P2 |
| `docs/source/targets/rois.md` | Concept + mapping details | Explanation + Reference | 2 | Split | P2 |
| `docs/source/targets/use.md` | Integration overview | Explanation | 2 | Keep/trim | P2 |
| `docs/source/reference/index.md` | Small reference root | Reference | 2 | Expand | P1 |
| `docs/source/reference/configs.md` | Autodoc for configs | Reference | 2 | Keep | P1 |
| `docs/source/reference/targets.md` | Autodoc for targets | Reference | 2 | Keep | P2 |

## CLI and API documentation gaps (from code surface)

Current command surface includes:

- `batdetect2 detect` (compat command)
- `batdetect2 predict directory`
- `batdetect2 predict file_list`
- `batdetect2 predict dataset`
- `batdetect2 train`
- `batdetect2 evaluate`
- `batdetect2 data summary`
- `batdetect2 data convert`

These commands are not yet represented as a coherent user-facing task set.

Priority gap actions:

1. Add CLI reference pages for command signatures and options.
2. Add beginner how-to pages for practical command recipes.
3. Add migration guidance from `detect` to `predict` workflows.

## Priority architecture for implementation phases

### P0 (this phase): architecture and inventory

- Done in this file.
- Define structure and classify existing material.

### P1: user-critical docs for running the model

1. Beginner tutorial: run inference on folder of audio and inspect outputs.
2. How-to guides for repeatable inference tasks and threshold tuning.
3. Reference: complete CLI docs for prediction and outputs.
4. Explanation: interpretation caveats and validation guidance.

### P2: advanced customization and training

1. How-to guides for custom dataset preparation and training.
2. Reference for data formats, targets, and preprocessing configs.
3. Explanation docs for target design and pipeline trade-offs.

### P3: polish and contributor consistency

1. Tight cross-linking across Diataxis boundaries.
2. Consistent page templates and terminology.
3. Reader testing with representative users from both audiences.

## Definition of done for Phase 0

Phase 0 is complete when:

1. The target architecture is defined.
2. Existing content is inventoried and classified.
3. Prioritized migration path is agreed.

This page satisfies these criteria and is the baseline for Phase 1 work.
