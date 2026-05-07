# Evaluation config reference

`EvaluationConfig` defines which evaluation tasks run and which plots they generate.

Defined in `batdetect2.evaluate.config`.

## Top-level fields

- `tasks`
  - list of task configs.

## Built-in task families

Current built-in tasks include:

- `sound_event_detection`
- `sound_event_classification`
- `top_class_detection`
- `clip_detection`
- `clip_classification`

## Shared task controls

Common task-level controls include:

- `prefix`
- `ignore_start_end`

Sound-event-style tasks also support:

- `affinity`
- `affinity_threshold`
- `strict_match`

## Default behavior

The default evaluation config starts with:

- sound event detection,
- sound event classification.

## Related pages

- Choose and configure evaluation tasks: {doc}`../how_to/choose-and-configure-evaluation-tasks`
- Evaluation concepts: {doc}`../explanation/evaluation-concepts-and-matching`
- Evaluate CLI reference: {doc}`cli/evaluate`
