# Evaluation concepts and matching

Evaluation is not just "run predictions and compute one number".

The reported metric depends on the evaluation task, the matching rule, and the
treatment of clip boundaries and generic labels.

## Task families answer different questions

Built-in task families include:

- sound event detection,
- sound event classification,
- top-class detection,
- clip detection,
- clip classification.

Choose the task that matches the scientific or engineering question.

## Matching matters

For sound-event-style tasks, predictions and annotations are matched using an
affinity function.

Important controls include:

- `affinity`,
- `affinity_threshold`,
- `strict_match`,
- `ignore_start_end`.

Small changes here can change the reported metric without changing the
underlying predictions.

## Boundary handling matters

The evaluation base task can exclude events near clip boundaries through
`ignore_start_end`.

This is useful when clip boundaries make matches ambiguous.

## Generic labels can matter in classification

Classification tasks can include or exclude generic targets depending on
configuration.

That affects what counts as a valid class-level comparison.

## Related pages

- Evaluate on a test set:
  {doc}`../tutorials/evaluate-on-a-test-set`
- Evaluation config reference:
  {doc}`../reference/configs/evaluation/evaluation-config`
- Model output and validation:
  {doc}`model-output-and-validation`
