# How to choose and configure evaluation tasks

Use this guide when the default evaluation tasks do not match the question you want to answer.

## Know the default first

By default, BatDetect2 evaluation starts with:

- sound event detection,
- sound event classification.

Those are good defaults for many projects, but not for all of them.

## Choose the task that matches the question

Common built-in task families include:

- `sound_event_detection`
- `sound_event_classification`
- `top_class_detection`
- `clip_detection`
- `clip_classification`

Choose based on the question you care about.

- Use sound-event tasks when you care about individual call events.
- Use clip tasks when you care about clip-level presence or clip-level class evidence.
- Use top-class detection when you want matching based on the highest-scoring class per detection.

## Configure tasks in `EvaluationConfig`

Example:

```yaml
tasks:
  - name: sound_event_detection
    prefix: detection
    affinity_threshold: 0.0
    strict_match: true
  - name: clip_classification
    prefix: clip_classification
```

Pass the config with:

```bash
batdetect2 evaluate \
  path/to/model.ckpt \
  path/to/test_dataset.yaml \
  --base-dir path/to/project_root \
  --evaluation-config path/to/evaluation.yaml
```

Include `--base-dir` when the dataset config resolves recordings through relative paths.

## Change one thing at a time

When comparing models or settings, avoid changing task definitions, thresholds, matching behavior, and datasets all at once.

Otherwise it becomes hard to explain why the metric changed.

## Related pages

- Evaluation tutorial: {doc}`../tutorials/evaluate-on-a-test-set`
- Evaluation config reference: {doc}`../reference/evaluation-config`
- Evaluation concepts: {doc}`../explanation/evaluation-concepts-and-matching`
