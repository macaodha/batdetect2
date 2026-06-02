# How to interpret evaluation outputs

Use this guide after `batdetect2 evaluate` has written metrics and plots to disk.

## Start by identifying the task

Do not interpret a metric until you know which evaluation task produced it.

For example, a detection score and a clip-classification score answer different questions.

## Read the output directory as a bundle

Treat the evaluation output directory as one package:

- metrics,
- plots,
- saved predictions,
- config context.

Do not lift a single number out of context and treat it as the whole story.

## Look for failure patterns, not just overall averages

Check:

- whether errors concentrate in certain taxa,
- whether specific sites or recorder setups behave differently,
- whether threshold choices are driving the result,
- whether predictions are near clip boundaries or matching thresholds.

## Keep validation and deployment questions separate

A model can look good on one task and still be a poor fit for your deployment question.

Interpret the outputs in relation to the real use case, not only the easiest metric to report.

## Related pages

- Evaluation tutorial: {doc}`../tutorials/evaluate-on-a-test-set`
- Evaluation concepts: {doc}`../explanation/evaluation-concepts-and-matching`
- Model output and validation: {doc}`../explanation/model-output-and-validation`
