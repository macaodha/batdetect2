# Home

Welcome to the BatDetect2 documentation.

## What is BatDetect2?

`batdetect2` detects bat echolocation calls in audio recordings.

It can help you screen large collections of recordings, find files that need
expert review, and support ecology and conservation work where manual review
alone would be slow.

In practice, BatDetect2 takes recordings, looks for likely bat calls, draws a
box around each detected event, and scores the most likely class for that event.

The current default model is trained for 17 UK species.

The library also supports custom training, fine-tuning, evaluation, and more
advanced use from Python.

For details on the underlying approach, see the pre-print:
[Towards a General Approach for Bat Echolocation Detection and Classification](https://www.biorxiv.org/content/10.1101/2022.12.14.520490v1)

## A good first use for BatDetect2

BatDetect2 is a good fit when you want to:

- scan many recordings for likely bat activity,
- prioritize files for expert review,
- compare outputs across projects with appropriate caution,
- build reviewed local datasets for later model improvement.

It is not a substitute for validation.

## Main user journeys

- I want to run the model on my recordings:
  {doc}`tutorials/run-inference-on-folder`
- I write code and want to use Python:
  {doc}`tutorials/integrate-with-a-python-pipeline`
- I want to train or fine-tune a custom model:
  {doc}`tutorials/train-a-custom-model`
- I want to evaluate a trained model on held-out data:
  {doc}`tutorials/evaluate-on-a-test-set`

```{warning}
Treat outputs as model predictions, not ground truth.
Always validate on reviewed local data before using results for ecological inference.
```

```{note}
Looking for the previous BatDetect2 workflow?
See {doc}`legacy/index`.
The legacy docs are still available, but new workflows should use `batdetect2 process` and `BatDetect2API`.
```

## How to use this site

Start with {doc}`getting_started` if you are new.

Then choose the section that matches what you need.

If you are here mainly to run the model on recordings, start with Tutorials.

| Section | Best for | Start here |
| --- | --- | --- |
| Tutorials | Step-by-step routes for the most common tasks | {doc}`tutorials/index` |
| How-to guides | Answers to specific practical questions | {doc}`how_to/index` |
| Reference | Detailed command and settings help | {doc}`reference/index` |
| Understanding | Concepts, interpretation, and trade-offs | {doc}`explanation/index` |
| Legacy | Previous workflow and migration guidance | {doc}`legacy/index` |

## Get in touch

- GitHub repository:
  [macaodha/batdetect2](https://github.com/macaodha/batdetect2)
- Questions, bug reports, and feature requests:
  [GitHub Issues](https://github.com/macaodha/batdetect2/issues)
- Common questions:
  {doc}`faq`
- Want to contribute?
  See {doc}`development/index`

## Cite this work

If you use BatDetect2 in research, please cite:

Mac Aodha, O., Martinez Balvanera, S., Damstra, E., et al.
(2022).
_Towards a General Approach for Bat Echolocation Detection and Classification_.
bioRxiv.

```{toctree}
:maxdepth: 1
:caption: Get Started

getting_started
faq
tutorials/index
how_to/index
reference/index
explanation/index
legacy/index
```

```{toctree}
:maxdepth: 1
:caption: Contributing

development/index
```
