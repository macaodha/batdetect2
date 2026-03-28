# Home

Welcome to the batdetect2 docs.

## What is batdetect2?

`batdetect2` is a bat echolocation detection model.
It detects each individual echolocation call in an input spectrogram, draws a
box around each call event, and predicts the most likely species for that call.
A recording can contain many calls from different species.

The current default model is trained for 17 UK species but you can also train
new models from your own annotated data.

For details on the approach please read our pre-print:
[Towards a General Approach for Bat Echolocation Detection and Classification](https://www.biorxiv.org/content/10.1101/2022.12.14.520490v1)

## What you can do

- Run inference on your recordings and export predictions for downstream
  analysis:
  {doc}`tutorials/run-inference-on-folder`
- Train a custom model on your own annotated data:
  {doc}`tutorials/train-a-custom-model`
- Evaluate model performance on a held-out test set:
  {doc}`tutorials/evaluate-on-a-test-set`
- Integrate batdetect2 into Python scripts and notebooks:
  {doc}`tutorials/integrate-with-a-python-pipeline`

```{warning}
Treat outputs as model predictions, not ground truth.
Always validate on reviewed local data before using results for ecological
inference.
```

## Where to start

If you are new, start with {doc}`getting_started`.

For a low-code path, go to {doc}`tutorials/index`.
If you are Python-savvy and want more control, go to {doc}`how_to/index` and
{doc}`reference/index`.

Each section has a different purpose:
some pages teach by example, some focus on practical tasks, some are lookup
material, and some explain trade-offs.

| Section       | Best for                                    | Start here               |
| ------------- | ------------------------------------------- | ------------------------ |
| Tutorials     | Learning by doing                           | {doc}`tutorials/index`   |
| How-to guides | Solving practical tasks                     | {doc}`how_to/index`      |
| Reference     | Looking up commands, configs, and APIs      | {doc}`reference/index`   |
| Explanation   | Understanding design choices and trade-offs | {doc}`explanation/index` |

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

If you use batdetect2 in research, please cite:

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
```

```{toctree}
:maxdepth: 1
:caption: Contributing

development/index
```
