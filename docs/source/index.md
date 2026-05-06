# Home

Welcome to the BatDetect2 documentation.

## What is BatDetect2?

`batdetect2` is a deep learning model and software package for detecting and
classifying bat echolocation calls in high-frequency audio recordings.

You can use it from the command line or from Python, depending on how much
control you need.

In practice, BatDetect2 scans a recording, finds sounds that look like bat
calls, and returns one result for each detected call.
Each result can include where the call appears in the recording, shown as a box
with start and end time and the lowest and highest frequency, how confident the
model is that it found a call, and how strongly it matches the available
classes.

The built-in default model is trained for 17 UK species.
The package also supports custom training, fine-tuning, evaluation, and more
advanced workflows from Python.

For more detail on the underlying approach, see the pre-print:
[Towards a General Approach for Bat Echolocation Detection and Classification](https://www.biorxiv.org/content/10.1101/2022.12.14.520490v1)

```{warning}
Treat outputs as model predictions, not ground truth.
Always validate on reviewed local data before using results for ecological inference.
```

## What can I do with it?

- I want to run the model on my recordings:
  {doc}`tutorials/run-inference-on-folder`
- I write code and want to use it from Python:
  {doc}`tutorials/integrate-with-a-python-pipeline`
- I want to train or fine-tune a custom model:
  {doc}`tutorials/train-a-custom-model`
- I want to evaluate a trained model on held-out data:
  {doc}`tutorials/evaluate-on-a-test-set`

```{note}
Looking for the previous BatDetect2 workflow?
See {doc}`legacy/index`.
The legacy docs are still available, but new workflows should use `batdetect2 process` and `BatDetect2API`.
```

## How to use this site

Start with {doc}`getting_started` if you are new.

Then choose the section that matches what you need.

If you are here mainly to run the model on recordings, start with Tutorials.

| Section       | Best for                                      | Start here               |
| ------------- | --------------------------------------------- | ------------------------ |
| Tutorials     | Step-by-step routes for the most common tasks | {doc}`tutorials/index`   |
| How-to guides | Answers to specific practical questions       | {doc}`how_to/index`      |
| Reference     | Detailed command and settings help            | {doc}`reference/index`   |
| Understanding | Concepts, interpretation, and trade-offs      | {doc}`explanation/index` |
| Legacy        | Previous workflow and migration guidance      | {doc}`legacy/index`      |

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

or the bibtex entry

```bibtex
@article{batdetect2_2022,
  title         = {Towards a General Approach for Bat Echolocation Detection and Classification},
  author        = {Mac Aodha, Oisin and Mart\'{i}nez Balvanera, Santiago and Damstra, Elise and Cooke, Martyn and Eichinski, Philip and Browning, Ella and Barataudm, Michel and Boughey, Katherine and Coles, Roger and Giacomini, Giada and MacSwiney G., M. Cristina and K. Obrist, Martin and Parsons, Stuart and Sattler, Thomas and Jones, Kate E.},
  journal       = {bioRxiv},
  year          = {2022}
}
```

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
