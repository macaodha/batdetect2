# Getting started

If you want to run BatDetect2 on your recordings, start with the command-line
route below.

You do not need to write Python code for a standard first run.

BatDetect2 also has a Python interface, but that is mainly for users writing
their own analysis scripts.

- Use the command-line route if you want to run an existing model or train your
  own model by typing commands in a terminal window.
- Use the Python route only if you already want to work in scripts or notebooks.

```{note}
If you are looking for the previous BatDetect2 workflow based on `batdetect2 detect` or `batdetect2.api`, go to {doc}`legacy/index`.
New docs default to the current `process` CLI and `BatDetect2API` workflow.
```

If you want to try BatDetect2 before installing anything locally:

- [Hugging Face demo (UK species)](https://huggingface.co/spaces/macaodha/batdetect2)
- [Google Colab notebook](https://colab.research.google.com/github/macaodha/batdetect2/blob/master/batdetect2_notebook.ipynb)

## The simplest route for most users

1. Install BatDetect2.
2. Use a model checkpoint.
3. Run the first tutorial on a folder of recordings.

If that is what you want, you can ignore the Python sections for now.

## Install BatDetect2

We recommend `uv` for both workflows.

`uv` is a tool that helps install Python software cleanly, without mixing it
into the rest of your machine.

- Use `uv tool` to install the CLI.
- Use `uv add` to add `batdetect2` as a dependency in a Python project.

Install `uv` first by following their
[installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

## Install the CLI

The following installs `batdetect2` in its own small environment and makes the
`batdetect2` command available on your machine.

```bash
uv tool install batdetect2
```

If you need to upgrade later:

```bash
uv tool upgrade batdetect2
```

Verify the CLI is available:

```bash
batdetect2 --help
```

Run your first workflow:

Go to {doc}`tutorials/run-inference-on-folder` for a complete first run.

## Choose a model checkpoint

The current command-line and Python workflows expect an explicit checkpoint
path.

A checkpoint is the saved model file that BatDetect2 will use for prediction.

You can use:

- a checkpoint you trained yourself, or
- a checkpoint distributed with your installation or repository checkout.

In this repository checkout, an example pretrained checkpoint is available at:

```text
src/batdetect2/models/checkpoints/Net2DFast_UK_same.pth.tar
```

Use that path in the tutorial commands if you want a concrete starting point
from this source tree.

## Python route for users writing code

If you are using BatDetect2 from Python code, add it to your Python project:

```bash
uv add batdetect2
```

This keeps your project settings and installed packages in sync.

### Alternative with `pip`

If you prefer `pip`, create and activate a virtual environment first:

```bash
python -m venv .venv
source .venv/bin/activate
```

Then install from PyPI:

```bash
pip install batdetect2
```

## What's next

- Run your first workflow on a folder of recordings:
  {doc}`tutorials/run-inference-on-folder`
- If you write code and want the Python route:
  {doc}`tutorials/integrate-with-a-python-pipeline`
- For common practical tasks, go to {doc}`how_to/index`
- For detailed command help, go to {doc}`reference/cli/index`
- To understand outputs and trade-offs, go to {doc}`explanation/index`
