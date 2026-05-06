# Getting started

BatDetect2 can be used in two ways: through the `batdetect2` command line interface (CLI), or as the `batdetect2` Python package.
The CLI route does not require coding.
You run commands in the terminal and, in some cases, write configuration files.
The Python route gives you more flexibility and lets you integrate the model into your own workflows or experiments.
For most common use cases, both routes give you the same results.

## Try it out

If you want to try BatDetect2 before installing anything locally:

- [Hugging Face demo (UK species)](https://huggingface.co/spaces/macaodha/batdetect2)
- [Google Colab notebook](https://colab.research.google.com/github/macaodha/batdetect2/blob/master/batdetect2_notebook.ipynb)

## Installation

To use `batdetect2` on your machine, you need to install it first.
We recommend using `uv` for that.
`uv` is a tool that helps manage Python software cleanly, without mixing it into the rest of your machine.
Install `uv` first by following the [installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

### One-off usage

If you are not ready to install `batdetect2` permanently, you can try it with:

```bash
uvx batdetect2
```

This still downloads the code and dependencies and runs them on your machine, but the environment is temporary.

### Install the CLI

If you want the `batdetect2` CLI to always be available in your terminal, run:

```bash
uv tool install batdetect2
```

If you need to upgrade later:

```bash
uv tool upgrade batdetect2
```

Verify the CLI is available:

```bash
batdetect2
```

You can then run your first workflow.
See {doc}`tutorials/run-inference-on-folder` for more details.

### Add it to your Python project

If you are using BatDetect2 from Python code and already manage your projects with `uv`, you can add it with:

```bash
uv add batdetect2
```

If you want to upgrade it later:

```bash
uv add -U batdetect2
```

#### Alternative with `pip`

If you prefer `pip`, you can use:

```bash
pip install batdetect2
```

It is a good idea to create a separate virtual environment first so this does not interfere with other Python environments.

```bash
python -m venv .venv
source .venv/bin/activate
```

## What's next

- Run your first workflow on a folder of recordings: {doc}`tutorials/run-inference-on-folder`
- If you write code and want the Python route: {doc}`tutorials/integrate-with-a-python-pipeline`
- For common practical tasks, go to {doc}`how_to/index`
- For detailed command help, go to {doc}`reference/cli/index`
- To understand the model and its outputs, go to {doc}`explanation/index`
