# Getting started

BatDetect2 is both a command line tool (CLI) and a Python library.

- Use the CLI if you want to run existing models or train your own models from
  the terminal.
- Use the Python package if you want to integrate BatDetect2 into your own
  scripts, notebooks, or analysis pipeline.

If you want to try BatDetect2 before installing anything locally:

- [Hugging Face demo (UK species)](https://huggingface.co/spaces/macaodha/batdetect2)
- [Google Colab notebook](https://colab.research.google.com/github/macaodha/batdetect2/blob/master/batdetect2_notebook.ipynb)

## Prerequisites

We recommend `uv` for both workflows.
`uv` is a fast Python package and environment manager that keeps installs
isolated and reproducible.

- Use `uv tool` to install the CLI.
- Use `uv add` to add `batdetect2` as a dependency in a Python project.

Install `uv` first by following their
[installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

## Install the CLI

The following installs `batdetect2` in an isolated tool environment and exposes
the `batdetect2` command on your machine.

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

## Integrate with your Python project

If you are using BatDetect2 from Python code, add it to your project
dependencies:

```bash
uv add batdetect2
```

This keeps dependency metadata and the environment in sync.

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

- Run your first detection workflow:
  {doc}`tutorials/run-inference-on-folder`
- For practical task recipes, go to {doc}`how_to/index`
- For command and option details, go to {doc}`reference/cli`
