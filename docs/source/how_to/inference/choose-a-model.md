# How to choose a model

Use this guide when you want to choose which model checkpoint BatDetect2 loads.

You can choose a model in both the CLI and the Python API.

## Where you can choose the model

In the CLI, use `--model` with commands that load a checkpoint, including:

- `batdetect2 process`
- `batdetect2 evaluate`
- `batdetect2 train`
- `batdetect2 finetune`

In Python, pass the model source to `BatDetect2API.from_checkpoint(...)`.

If you do not choose a model, BatDetect2 uses the built-in default UK model.

## Use a local checkpoint path

Use a local path when you already have a checkpoint file on disk.

CLI example:

```bash
batdetect2 process directory \
    path/to/audio \
    path/to/outputs \
    --model path/to/model.ckpt
```

Python example:

```python
from batdetect2.api_v2 import BatDetect2API

api = BatDetect2API.from_checkpoint("path/to/model.ckpt")
```

## Use a bundled checkpoint alias

BatDetect2 also supports bundled checkpoint aliases.

The built-in UK model is available as `uk_same`.
The alias `batdetect2_uk_same` also works.

CLI example:

```bash
batdetect2 process directory \
    path/to/audio \
    path/to/outputs \
    --model uk_same
```

Python example:

```python
from batdetect2.api_v2 import BatDetect2API

api = BatDetect2API.from_checkpoint("uk_same")
```

## Use a Hugging Face URI

You can also load a checkpoint from Hugging Face with a URI like:

```text
hf://owner/repo/path/to/model.ckpt
```

This needs the optional Hugging Face dependency to be installed.
For example, install it with `pip install batdetect2[huggingface]`.

CLI example:

```bash
batdetect2 process directory \
    path/to/audio \
    path/to/outputs \
    --model hf://owner/repo/path/to/model.ckpt
```

Python example:

```python
from batdetect2.api_v2 import BatDetect2API

api = BatDetect2API.from_checkpoint(
    "hf://owner/repo/path/to/model.ckpt"
)
```

## Choose the right source

- Use a local path when you already have a checkpoint file.
- Use an alias when you want one of the bundled models.
- Use a Hugging Face URI when the checkpoint lives in a Hugging Face repo.

## Related pages

- Run inference on a folder:
  {doc}`../../tutorials/run-inference-on-folder`
- `BatDetect2API` reference:
  {doc}`../../reference/api`
- Process command reference:
  {doc}`../../reference/cli/predict`
- Train a custom model:
  {doc}`../../tutorials/train-a-custom-model`
- Fine-tune from a checkpoint:
  {doc}`../training/fine-tune-from-a-checkpoint`
