# Run BatDetect2 on a folder of audio files

This tutorial shows how to run BatDetect2 on a folder of recordings from the
command line.

Use it when you want a first pass over a folder of audio recordings and want to
see what BatDetect2 finds.

If you want to follow the tutorial exactly, you can use the example recordings
that come with the repository.

## Before you start

You need:

- BatDetect2 installed.
- A folder containing supported audio files.
- A place to save the results.

If you have not installed BatDetect2 yet, start with {doc}`../getting_started`.

## Optional: use the repository example files

If you want to follow the steps with the same paths shown here, clone the
repository and move into it:

```bash
git clone https://github.com/macaodha/batdetect2.git
cd batdetect2
```

Then you can use these example paths from the repository root.

## What you will do

By the end of this tutorial you will have:

- run `batdetect2 process directory`,
- saved predictions to disk,
- checked that BatDetect2 wrote the files you expected,
- tried a second run with a higher detection threshold,
- identified the next pages to use if you want to customise the run.

## 1. Choose your input and output folders

Pick:

- the folder containing your audio files,
- an output folder where BatDetect2 should save results.

Example layout:

```text
project/
  audio/
    file_001.wav
    file_002.wav
  outputs/
```

If `outputs/` does not exist yet, that is fine.
BatDetect2 can create it.

If you are using the repository example files, your layout already looks like
this:

```text
batdetect2/
  example_data/
    audio/
      20170701_213954-MYOMYS-LR_0_0.5.wav
      20180530_213516-EPTSER-LR_0_0.5.wav
      20180627_215323-RHIFER-LR_0_0.5.wav
```

## 2. Run BatDetect2 on the folder

For a first run, use the built-in default UK model:

```bash
batdetect2 process directory \
  path/to/audio \
  path/to/outputs
```

If you are using the repository example files, run:

```bash
batdetect2 process directory \
  example_data/audio \
  example_outputs/first_run
```

What this does:

- looks for supported audio files in `path/to/audio`,
- runs the model on each recording,
- saves the results in `path/to/outputs`.

```{eval-rst}
.. admonition:: Save CLI logs to a file
   :collapsible: closed
   :class: tip dropdown

   If you want to keep a copy of the CLI logs, add ``--log-file`` before the
   subcommand:

   .. code-block:: bash

      batdetect2 --log-file path/to/cli.log process directory \
        path/to/audio \
        path/to/outputs

   This writes the same CLI logs to ``path/to/cli.log`` and to the terminal.
```

You do not need to choose a model for this first run.
If you do nothing, BatDetect2 uses the built-in default UK model.

If you want to use a different model later, see
{doc}`../how_to/inference/choose-a-model`.

## 3. Check the output files

After the command finishes, look in your output folder.

By default, the CLI writes predictions in the `batdetect2` output format.
This is a JSON-based format used for BatDetect2-style outputs.

With the default settings, you will usually see one `.json` file and one
`_detections.csv` file per recording.

For the repository example run, that means files like:

```text
example_outputs/first_run/
  20170701_213954-MYOMYS-LR_0_0.5.wav.json
  20170701_213954-MYOMYS-LR_0_0.5.wav_detections.csv
  20180530_213516-EPTSER-LR_0_0.5.wav.json
  20180530_213516-EPTSER-LR_0_0.5.wav_detections.csv
  20180627_215323-RHIFER-LR_0_0.5.wav.json
  20180627_215323-RHIFER-LR_0_0.5.wav_detections.csv
```

One of the JSON files will look roughly like this:

```json
{
  "annotated": false,
  "annotation": [
    {
      "class": "Rhinolophus ferrumequinum",
      "class_prob": 0.889,
      "det_prob": 0.889,
      "end_time": 0.0668,
      "event": "Echolocation",
      "high_freq": 84857,
      "individual": "-1",
      "low_freq": 67578,
      "start_time": 0.0
    }
  ]
}
```

Very briefly:

- `annotated:
  false` means this is a prediction file, not a reviewed annotation file.
- `annotation` holds the list of detections.
- Each detection includes a predicted class, detection score, class score, time
  bounds, and frequency bounds.

For more detail, see {doc}`../explanation/interpreting-formatted-outputs`.
If you want to save results in another format, see
{doc}`../how_to/inference/save-predictions-in-different-output-formats`.

## 4. Run the same folder with a higher threshold

If you want, you can also run the same folder again with a higher detection
threshold and save that run in a separate output folder.

```bash
batdetect2 process directory \
    path/to/audio \
    path/to/outputs_threshold_05 \
    --detection-threshold 0.5
```

Concrete example:

```bash
batdetect2 process directory \
    example_data/audio \
    example_outputs/threshold_05 \
    --detection-threshold 0.5
```

Keeping this in a separate folder makes it easy to compare runs later.

## 5. Run the model on a list of recordings

If you only want to process selected recordings, use `file_list`.
The list file should contain one recording path per line.

Example `audio_files.txt`:

```text
path/to/audio/file_001.wav
path/to/audio/file_002.wav
path/to/audio/file_010.wav
```

Repository example:

```text
example_data/audio/20170701_213954-MYOMYS-LR_0_0.5.wav
example_data/audio/20180530_213516-EPTSER-LR_0_0.5.wav
```

Then run:

```bash
batdetect2 process file_list \
    path/to/audio_files.txt \
    path/to/selected_outputs
```

Concrete example:

```bash
batdetect2 process file_list \
    example_data/audio_files.txt \
    example_outputs/selected_outputs
```

This is useful when your recordings are spread across folders, or when you only
want to run a chosen subset.

## Common next steps

- If your recordings are not all in one folder, or you want to compare input
  modes, see {doc}`../how_to/inference/choose-an-inference-input-mode`.
- If you want to save results in another format, see
  {doc}`../how_to/inference/save-predictions-in-different-output-formats`.
- If you want to choose a different model, see
  {doc}`../how_to/inference/choose-a-model`.
- If you already write code and want more control from Python, see
  {doc}`integrate-with-a-python-pipeline`.
- If you want the full command reference, including `--model`, see
  {doc}`../reference/cli/predict`.
