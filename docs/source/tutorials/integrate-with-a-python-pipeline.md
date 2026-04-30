# Tutorial: Integrate with a Python pipeline

This tutorial shows a minimal Python workflow for loading audio, running
batdetect2, and collecting detections for downstream analysis.

This tutorial is for people who already want to work in Python.

If you mainly want to run the model on recordings,
start with {doc}`run-inference-on-folder` instead.

## Before you start

- BatDetect2 installed in your Python environment.
- A model checkpoint.
- At least one input audio file.

```{note}
This page is more technical than the standard first-run tutorial.
You do not need this page for a normal first use of BatDetect2.
```

If you are working from this repository checkout, you can start with:

```text
src/batdetect2/models/checkpoints/Net2DFast_UK_same.pth.tar
```

## Outcome

By the end of this tutorial you will have:

- created a `BatDetect2API` object,
- run inference on one file,
- inspected the top class, class-score list, and detection score,
- identified where to go next for feature extraction, saving predictions, and batch workflows.

## 1. Create the API instance

Load the checkpoint once and reuse the API object for multiple files.

```python
from pathlib import Path

from batdetect2.api_v2 import BatDetect2API

api = BatDetect2API.from_checkpoint(Path("path/to/model.ckpt"))
```

## 2. Run inference on one file

`process_file` is the simplest Python entry point when you want one prediction object per recording.

```python
from pathlib import Path

from batdetect2.api_v2 import BatDetect2API

api = BatDetect2API.from_checkpoint(Path("path/to/model.ckpt"))
prediction = api.process_file(Path("path/to/audio.wav"))

for detection in prediction.detections:
    top_class = api.get_top_class_name(detection)
    score = detection.detection_score
    print(top_class, score)
```

`prediction` is a `ClipDetections` object.

It contains:

- the clip metadata,
- a list of detections,
- a box for each detected event,
- one detection score per event,
- a full list of class scores per event,
- a feature vector per event.

## 3. Inspect class scores, not just the top class

If you are exploring results,
it is often useful to inspect the full ranked class-score list.

```python
for detection in prediction.detections:
    print("top class:", api.get_top_class_name(detection))
    print("detection score:", detection.detection_score)
    print("class scores:")
    for class_name, score in api.get_class_scores(detection):
        print(f"  {class_name}: {score:.3f}")
```

This helps separate two different questions:

- "Did the model think there was a call here?"
- "If there was a call, which class did it score highest?"

## 4. Keep the first workflow small

Before scaling up, run the API on a few representative files and inspect the results manually.

This catches path issues and obviously implausible outputs early.

## 5. Move to the right next workflow

Once the single-file path is working, choose the next page based on what you need:

- save predictions to disk,
- inspect class scores more carefully,
- inspect detection features,
- process many files in one run.

## What to do next

- API reference: {doc}`../reference/api`
- Inspect ranked class scores: {doc}`../how_to/inspect-class-scores-in-python`
- Inspect detection features: {doc}`../how_to/inspect-detection-features-in-python`
- Save predictions to disk: {doc}`../how_to/save-predictions-in-different-output-formats`
- Learn the CLI happy path: {doc}`run-inference-on-folder`
