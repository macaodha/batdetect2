# Tutorial: Integrate with a Python pipeline

This tutorial shows a minimal Python workflow for loading audio, running
batdetect2, and collecting detections for downstream analysis.

## Before you start

- BatDetect2 installed in your Python environment.
- A model checkpoint.
- At least one input audio file.

## Tutorial steps

1. Load BatDetect2 in Python.
2. Create an API instance from a checkpoint.
3. Run `process_file` on one audio file.
4. Read detection fields and class scores.
5. Save or pass detections to your downstream pipeline.

## Example code

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

## What to do next

- See API/config references: {doc}`../reference/index`
- Learn practical CLI alternatives: {doc}`run-inference-on-folder`

This page is a starter scaffold and will be expanded with a full worked
example.
