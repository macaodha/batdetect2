# How to inspect class scores in Python

Use this guide when you need more than the top class label for each detection.

## Get the ranked class scores

`BatDetect2API.get_class_scores` returns `(class_name, score)` pairs for one detection.

```python
from pathlib import Path

from batdetect2.api_v2 import BatDetect2API

api = BatDetect2API.from_checkpoint(Path("path/to/model.ckpt"))
prediction = api.process_file(Path("path/to/audio.wav"))

for detection in prediction.detections:
    print("detection score:", detection.detection_score)
    for class_name, score in api.get_class_scores(detection):
        print(class_name, score)
```

## Separate detection confidence from class ranking

Keep these two ideas separate:

- `detection_score` tells you how strongly the model kept the event as a detection,
- `class_scores` tell you how the model ranked classes for that detected event.

A detection can have a reasonable detection score while still having uncertain class ranking.

## Hide the top class if needed

If you want to inspect only the alternatives, pass `include_top_class=False`.

```python
api.get_class_scores(detection, include_top_class=False)
```

## Related pages

- Python tutorial: {doc}`../tutorials/integrate-with-a-python-pipeline`
- API reference: {doc}`../reference/api`
- Understanding scores: {doc}`../explanation/what-batdetect2-predicts`
