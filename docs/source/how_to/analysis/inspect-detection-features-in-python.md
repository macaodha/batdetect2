# How to inspect detection features in Python

Use this guide when you want the per-detection feature vectors exposed by the
current API.

## Get the feature vector for one detection

Each detection carries a `features` vector.

The API exposes it through `get_detection_features`.

```python
from pathlib import Path

from batdetect2.api_v2 import BatDetect2API

api = BatDetect2API.from_checkpoint(Path("path/to/model.ckpt"))
prediction = api.process_file(Path("path/to/audio.wav"))

for detection in prediction.detections:
    features = api.get_detection_features(detection)
    print(features.shape)
```

## Use features for exploration, not as ground truth labels

These features are internal model representations attached to detections.

They can be useful for:

- exploratory visualization,
- downstream clustering,
- comparison across detections,
- building extra analysis pipelines.

They do not replace validation.

They also do not automatically have a one-to-one interpretation as ecological
variables.

## Save predictions with features included

If you need features on disk, use an output format that supports them, such as
`raw` or `parquet`, and keep feature inclusion enabled.

See {doc}`../inference/save-predictions-in-different-output-formats`.

## Related pages

- Understanding features and embeddings:
  {doc}`../../explanation/extracted-features-and-embeddings`
- Output formats reference:
  {doc}`../../reference/configs/outputs/output-formats`
- API reference:
  {doc}`../../reference/api`
