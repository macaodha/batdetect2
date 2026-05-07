# Detections reference

These are the main prediction objects returned by BatDetect2 inference methods.

Defined in `batdetect2.postprocess.types`.

## `ClipDetections`

`ClipDetections` represents the predictions for one clip or one full recording.

Fields:

- `clip`
  - the `soundevent` clip metadata for the processed audio.
- `detections`
  - list of `Detection` objects for that clip.

## `Detection`

`Detection` represents one detected event.

Fields:

- `geometry`
  - time-frequency geometry for the detected event.
- `detection_score`
  - confidence that there is an event at this location.
- `class_scores`
  - class ranking scores for the detected event.
- `features`
  - per-detection feature vector from the model.

## Related pages

- Python tutorial:
  {doc}`../tutorials/integrate-with-a-python-pipeline`
- API reference:
  {doc}`api`
- What BatDetect2 predicts:
  {doc}`../explanation/what-batdetect2-predicts`
- Features and embeddings:
  {doc}`../explanation/extracted-features-and-embeddings`
