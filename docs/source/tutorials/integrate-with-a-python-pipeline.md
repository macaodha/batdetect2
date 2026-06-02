# Integrate with a Python pipeline

This tutorial shows a simple Python workflow for loading audio, running
BatDetect2, and inspecting the detections.

Use it when you want to work directly in Python rather than through the CLI.

If you mainly want to run the model on recordings, start with
{doc}`run-inference-on-folder` instead.

## Before you start

You need:

- BatDetect2 installed in your Python environment,
- at least one input audio file.

## What you will do

By the end of this tutorial you will have:

- created a `BatDetect2API` object,
- run inference on one file,
- inspected detections, scores, and features,
- used lower-level audio and spectrogram methods for more control,
- identified the next API workflows for batch processing, training, fine-tuning,
  and evaluation.

## 1. Create the API instance

For a first run, use the built-in default UK model:

```python
from batdetect2 import BatDetect2API

# If you don't specify a checkpoint the default model will be loaded
api = BatDetect2API.from_checkpoint()
```

If you want to use a different checkpoint later, see
{doc}`../how_to/inference/choose-a-model`.

## 2. Run inference on one file

`process_file` is the simplest Python entry point when you want one prediction
object per recording.

```python
from batdetect2 import BatDetect2API

api = BatDetect2API.from_checkpoint()
prediction = api.process_file("path/to/audio.wav")

for detection in prediction.detections:
    top_class = api.get_top_class_name(detection)
    score = detection.detection_score
    print(top_class, score)
```

## 3. Understand the prediction objects

`prediction` is a `ClipDetections` object.
See {doc}`../reference/detections` for the full reference.

Very briefly, `ClipDetections` represents all detections for one processed clip
or recording.
It includes:

- the clip metadata,
- the list of detections for that clip.

Each item in `prediction.detections` is a `Detection` object.

Each `Detection` includes:

- the time-frequency geometry of the event,
- a detection score,
- the class scores,
- a feature vector.

## 4. Inspect detection score and class scores

The detection score and the class scores answer different questions.

- `detection_score` is about whether the model thinks there is a call at that
  time-frequency location.
- `class_scores` are about which class the model prefers for that detected
  event.

So a detection can have a fairly strong detection score, but still have a more
uncertain class ranking.

```python
for detection in prediction.detections:
    print("top class:", api.get_top_class_name(detection))
    print("detection score:", detection.detection_score)
    print("class scores:")
    for class_name, score in api.get_class_scores(detection):
        print(f"  {class_name}: {score:.3f}")
```

If you want more detail on class-score inspection, see
{doc}`../how_to/analysis/inspect-class-scores-in-python`.

## 5. Inspect the detection features

Each detection also carries a `features` vector.

These are internal model features attached to the detection.
They can be useful for things like:

- exploratory visualisation,
- clustering similar detections,
- comparing detections across files,
- building downstream analysis pipelines.

They are useful descriptors, but they are not direct ecological labels by
themselves.

For more detail, see
{doc}`../how_to/analysis/inspect-detection-features-in-python` and
{doc}`../explanation/extracted-features-and-embeddings`.

## 6. Use lower-level audio and spectrogram methods for more control

If you want finer control over what gets processed and when, the API also lets
you work step by step.

For example, you can load the audio yourself, inspect the waveform length,
generate the spectrogram, and then run detection on that spectrogram:

```python
from batdetect2 import BatDetect2API

api = BatDetect2API.from_checkpoint()

audio = api.load_audio("path/to/audio.wav")
print(audio.shape)

spec = api.generate_spectrogram(audio)
print(spec.shape)

detections = api.process_spectrogram(spec)
print(len(detections))
```

This is helpful when you want to:

- inspect the loaded audio before inference,
- inspect the generated spectrogram,
- control which audio segment is processed,
- run only part of the pipeline in custom code.

You can also call `process_audio(audio)` directly if you already have the
waveform array in memory.

## 7. Use the wider API workflows

The Python API is not only for single-file inference.
It also exposes methods for batch processing, training, evaluation, and
fine-tuning.

Examples:

- `process_files(...)` for batch processing from Python,
- `train(...)` for training,
- `evaluate(...)` for evaluation,
- `finetune(...)` for fine-tuning.

Useful next pages:

- Choose a different model:
  {doc}`../how_to/inference/choose-a-model`
- Run batch predictions:
  {doc}`../how_to/inference/run-batch-predictions`
- Train a custom model:
  {doc}`train-a-custom-model`
- Evaluate on a test set:
  {doc}`evaluate-on-a-test-set`
- Fine-tune from a checkpoint:
  {doc}`../how_to/training/fine-tune-from-a-checkpoint`
- API reference:
  {doc}`../reference/api`
