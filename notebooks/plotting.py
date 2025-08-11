import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    from batdetect2.data import load_dataset_config, load_dataset
    from batdetect2.preprocess import load_preprocessing_config, build_preprocessor
    from batdetect2 import api
    from soundevent import data
    from batdetect2.evaluate.types import MatchEvaluation
    from batdetect2.types import Annotation
    from batdetect2.compat import annotation_to_sound_event_prediction
    from batdetect2.plotting import (
        plot_clip,
        plot_clip_annotation,
        plot_clip_prediction,
        plot_matches,
        plot_false_positive_match,
        plot_false_negative_match,
        plot_true_positive_match,
        plot_cross_trigger_match,
    )
    return (
        MatchEvaluation,
        annotation_to_sound_event_prediction,
        api,
        build_preprocessor,
        data,
        load_dataset,
        load_dataset_config,
        load_preprocessing_config,
        plot_clip_annotation,
        plot_clip_prediction,
        plot_cross_trigger_match,
        plot_false_negative_match,
        plot_false_positive_match,
        plot_matches,
        plot_true_positive_match,
    )


@app.cell
def _(build_preprocessor, load_dataset_config, load_preprocessing_config):
    dataset_config = load_dataset_config(
        path="example_data/config.yaml", field="datasets.train"
    )

    preprocessor_config = load_preprocessing_config(
        path="example_data/config.yaml", field="preprocess"
    )

    preprocessor = build_preprocessor(preprocessor_config)
    return dataset_config, preprocessor


@app.cell
def _(dataset_config, load_dataset):
    dataset = load_dataset(dataset_config)
    return (dataset,)


@app.cell
def _(dataset):
    clip_annotation = dataset[1]
    return (clip_annotation,)


@app.cell
def _(clip_annotation, plot_clip_annotation, preprocessor):
    plot_clip_annotation(
        clip_annotation, preprocessor=preprocessor, figsize=(15, 5)
    )
    return


@app.cell
def _(annotation_to_sound_event_prediction, api, clip_annotation, data):
    audio = api.load_audio(clip_annotation.clip.recording.path)
    detections, features, spec = api.process_audio(audio)
    clip_prediction = data.ClipPrediction(
        clip=clip_annotation.clip,
        sound_events=[
            annotation_to_sound_event_prediction(
                prediction, clip_annotation.clip.recording
            )
            for prediction in detections
        ],
    )
    return (clip_prediction,)


@app.cell
def _(clip_prediction, plot_clip_prediction):
    plot_clip_prediction(clip_prediction, figsize=(15, 5))
    return


@app.cell
def _():
    from batdetect2.evaluate import match_predictions_and_annotations
    import random
    return match_predictions_and_annotations, random


@app.cell
def _(data, random):
    def add_noise(clip_annotation, time_buffer=0.003, freq_buffer=1000):
        def _add_bbox_noise(bbox):
            start_time, low_freq, end_time, high_freq = bbox.coordinates
            return data.BoundingBox(
                coordinates=[
                    start_time + random.uniform(-time_buffer, time_buffer),
                    low_freq + random.uniform(-freq_buffer, freq_buffer),
                    end_time + random.uniform(-time_buffer, time_buffer),
                    high_freq + random.uniform(-freq_buffer, freq_buffer),
                ]
            )

        def _add_noise(se):
            return se.model_copy(
                update=dict(
                    sound_event=se.sound_event.model_copy(
                        update=dict(
                            geometry=_add_bbox_noise(se.sound_event.geometry)
                        )
                    )
                )
            )

        return clip_annotation.model_copy(
            update=dict(
                sound_events=[
                    _add_noise(se) for se in clip_annotation.sound_events
                ]
            )
        )


    def drop_random(obj, p=0.5):
        return obj.model_copy(
            update=dict(
                sound_events=[se for se in obj.sound_events if random.random() > p]
            )
        )
    return add_noise, drop_random


@app.cell
def _(
    add_noise,
    clip_annotation,
    clip_prediction,
    drop_random,
    match_predictions_and_annotations,
):


    matches = match_predictions_and_annotations(
        drop_random(add_noise(clip_annotation), p=0.2),
        drop_random(clip_prediction),
    )
    return (matches,)


@app.cell
def _(clip_annotation, matches, plot_matches):
    plot_matches(matches, clip_annotation.clip, figsize=(15, 5))
    return


@app.cell
def _(matches):
    true_positives = []
    false_positives = []
    false_negatives = []

    for match in matches:
        if match.source is None and match.target is not None:
            false_negatives.append(match)
        elif match.target is None and match.source is not None:
            false_positives.append(match)
        elif match.target is not None and match.source is not None:
            true_positives.append(match)
        else:
            continue

    return false_negatives, false_positives, true_positives


@app.cell
def _(MatchEvaluation, false_positives, plot_false_positive_match):
    false_positive = false_positives[0]
    false_positive_eval = MatchEvaluation(
        match=false_positive,
        gt_det=False,
        gt_class=None,
        pred_score=false_positive.source.score,
        pred_class_scores={
            "myomyo": 0.2
        }
    )

    plot_false_positive_match(false_positive_eval)
    return


@app.cell
def _(MatchEvaluation, false_negatives, plot_false_negative_match):
    false_negative = false_negatives[0]
    false_negative_eval = MatchEvaluation(
        match=false_negative,
        gt_det=True,
        gt_class="myomyo",
        pred_score=None,
        pred_class_scores={}
    )

    plot_false_negative_match(false_negative_eval)

    return


@app.cell
def _(MatchEvaluation, plot_true_positive_match, true_positives):
    true_positive = true_positives[0]
    true_positive_eval = MatchEvaluation(
        match=true_positive,
        gt_det=True,
        gt_class="myomyo",
        pred_score=0.87,
        pred_class_scores={
            "pyomyo": 0.84,
            "pippip": 0.84,
        }
    )

    plot_true_positive_match(true_positive_eval)
    return (true_positive,)


@app.cell
def _(MatchEvaluation, plot_cross_trigger_match, true_positive):
    cross_trigger_eval = MatchEvaluation(
        match=true_positive,
        gt_det=True,
        gt_class="myomyo",
        pred_score=0.87,
        pred_class_scores={
            "pippip": 0.84,
            "myomyo": 0.84,
        }
    )

    plot_cross_trigger_match(cross_trigger_eval)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
