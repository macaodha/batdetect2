import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from batdetect2.data import (
        load_dataset_config,
        load_dataset,
        extract_recordings_df,
        extract_sound_events_df,
        compute_class_summary,
    )
    return (
        compute_class_summary,
        extract_recordings_df,
        extract_sound_events_df,
        load_dataset,
        load_dataset_config,
    )


@app.cell
def _(mo):
    dataset_config_browser = mo.ui.file_browser(
        selection_mode="file",
        multiple=False,
    )
    dataset_config_browser
    return (dataset_config_browser,)


@app.cell
def _(dataset_config_browser, load_dataset_config, mo):
    mo.stop(dataset_config_browser.path() is None)
    dataset_config = load_dataset_config(dataset_config_browser.path())
    return (dataset_config,)


@app.cell
def _(dataset_config, load_dataset):
    dataset = load_dataset(dataset_config, base_dir="../paper/")
    return (dataset,)


@app.cell
def _():
    from batdetect2.targets import load_target_config, build_targets
    return build_targets, load_target_config


@app.cell
def _(mo):
    targets_config_browser = mo.ui.file_browser(
        selection_mode="file",
        multiple=False,
    )
    targets_config_browser
    return (targets_config_browser,)


@app.cell
def _(load_target_config, mo, targets_config_browser):
    mo.stop(targets_config_browser.path() is None)
    targets_config = load_target_config(targets_config_browser.path())
    return (targets_config,)


@app.cell
def _(build_targets, targets_config):
    targets = build_targets(targets_config)
    return (targets,)


@app.cell
def _():
    import pandas as pd
    from soundevent.geometry import compute_bounds
    return


@app.cell
def _(dataset, extract_recordings_df):
    recordings = extract_recordings_df(dataset)
    recordings
    return


@app.cell
def _(dataset, extract_sound_events_df, targets):
    sound_events = extract_sound_events_df(dataset, targets)
    sound_events
    return


@app.cell
def _(compute_class_summary, dataset, targets):
    compute_class_summary(dataset, targets)
    return


@app.cell
def _():
    from batdetect2.data.split import split_dataset_by_recordings
    return (split_dataset_by_recordings,)


@app.cell
def _(dataset, split_dataset_by_recordings, targets):
    train_dataset, val_dataset = split_dataset_by_recordings(dataset, targets, random_state=42)
    return train_dataset, val_dataset


@app.cell
def _(compute_class_summary, targets, train_dataset):
    compute_class_summary(train_dataset, targets)
    return


@app.cell
def _(compute_class_summary, targets, val_dataset):
    compute_class_summary(val_dataset, targets)
    return


@app.cell
def _():
    from soundevent import io, data
    from pathlib import Path
    return Path, data, io


@app.cell
def _(Path, data, io, train_dataset):
    io.save(
        data.AnnotationSet(
            name="batdetect2_tuning_train",
            description="Set of annotations used as the train dataset for the hyper-parameter tuning stage.",
            clip_annotations=train_dataset,
        ),
        Path("../paper/data/datasets/annotation_sets/tuning_train.json"),
        audio_dir=Path("../paper/data/datasets/"),
    )
    return


@app.cell
def _(Path, data, io, val_dataset):
    io.save(
        data.AnnotationSet(
            name="batdetect2_tuning_val",
            description="Set of annotations used as the validation dataset for the hyper-parameter tuning stage.",
            clip_annotations=val_dataset,
        ),
        Path("../paper/data/datasets/annotation_sets/tuning_val.json"),
        audio_dir=Path("../paper/data/datasets/"),
    )
    return


@app.cell
def _(load_dataset, load_dataset_config):
    config = load_dataset_config("../paper/conf/datasets/train/uk_tune.yaml")
    rec = load_dataset(config, base_dir="../paper/")
    return (rec,)


@app.cell
def _(rec):
    dict(rec[0].sound_events[0].tags[0].term)
    return


@app.cell
def _(compute_class_summary, rec, targets):
    compute_class_summary(rec,targets)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
