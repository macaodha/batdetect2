import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from batdetect2.data import load_dataset_config, load_dataset
    return load_dataset, load_dataset_config


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
    return compute_bounds, pd


@app.cell
def _(dataset, pd):
    def get_recording_df(dataset):
        recordings = []

        for clip_annotation in dataset:
            recordings.append(
                {
                    "recording_id": clip_annotation.clip.recording.uuid,
                    "duration": clip_annotation.clip.duration,
                    "clip_annotation_id": clip_annotation.uuid,
                    "samplerate": clip_annotation.clip.recording.samplerate,
                    "path": clip_annotation.clip.recording.path.name,
                }
            )

        return pd.DataFrame(recordings)


    recordings = get_recording_df(dataset)
    recordings
    return (recordings,)


@app.cell
def _(compute_bounds, dataset, pd, targets):
    def get_sound_event_df(dataset):
        sound_events = []

        for clip_annotation in dataset:
            for sound_event in clip_annotation.sound_events:
                if not targets.filter(sound_event):
                    continue

                if sound_event.sound_event.geometry is None:
                    continue

                class_name = targets.encode_class(sound_event)

                if class_name is None:
                    continue

                start_time, low_freq, end_time, high_freq = compute_bounds(
                    sound_event.sound_event.geometry
                )

                sound_events.append(
                    {
                        "clip_annotation_id": clip_annotation.uuid,
                        "sound_event_id": sound_event.uuid,
                        "class_name": class_name,
                        "start_time": start_time,
                        "end_time": end_time,
                        "low_freq": low_freq,
                        "high_freq": high_freq,
                    }
                )

        return pd.DataFrame(sound_events)


    sound_events = get_sound_event_df(dataset)
    sound_events
    return get_sound_event_df, sound_events


@app.cell
def _(recordings, sound_events):
    def produce_summary(sound_events):
        num_calls = (
            sound_events.groupby("class_name")
            .size()
            .sort_values(ascending=False)
            .rename("num calls")
        )
        num_recs = (
            sound_events.groupby("class_name")["clip_annotation_id"]
            .nunique()
            .sort_values(ascending=False)
            .rename("num recordings")
        )
        durations = (
            sound_events.groupby("class_name")
            .apply(
                lambda group: recordings[
                    recordings["clip_annotation_id"].isin(
                        group["clip_annotation_id"]
                    )
                ]["duration"].sum(),
                include_groups=False,
            )
            .sort_values(ascending=False)
            .rename("duration")
        )
        return (
            num_calls.to_frame()
            .join(num_recs)
            .join(durations)
            .sort_values("num calls", ascending=False)
            .assign(call_rate=lambda df: df["num calls"] / df["duration"])
        )


    produce_summary(sound_events)
    return (produce_summary,)


@app.cell
def _(sound_events):
    majority_class = (
        sound_events.groupby("clip_annotation_id")
        .apply(
            lambda group: group["class_name"]
            .value_counts()
            .sort_values(ascending=False)
            .index[0],
            include_groups=False,
        )
        .rename("class_name")
        .to_frame()
        .reset_index()
    )
    return (majority_class,)


@app.cell
def _(majority_class):
    majority_class
    return


@app.cell
def _():
    from sklearn.model_selection import train_test_split
    return (train_test_split,)


@app.cell
def _(majority_class, train_test_split):
    train, val = train_test_split(
        majority_class["clip_annotation_id"],
        stratify=majority_class["class_name"],
    )
    return train, val


@app.cell
def _(dataset, train, val):
    train_dataset = [
        clip_annotation
        for clip_annotation in dataset
        if clip_annotation.uuid in set(train.values)
    ]
    val_dataset = [
        clip_annotation
        for clip_annotation in dataset
        if clip_annotation.uuid in set(val.values)
    ]
    return train_dataset, val_dataset


@app.cell
def _(get_sound_event_df, produce_summary, train_dataset):
    train_sound_events = get_sound_event_df(train_dataset)
    train_summary = produce_summary(train_sound_events)
    train_summary
    return


@app.cell
def _(get_sound_event_df, produce_summary, val_dataset):
    val_sound_events = get_sound_event_df(val_dataset)
    val_summary = produce_summary(val_sound_events)
    val_summary
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
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
