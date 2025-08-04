import marimo

__generated_with = "0.14.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from batdetect2.data import load_dataset, load_dataset_config
    return load_dataset, load_dataset_config


@app.cell
def _(mo):
    dataset_input = mo.ui.file_browser(label="dataset config file")
    dataset_input
    return (dataset_input,)


@app.cell
def _(mo):
    audio_dir_input = mo.ui.file_browser(
        label="audio directory", selection_mode="directory"
    )
    audio_dir_input
    return (audio_dir_input,)


@app.cell
def _(dataset_input, load_dataset_config):
    dataset_config = load_dataset_config(
        path=dataset_input.path(), field="datasets.train",
    )
    return (dataset_config,)


@app.cell
def _(audio_dir_input, dataset_config, load_dataset):
    dataset = load_dataset(dataset_config, base_dir=audio_dir_input.path())
    return (dataset,)


@app.cell
def _(dataset):
    len(dataset)
    return


@app.cell
def _(dataset):
    tag_groups = [
        se.tags
        for clip in dataset
        for se in clip.sound_events
        if se.tags
    ]
    all_tags = [
        tag for group in tag_groups for tag in group
    ]
    return (all_tags,)


@app.cell
def _(mo):
    key_search = mo.ui.text(label="key", debounce=0.1)
    value_search = mo.ui.text(label="value", debounce=0.1)
    mo.hstack([key_search, value_search]).left()
    return key_search, value_search


@app.cell
def _(all_tags, key_search, mo, value_search):
    filtered_tags = list(set(all_tags))

    if key_search.value:
        filtered_tags = [tag for tag in filtered_tags if key_search.value.lower() in tag.key.lower()]

    if value_search.value:
        filtered_tags = [tag for tag in filtered_tags if value_search.value.lower() in tag.value.lower()]

    mo.vstack([mo.md(f"key={tag.key} value={tag.value}") for tag in filtered_tags[:5]])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
