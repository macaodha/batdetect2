import gradio as gr
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from batdetect2 import api, plot

MAX_DURATION = 2
DETECTION_THRESHOLD = 0.3


examples = [
    [
        "example_data/audio/20170701_213954-MYOMYS-LR_0_0.5.wav",
        DETECTION_THRESHOLD,
    ],
    [
        "example_data/audio/20180530_213516-EPTSER-LR_0_0.5.wav",
        DETECTION_THRESHOLD,
    ],
    [
        "example_data/audio/20180627_215323-RHIFER-LR_0_0.5.wav",
        DETECTION_THRESHOLD,
    ],
]


def make_prediction(file_name, detection_threshold=DETECTION_THRESHOLD):
    # configure the model run
    run_config = api.get_config(
        detection_threshold=detection_threshold,
        max_duration=MAX_DURATION,
    )

    # process the file to generate predictions
    results = api.process_file(file_name, config=run_config)

    # extract the detections
    detections = results["pred_dict"]["annotation"]

    # create a dataframe of the predictions
    df = pd.DataFrame(
        [
            {
                "species": pred["class"],
                "time": pred["start_time"],
                "detection_prob": pred["class_prob"],
                "species_prob": pred["class_prob"],
            }
            for pred in detections
        ]
    )
    im = generate_results_image(file_name, detections, run_config)

    return im, df


def generate_results_image(file_name, detections, config):
    audio = api.load_audio(
        file_name,
        max_duration=config["max_duration"],
        time_exp_fact=config["time_expansion"],
        target_samp_rate=config["target_samp_rate"],
    )

    spec = api.generate_spectrogram(audio, config=config)

    # create fig
    plt.close("all")
    fig = plt.figure(
        1,
        figsize=(15, 4),
        dpi=100,
        frameon=False,
    )
    ax = fig.add_subplot(111)
    plot.spectrogram_with_detections(spec, detections, ax=ax)
    plt.tight_layout()

    # convert fig to image
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    return im


descr_txt = (
    "Demo of BatDetect2 deep learning-based bat echolocation call detection. "
    "<br>This model is only trained on bat species from the UK. If the input "
    "file is longer than 2 seconds, only the first 2 seconds will be processed."
    "<br>Check out the paper [here](https://www.biorxiv.org/content/10.1101/2022.12.14.520490v1)."
)

gr.Interface(
    fn=make_prediction,
    inputs=[
        gr.Audio(
            source="upload",
            type="filepath",
            label="Audio File",
            info="Upload an audio file to be processed.",
        ),
        gr.Slider(
            minimum=0,
            maximum=1,
            value=DETECTION_THRESHOLD,
            label="Detection Threshold",
            step=0.1,
            info=(
                "All detections with a detection probability below this "
                "threshold will be ignored."
            ),
        ),
    ],
    live=True,
    outputs=[
        gr.Image(label="Visualisation"),
        gr.Dataframe(
            headers=["species", "time", "detection_prob", "species_prob"],
            datatype=["str", "number", "number", "number"],
            row_count=1,
            col_count=(4, "fixed"),
            label="Predictions",
        ),
    ],
    theme="huggingface",
    title="BatDetect2 Demo",
    description=descr_txt,
    examples=examples,
    allow_flagging="never",
).launch()
