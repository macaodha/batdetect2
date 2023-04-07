import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bat_detect.utils.audio_utils as au
import bat_detect.utils.detector_utils as du
import bat_detect.utils.plot_utils as viz

# setup the arguments
args = {}
args = du.get_default_run_config()
args["detection_threshold"] = 0.3
args["time_expansion_factor"] = 1
args["model_path"] = "models/Net2DFast_UK_same.pth.tar"
max_duration = 2.0

# load the model
model, params = du.load_model(args["model_path"])


df = gr.Dataframe(
    headers=["species", "time", "detection_prob", "species_prob"],
    datatype=["str", "str", "str", "str"],
    row_count=1,
    col_count=(4, "fixed"),
    label="Predictions",
)

examples = [
    ["example_data/audio/20170701_213954-MYOMYS-LR_0_0.5.wav", 0.3],
    ["example_data/audio/20180530_213516-EPTSER-LR_0_0.5.wav", 0.3],
    ["example_data/audio/20180627_215323-RHIFER-LR_0_0.5.wav", 0.3],
]


def make_prediction(file_name=None, detection_threshold=0.3):
    if file_name is not None:
        audio_file = file_name
    else:
        return "You must provide an input audio file."

    if detection_threshold is not None and detection_threshold != "":
        args["detection_threshold"] = float(detection_threshold)

    run_config = {
        **params,
        **args,
        "max_duration": max_duration,
    }

    # process the file to generate predictions
    results = du.process_file(
        audio_file,
        model,
        run_config,
    )

    anns = [ann for ann in results["pred_dict"]["annotation"]]
    clss = [aa["class"] for aa in anns]
    st_time = [aa["start_time"] for aa in anns]
    cls_prob = [aa["class_prob"] for aa in anns]
    det_prob = [aa["det_prob"] for aa in anns]
    data = {
        "species": clss,
        "time": st_time,
        "detection_prob": det_prob,
        "species_prob": cls_prob,
    }

    df = pd.DataFrame(data=data)
    im = generate_results_image(audio_file, anns)

    return [df, im]


def generate_results_image(audio_file, anns):

    # load audio
    sampling_rate, audio = au.load_audio(
        audio_file,
        args["time_expansion_factor"],
        params["target_samp_rate"],
        params["scale_raw_audio"],
        max_duration=max_duration,
    )
    duration = audio.shape[0] / sampling_rate

    # generate spec
    spec, spec_viz = au.generate_spectrogram(
        audio, sampling_rate, params, True, False
    )

    # create fig
    plt.close("all")
    fig = plt.figure(
        1,
        figsize=(spec.shape[1] / 100, spec.shape[0] / 100),
        dpi=100,
        frameon=False,
    )
    spec_duration = au.x_coords_to_time(
        spec.shape[1],
        sampling_rate,
        params["fft_win_length"],
        params["fft_overlap"],
    )
    viz.create_box_image(
        spec,
        fig,
        anns,
        0,
        spec_duration,
        spec_duration,
        params,
        spec.max() * 1.1,
        False,
        True,
    )
    plt.ylabel("Freq - kHz")
    plt.xlabel("Time - secs")
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
        gr.Audio(source="upload", type="filepath", optional=True),
        gr.Dropdown([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
    ],
    outputs=[df, gr.Image(label="Visualisation")],
    theme="huggingface",
    title="BatDetect2 Demo",
    description=descr_txt,
    examples=examples,
    allow_flagging="never",
).launch()
