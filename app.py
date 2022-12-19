import gradio as gr
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import bat_detect.utils.detector_utils as du
import bat_detect.utils.audio_utils as au
import bat_detect.utils.plot_utils as viz


# setup the arguments
args = {}
args = du.get_default_bd_args()
args['detection_threshold'] = 0.3
args['time_expansion_factor'] = 1
args['model_path'] = 'models/Net2DFast_UK_same.pth.tar'

# load the model
model, params = du.load_model(args['model_path'])


df =  gr.Dataframe(
        headers=["species", "time_in_file", "species_prob"],
        datatype=["str", "str", "str"],
        row_count=1,
        col_count=(3, "fixed"),
    )
    

examples = [['example_data/audio/20170701_213954-MYOMYS-LR_0_0.5.wav', 0.3],
            ['example_data/audio/20180530_213516-EPTSER-LR_0_0.5.wav', 0.3],
            ['example_data/audio/20180627_215323-RHIFER-LR_0_0.5.wav', 0.3]]


def make_prediction(file_name=None, detection_threshold=0.3):
    
    if file_name is not None:
        audio_file = file_name
    else:
        return "You must provide an input audio file."
    
    if detection_threshold != '':
        args['detection_threshold'] = float(detection_threshold)
    
    results = du.process_file(audio_file, model, params, args, max_duration=5.0)
    
    clss = [aa['class'] for aa in results['pred_dict']['annotation']]
    st_time = [aa['start_time'] for aa in results['pred_dict']['annotation']]
    cls_prob = [aa['class_prob'] for aa in results['pred_dict']['annotation']]

    data = {'species': clss, 'time_in_file': st_time, 'species_prob': cls_prob}
    df = pd.DataFrame(data=data)

    return df


descr_txt = "Demo of BatDetect2 deep learning-based bat echolocation call detection. " \
          "<br>This model is only trained on bat species from the UK. If the input " \
          "file is longer than 5 seconds, only the first 5 seconds will be processed." \
          "<br>Check out the paper [here](https://www.biorxiv.org/content/10.1101/2022.12.14.520490v1)."

gr.Interface(
    fn = make_prediction,
    inputs = [gr.Audio(source="upload", type="filepath", optional=True), 
              gr.Dropdown([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])],
    outputs = df,
    theme = "huggingface",
    title = "BatDetect2 Demo",
    description = descr_txt,
    examples = examples,
    allow_flagging = 'never',
).launch()

    
