# BatDetect2
<img align="left" width="64" height="64" src="ims/bat_icon.png">
Code for detecting and classifying bat echolocation calls in high frequency audio recordings.


### Getting started
1) Install the Anaconda Python 3.9 distribution for your operating system from [here](https://www.continuum.io/downloads).  
2) Download this code from the repository (by clicking on the green button on top right) and unzip it.  
3) Create a new environment and install the required packages:  
`conda create -y --name batdetect python==3.9`  
`conda activate batdetect`  
`conda install --file requirements.txt`  


### Try the model on Colab
Click [here](https://colab.research.google.com/github/macaodha/batdetect2/blob/master/batdetect2_notebook.ipynb) to run run the model using Colab.


### Running the model on your own data
After following the above steps you can run the model on your own data by opening the command line where the code is located and typing:  
`python run_batdetect.py AUDIO_DIR ANN_DIR DETECTION_THRESHOLD`  

`AUDIO_DIR` is the path on your computer to the files of interest.  
`ANN_DIR` is the path on your computer where the detailed predictions will be saved. The model will output both `.csv` and `.json` results for each audio file.   
`DETECTION_THRESHOLD` is a number between 0 and 1 specifying the cut-off threshold applied to the calls. A smaller number will result in more calls detected, but with the chance of introducing more mistakes:  
`python run_batdetect.py example_data/audio/ example_data/anns/ 0.3`  

There are also optional arguments e.g. you can request that the model outputs features (i.e. call parameters) such as duration, max_frequency, etc. by setting the flag `--spec_features`. These will be saved as `*_spec_features.csv` files:  
`python run_batdetect.py example_data/audio/ example_data/anns/ 0.3 --spec_features`   

You can also specify which model to use by setting the `--model_path` argument. If not specified, it will default to using a model trained on UK data.  


### Requirements
The code has been tested using Python3.9 with the following package versions described in `requirements.txt`.    


### FAQ
For more information please consult our [FAQ](faq.md).  


### Reference
If you find our work useful in your research please consider citing our paper:
```
@article{batdetect2_2022,
    author    = {TODO},
    title     = {TODO},
    journal   = {TODOD},
    year      = {2022}
}
```

### Acknowledgements
TODO
