# BatDetect2
<img align="left" width="64" height="64" src="ims/bat_icon.png">  

Code for detecting and classifying bat echolocation calls in high frequency audio recordings.


### Getting started
1) Install the Anaconda Python 3.10 distribution for your operating system from [here](https://www.continuum.io/downloads).  
2) Download this code from the repository (by clicking on the green button on top right) and unzip it.  
3) Create a new environment and install the required packages:  
`conda create -y --name batdetect2 python==3.10`  
`conda activate batdetect2`  
`conda install --file requirements.txt`  


### Try the model
Click [here](https://colab.research.google.com/github/macaodha/batdetect2/blob/master/batdetect2_notebook.ipynb) to run the model using Google Colab. You can also run this notebook locally.  


### Running the model on your own data
After following the above steps to install the code you can run the model on your own data by opening the command line where the code is located and typing:  
`python run_batdetect.py AUDIO_DIR ANN_DIR DETECTION_THRESHOLD`  
e.g.  
`python run_batdetect.py example_data/audio/ example_data/anns/ 0.3`  


`AUDIO_DIR` is the path on your computer to the audio wav files of interest.  
`ANN_DIR` is the path on your computer where the model predictions will be saved. The model will output both `.csv` and `.json` results for each audio file.   
`DETECTION_THRESHOLD` is a number between 0 and 1 specifying the cut-off threshold applied to the calls. A smaller number will result in more calls detected, but with the chance of introducing more mistakes.   

There are also optional arguments, e.g. you can request that the model outputs features (i.e. estimated call parameters) such as duration, max_frequency, etc. by setting the flag `--spec_features`. These will be saved as `*_spec_features.csv` files:  
`python run_batdetect.py example_data/audio/ example_data/anns/ 0.3 --spec_features`   

You can also specify which model to use by setting the `--model_path` argument. If not specified, it will default to using a model trained on UK data e.g.    
`python run_batdetect.py example_data/audio/ example_data/anns/ 0.3 --model_path models/Net2DFast_UK_same.pth.tar`  


### Training the model on your own data  
Take a look at the steps outlined in fintuning readme [here](bat_detect/finetune/readme.md) for a description of how to train your own model.  


### Data and annotations  
The raw audio data and annotations used to train the models in the paper will be added soon. 
The audio interface used to annotate audio data for training and evaluation is available [here](https://github.com/macaodha/batdetect2_GUI).  


### Warning  
The models developed and shared as part of this repository should be used with caution.
While they have been evaluated on held out audio data, great care should be taken when using the model outputs for any form of biodiversity assessment.
Your data may differ, and as a result it is very strongly recommended that you validate the model first using data with known species to ensure that the outputs can be trusted.


### FAQ
For more information please consult our [FAQ](faq.md).  


### Reference
If you find our work useful in your research please consider citing our paper which you can find [here](https://www.biorxiv.org/content/10.1101/2022.12.14.520490v1):
```
@article{batdetect2_2022,
    title     = {Towards a General Approach for Bat Echolocation Detection and Classification},
    author    = {Mac Aodha, Oisin and  Mart\'{i}nez Balvanera, Santiago and  Damstra, Elise and  Cooke, Martyn and  Eichinski, Philip and  Browning, Ella and  Barataudm, Michel and  Boughey, Katherine and  Coles, Roger and  Giacomini, Giada and MacSwiney G., M. Cristina and  K. Obrist, Martin and Parsons, Stuart and  Sattler, Thomas and  Jones, Kate E.},
    journal   = {bioRxiv},
    year      = {2022}
}
```

### Acknowledgements
Thanks to all the contributors who spent time collecting and annotating audio data.  


### TODOs
- [x] Release the code and pretrained model  
- [ ] Release the datasets and annotations used the experiments in the paper 
- [ ] Add the scripts used to generate the tables and figures from the paper 
