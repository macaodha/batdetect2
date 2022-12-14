# BatDetect2 - FAQ

## Installation

#### Do I need to know Python to be able to use this?
No. To simply run the code on your own data you do not need any knowledge of Python. However, a small bit of familiarity with the terminal (i.e. command line) in Windows/Linux/OSX may make things easier.


#### Are there any plans for an R version?
Currently no. All the scripts export simple `.csv` files that can be read using any programming language of choice.  


#### How do I install the code?
The codebase has been tested under Windows 10, Ubuntu, and OSX. Read the instructions in the main readme to get started. If you are having problems getting it working and you feel like you have tried everything (e.g. confirming that your Anaconda Python distribution is correctly installed) feel free to open an issue on GitHub.


## Performance

#### The model does not work very well on my data?
Our model is based on a machine learning approach and as such if your data is very different from our training set it may not work as well. Feel free to use our annotation tools to label some of your own data and retrain the model. Even better, if you have large quantities of audio data with reliable species data that you are willing to share with the community please get in touch.


#### The model is incorrectly classifying insects/noise/... as bats?
Fine-tuning the model on your data can make a big difference. See previous answer.


#### The model fails to correctly detect feeding buzzes and social calls?
This is a limitation of our current training data. If you have such data or would be willing to label some for us please get in touch.  


#### Calls that are clearly belonging to the same call sequence are being predicted as coming from different species?
Currently we do not do any sophisticated post processing on the results output by the model. We return a probability associated with each species for each call. You can use these predictions to clean up the noisy predictions for sequences of calls.  


#### Can I trust the model outputs?  
The models developed and shared as part of this repository should be used with caution. While they have been evaluated on held out audio data, great care should be taken when using the model outputs for any form of biodiversity assessment. Your data may differ, and as a result it is very strongly recommended that you validate the model first using data with known species to ensure that the outputs can be trusted.


#### The code works well but it is slow?
Try a different/faster computer. On a reasonably recent desktop it takes about 13 seconds (on the GPU) or 1.3 minutes (on the CPU) to process 7.5 minutes of audio. In general, we observe a factor of ~5-10 speed up using recent Nvidia GPUs compared to CPU only systems.


#### My audio files are very big and as a result the code is slow.
If your audio files are very long in duration (i.e. multiple minutes) it might be better to split them up into several smaller files. Have a look at the instructions and scripts in our annotation GUI codebase for how to crop your files into shorter ones - see [here](https://github.com/macaodha/batdetect2_GUI). 


## Training a new model

#### Can I train a model on my own bat data with different species?
Yes. You just need to provide annotations in the correct format.


#### Will this work for frequency-division or zero-crossing recordings?
No. The code assumes that we can convert the input audio into a spectrogram.


#### Will this code work for non-bat audio data e.g. insects or birds?
In principle yes, however you may need to change some of the training hyper-parameters to ignore high frequency information when you re-train. Please open an issue on GitHub if you have a specific request.



## Usage

#### Can I use the code for commercial purposes or incorporate raw source code or trained models into my commercial system?
No. This codebase is currently only for non-commercial use. See the license.  
