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


#### The code works well but it is slow?
Try a different/faster computer. On a reasonably recent desktop it takes about 13 seconds (on the GPU) or 1.3 minutes (on the CPU) to process 7.5 minutes of audio. In general, we observe a factor of ~5-10 speed up using recent Nvidia GPUs compared to CPU only systems.


#### My audio files are very big and as a result the code is slow.
If your audio files are very long in duration (i.e. mulitple minutes) it might be better to split them up into several smaller files. `sox` is a command line tool that can achieve this. It's easy to install on Ubuntu (e.g. `sudo apt-get install sox`) and is also available for Windows and OSX [here](http://sox.sourceforge.net/). To split up a file into 8 second chunks:  
`sox INPUT_FILENAME.wav OUTPUT_FILENAME.wav trim 0 8 : newfile : restart`  
This will result in a bunch of individual wav files appended with a number e.g. OUTPUT_FILENAME001.wav, OUTPUT_FILENAME002.wav, ... If you have time expanded files you might want to take the time expansion factor into account when splitting the files e.g. if the files are time expanded by 10 you should multiply the chuck length by 10. So to get 8 seconds in real time you would want to split the files into 8x10 second chunks.




## Training a new model

#### Can I train a model on my own bat data with different species?
Yes. You just need to provide annotations in the correct format.


#### Will this work for frequency-division or zero-crossing recordings?
No. The code assumes that we can convert the input audio into a spectrogram.


#### Will this code work for non-bat audio data e.g. insects or birds?
In principle yes, however you may need to change some of the training hyper-parameters to ignore high frequency information when you re-train. Please open an issue on GitHub if you have a specific request.



## Usage

#### Can I use the code for commercial purposes or incorporate raw source code or trained models into my commercial system?
No. This codebase is only for non-commercial use.
