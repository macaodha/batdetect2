This directory contains some scripts for visualizing the raw data and model outputs.


`gen_spec_image.py`:  saves the model predictions on a spectrogram of the input audio file.   
e.g.  
`python gen_spec_image.py ../example_data/audio/20170701_213954-MYOMYS-LR_0_0.5.wav ../models/Net2DFast_UK_same.pth.tar`  


`gen_spec_video.py`:  generates a video showing the model predictions for a file.
e.g.   
`python gen_spec_video.py ../example_data/audio/20170701_213954-MYOMYS-LR_0_0.5.wav ../models/Net2DFast_UK_same.pth.tar`  



`gen_dataset_summary_image.py`:  generates an image displaying the mean spectrogram for each class in a specified dataset.  
e.g.  
`python gen_dataset_summary_image.py --ann_file PATH_TO_ANN/australia_TRAIN.json PATH_TO_AUDIO/audio/ ../plots/australia/`
