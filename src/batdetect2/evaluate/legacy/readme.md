# Evaluating BatDetect2

> **Warning**
> This code in currently broken. Will fix soon, stay tuned.

This script evaluates a trained model and outputs several plots summarizing the performance. It is used as follows:  
`python path_to_store_images/ path_to_audio_files/ path_to_annotation_file/ path_to_trained_model/`

e.g.  
`python evaluate_models.py ../../plots/results_compare_yuc/ /data1/bat_data/data/yucatan/audio/ /data1/bat_data/annotations/anns_finetune/ ../../experiments/2021_12_17__15_58_43/2021_12_17__15_58_43.pth.tar`

By default this will just evaluate the set of test files that are already specified in the model at training time. However, you can also specify a single set of annotations to evaluate using the `--test_file` flag. These must be stored in one annotation file, containing a list of the individual files.  

e.g.    
`python evaluate_models.py ../../plots/results_compare_yuc/ /data1/bat_data/data/yucatan/audio/ /data1/bat_data/annotations/anns_finetune/ ../../experiments/2021_12_17__15_58_43/2021_12_17__15_58_43.pth.tar --test_file yucatan_TEST.json`  

You can also specify if the plots are saved as a .png or .pdf using `--file_type` and you can set title text for a plot using `--title_text`, e.g. `--file_type pdf --title_text "My Dataset Name"`




### Comparing to Tadarida-D
It is also possible to compare to Tadarida-D. For Tadarida-D the following steps are performed:  
- Matches Tadarida's detections to manually annotated calls   
- Trains a RandomForest classifier using Tadarida call features  
- Evaluate the classifier on a held out set  

Uses precomputed binaries for Tadarida-D from:  
`https://github.com/YvesBas/Tadarida-D/archive/master.zip`  

Needs to be run using the following arguments:  
`./TadaridaD -t 4 -x 1 ip_dir/`  
-t 4 means 4 threads  
-x 1 means time expansions of 1  

This will generate a folder called `txt` which contains a corresponding `.ta` file for each input audio file. Example usage is as follows:  
`python evaluate_models.py ../../plots/results_compare_yuc/ /data1/bat_data/data/yucatan/audio/ /data1/bat_data/annotations/anns_finetune/ ../../experiments/2021_12_17__15_58_43/2021_12_17__15_58_43.pth.tar --td_ip_dir /data1/bat_data/baselines/tadarida_D/`
