# Finetuning the BatDetet2 model on your own data  

> **Warning**
> This code in currently broken. Will fix soon, stay tuned.

Main steps:  
1. Annotate your data using the annotation GUI.
2. Run `prep_data_finetune.py` to create a training and validation split for your data.  
3. Run `finetune_model.py` to finetune a model on your data.  


## 1. Annotate calls of interest in audio data
Use the annotation tools provided [here](https://github.com/macaodha/batdetect2_GUI) to manually identify where the events of interest (e.g. bat echolocation calls) are in your files.  
This will result in a directory of audio files and a directory of annotation files, where each audio file will have a corresponding `.json` annotation file.
Make sure to annotation all instances of a bat call.
If unsure of the species, just label the call as `Bat`.  


## 2. Split data into train and test sets
After performing the previous step you should have a directory of annotations files saved as jsons, one for each audio file you have annotated. 
* The next step is to split these into training and testing subsets.  
Run `prep_data_finetune.py` to split the data into train and test sets. This will result in two separate files, a train and a test one, i.e.  
`python prep_data_finetune.py dataset_name path_to_audio/ path_to_annotations/ path_to_output_anns/`   
This may result an error if it does not generate output files containing the same set of species in the train and test splits. You can try different random seeds if this is an issue e.g. `--rand_seed 123456`.  

* You can also load the train and test split using text files, where each line of the text file is the name of a `wav` file (without the file path) e.g.  
`python prep_data_finetune.py dataset_name path_to_audio/ path_to_annotations/ path_to_output/ --train_file path_to_file/list_of_train_files.txt --test_file path_to_file/list_of_test_files.txt`


* Can also replace class names. This can be helpful if you don't think you have enough calls/files for a given species. Use semi-colons to separate, without spaces between them e.g.  
`python prep_data_finetune.py dataset_name path_to_audio/audio/ path_to_annotations/anns/ path_to_output/ --input_class_names "Histiotus;Molossidae;Lasiurus;Myotis;Rhogeesa;Vespertilionidae" --output_class_names "Group One;Group One;Group One;Group Two;Group Two;Group Three"`  


## 3. Finetuning the model  
Finally, you can finetune the model using your data i.e.  
`python finetune_model.py path_to_audio/ path_to_train/TRAIN.json path_to_train/TEST.json ../../models/Net2DFast_UK_same.pth.tar`  
Here, `TRAIN.json` and `TEST.json` are the splits created in the previous steps.  


#### Additional notes
* For the first step it is better to cut the files into less than 5 second audio clips and make sure to annotate them exhaustively (i.e. all bat calls should be annotated).  
* You can train the model for longer, by setting the `--num_epochs` flag to a larger number e.g. `--num_epochs 400`. The default is `200`.  
* If you do not want to finetune the model, but instead want to train it from scratch, you can set the `--train_from_scratch` flag.  
