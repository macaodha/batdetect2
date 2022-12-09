# Finetuning the BatDetet2 model on your own data  
1. Annotate your data using the annotation GUI.
2. Run `prep_data_finetune.py` to create a training and validation split for your data.  
3. Run `finetune_model.py` to finetune a model on your data.  


## 1. Annotate calls of interest in audio data
This will result in a directory of audio files and a directory of annotation files, where each audio file will have a corresponding `.json` annotation file.
Use the annotation GUI to do this.
Make sure to annotation all instances of a bat call.
If unsure of the species, just label the call as `Bat`.  


## 2. Split data into train and test  
* Run `prep_data_finetune.py` to split the data into train and test sets. This will result in two separate files, a train and a test one.   
Example usage:   
`python prep_data_finetune.py dataset_name path_to_audio/audio/ path_to_annotations/anns/ path_to_output_anns/`   
This may result an error if it does not result in the files containing the same species in the train and test splits. You can try different random seeds if this is an issue e.g. `--rand_seed 123456`.  

* Can also load split from text files, where each line of the text file is the name of a .wav file e.g.  
`python prep_data_finetune.py yucatan /data1/bat_data/data/yucatan/audio/ /data1/bat_data/data/yucatan/anns/ /data1/bat_data/annotations/anns_finetune/ --train_file path_to_file/yucatan_train_split.txt --test_file path_to_file/yucatan_test_split.txt`


* Can also replace class names. Use semi colons to separate, without spaces between them e.g.
`python prep_data_finetune.py brazil_data /data1/bat_data/data/brazil_data/audio/ /data1/bat_data/data/brazil_data/anns/ /data1/bat_data/annotations/anns_finetune/ --input_class_names "Histiotus;Molossidae;Lasiurus;Myotis;Rhogeesa;Vespertilionidae" --output_class_names "Group One;Group One;Group One;Group Two;Group Two;Group Three"`  


## 3. Finetune the model  
Example usage:  
`python finetune_model.py path_to_audio/audio/ path_to_train/TRAIN.json path_to_train/TEST.json ../../models/model_to_finetune.pth.tar`


## Additional notes
* For the first step it is better to cut the files into less than 5 second audio clips and make sure to annotate them exhaustively (i.e. all bat calls should be annotated).  
* You can train the model for longer, by setting the `--num_epochs` flag to a larger number e.g. `--num_epochs 400`. The default is 200.  
* If you do not want to finetune the model, but instead want to train it from scratch you can set the `--train_from_scratch` flag.  
