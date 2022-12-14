## How to train a model from scratch
`python train_model.py data_dir annotation_dir` e.g.   
`python train_model.py /data1/bat_data/data/ /data1/bat_data/annotations/anns/` 

More comprehensive instructions are provided in the finetune directory.  


## Training on your own data
You can either use the finetuning scripts to finetune from an existing training dataset. Follow the instructions in the `../finetune/` directory.  

Alternatively, you can train from scratch. First, you will need to create your own annotation file (like in the finetune example), and then you will need to edit `train_split.py` to add your new dataset and specify which combination of files you want to train on.  

Note, if training from scratch and you want to include the existing data, you may need to set all the class names to the generic class name ('Bat') so that the existing species are not added to your model, but instead just used to help perform the bat/not bat task.   

## Additional notes
Having blank files with no bats in them is also useful, just make sure that the annotation files lists them as not being annotated (i.e. `is_annotated=True`). 

Training will be slow without a GPU.  
