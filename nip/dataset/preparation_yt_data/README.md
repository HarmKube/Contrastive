
# Preparation of the pretraining dataset

The following files were executed in place in the datadirectory on the hpc server.

The scripts were executed in the following order:

1. convert_xml_to_yolo.py: creates a new folder with the yolo labels for the pretrain dataset
2. prepare_dataset.py: 
    - creates "dataset.csv", a csv similar to the GERALD csv containing image and annotation path based on the folders with the data for further processing
    - creates "prepared_dataset.csv", a csv with additional columns about objects and similarity groups
    - output.log contains the prints statements from the prepare_dataset.py
3. delete_files.py: deletes similar group and duplicate images
4. datasplit.py: sorts the images and labels into train and val folders ready for pretraining



