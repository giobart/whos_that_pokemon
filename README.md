# Who's that Pokemon

## Requirements
In order to run the project the following dependencies must be resolved:
- torch
- torchvision
- pytorch_lightning
- numpy

## Download the dataset
The current dataset used is the LFW and can be download from [LFW-People](https://www.kaggle.com/atulanandjha/lfwpeople)

- The dataset can be automatically downloaded with the command `dataset_download_targz()` as shown in the `data_visualization` notebook.

## Dataset data module
The Data Module takes in input a dataset and generates training,validation and test set. 
Currently the Data Module uses an image transformation that aligns the faces using an affine transformation. 
This can be changed inside `lfw_lightning_data_module` changing the default `FaceAlignTransform(FaceAlignTransform.AFFINE)` to `FaceAlignTransform(FaceAlignTransform.ROTATION)` in order to have a simpler rotation transform. 



