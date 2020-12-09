from os import path
import os
import requests
import zipfile
from config import *
import glob

TARGET_FOLDER = path.join(".",DATASET_FOLDER_IMG) 
FOLDER_LIST = []

def get_dataset_filename_map(min_val=2, max_val=-1):
    """
        Function used to return a map key-value
        Key: Name of the person
        Value: Image path list 
        This function mu
        
        Arguments: 
        min_val= minim amount of element that the Value list must contain. Default: 2
    """
    result = {}
    total_size = 0
    for folder_path in FOLDER_LIST:
        person_name = os.path.basename(os.path.normpath(folder_path))
        image_list = glob.glob(path.join(folder_path, "*"+DATASET_IMAGE_EXTENSION))
        if len(image_list) >= min_val:
            if max_val == -1:
                result[person_name] = image_list
            else:
                if total_size < max_val:
                    result[person_name] = image_list[:max_val - total_size]
                if total_size >= max_val:
                    break

            total_size += len(result[person_name])
            
    return result
        

FOLDER_LIST=glob.glob(path.join(DATASET_FOLDER_IMG, "*"))
if len(FOLDER_LIST) == 0:
    raise("place the dataset folder into the main project folder and update the config file correspondingly")
