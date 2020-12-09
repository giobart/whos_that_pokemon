from os import path
import os
import requests
import tarfile
from config import *
import glob
import gdown

TARGET_FOLDER = path.join(".",DATASET_FOLDER_IMG) 
FOLDER_LIST = []

def dataset_download_targz(url=DATASET_URL):
    folder_list = glob.glob(path.join(DATASET_FOLDER_IMG, "*"))
    if len(folder_list) != 0:
        print("Dataset already downloaded")
        return
    r = requests.get(DATASET_URL, allow_redirects=True)
    target_filepath = path.join(".", DATASET_ZIP_NAME)
    print(target_filepath)
    open(target_filepath, 'wb').write(r.content)
    with tarfile.open(target_filepath) as tar:
        tar.extractall(path=DATASET_MAIN_FOLDER_NAME)

def dataset_gdrive_download(url=DRIVE_URL):
    folder_list = glob.glob(path.join(DATASET_FOLDER_IMG, "*"))
    if len(folder_list) != 0:
        print("Dataset already downloaded")
        return
    target_filepath = path.join(".", DATASET_ZIP_NAME)
    gdown.download(url, target_filepath, quiet=False)
    zip_file = zipfile.ZipFile(target_filepath, "r")
    zip_file.extractall(path=DATASET_MAIN_FOLDER_NAME)
    zip_file.close()

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

