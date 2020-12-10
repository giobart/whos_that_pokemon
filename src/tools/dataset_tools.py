from os import path
import os
import requests
import tarfile
from config import *
import glob
import gdown

TARGET_FOLDER = path.join(".", DATASET_FOLDER_IMG)
FOLDER_LIST = []


def dataset_download_targz(url=DATASET_URL):
    folder_list = glob.glob(path.join(DATASET_FOLDER_IMG, "*"))
    if len(folder_list) != 0:
        print("Dataset already downloaded")
        return
    target_filepath = path.join(".", DATASET_ZIP_NAME)
    gdown.download(DATASET_URL, target_filepath, quiet=False)
    with tarfile.open(target_filepath) as tar:
        tar.extractall(path=DATASET_MAIN_FOLDER_NAME)
    os.remove(target_filepath)


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
    os.remove(target_filepath)


def get_dataset_filename_map(min_val=2):
    """
        Function used to return a map key-value
        Key: Name of the person
        Value: Image path list 
        This function mu
        
        Arguments: 
        min_val= minim amount of element that the Value list must contain. Default: 2
    """
    FOLDER_LIST = glob.glob(path.join(DATASET_FOLDER_IMG, "*"))
    if len(FOLDER_LIST) == 0:
        raise (
            "place the dataset folder into the main project folder and update the config file correspondingly or download it first with donwload_dataset()")
    result = {}
    for folder_path in FOLDER_LIST:
        person_name = os.path.basename(os.path.normpath(folder_path))
        image_list = glob.glob(path.join(folder_path, "*" + DATASET_IMAGE_EXTENSION))
        if len(image_list) >= min_val:
            result[person_name] = image_list

    return result



