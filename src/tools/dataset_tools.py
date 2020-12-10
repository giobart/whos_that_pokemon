from os import path
import os
import requests
import tarfile
from config import *
import glob
import gdown
import random

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


def get_pairs():
    """
    :return: {"train": [], "validation": [], "test": []} containing a each a list [[v1,v2,v3,v4]...] of pair.txt
    """

    folder_list = glob.glob(path.join(DATASET_FOLDER_IMG, "*.txt"))
    if len(folder_list) == 0:
        print("downloading pair ")
        target_filepath = path.join(".", PAIR_TXT_TRAIN_PATH)
        gdown.download(PAIR_TXT_TRAIN_URL, target_filepath, quiet=False)
        target_filepath = path.join(".", PAIR_TXT_VALID_PATH)
        gdown.download(PAIR_TXT_VALID_URL, target_filepath, quiet=False)

    train_file = open(PAIR_TXT_TRAIN_PATH)
    valid_file = open(PAIR_TXT_VALID_PATH)

    pairmap = {"train": [], "valid": [], "test": []}

    lines_train = train_file.readlines()[1:]
    train_file.close()
    for line in lines_train:
        pairmap["train"].append(line.split("\t"))

    lines_valid = valid_file.readlines()[1:]
    train_file.close()
    for line in lines_valid:
        pairmap["valid"].append(line.split("\t"))
        random.shuffle(pairmap["valid"])

    pairmap["test"]=pairmap["valid"][0:int(len(pairmap["valid"])/2)]
    pairmap["valid"] = pairmap["valid"][int(len(pairmap["valid"]) / 2):]

    return pairmap


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
