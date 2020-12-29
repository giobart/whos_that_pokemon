from os import path
import os
import tarfile
import config
import glob
from google_drive_downloader import GoogleDriveDownloader as gdd
import random
import zipfile

# TARGET_FOLDER = path.join(".", config.DATASET_FOLDER_IMG)
FOLDER_LIST = []

def dataset_download_targz(config=config):
    folder_list = glob.glob(path.join(config.DATASET_FOLDER_IMG, "*"))
    if len(folder_list) != 0:
        print("Dataset already downloaded")
        return
    target_filepath = path.join(".", config.DATASET_ZIP_NAME)
    # gdown.download(config.DATASET_URL, target_filepath, quiet=False)
    gdd.download_file_from_google_drive(file_id=config.DATASET_URL,
                                        dest_path=target_filepath,
                                        unzip=False)
    with tarfile.open(target_filepath) as tar:
        tar.extractall(path=config.DATASET_MAIN_FOLDER_NAME)
    os.remove(target_filepath)

def dataset_gdrive_download(config, url=None):
    folder_list = glob.glob(path.join(config.DATASET_FOLDER_IMG, "*"))
    if len(folder_list) != 0:
        print("Dataset already downloaded")
    else:
        target_filepath = path.join(".", config.DATASET_ZIP_NAME)
#         if url is not None:
#             gdown.download(url, target_filepath, quiet=False)
#         else:
#             gdown.download(config.DRIVE_URL, target_filepath, quiet=False)


        if url is not None:
            gdd.download_file_from_google_drive(file_id=url,
                                        dest_path=target_filepath,
                                        unzip=False)
        else:
            gdd.download_file_from_google_drive(file_id=config.DRIVE_URL,
                            dest_path=target_filepath,
                            unzip=False)
    

        zip_file = zipfile.ZipFile(target_filepath, "r")
        zip_file.extractall(path=config.DATASET_MAIN_FOLDER_NAME)
        zip_file.close()
        os.remove(target_filepath)

    # get the labels txt
    if hasattr(config, "LABEL_TXT_URL") and len(config.LABEL_TXT_URL) > 0:
        if os.path.exists(config.LABEL_TXT_PATH):
            print("Labels already downloaded")
            return

#         gdown.download(config.LABEL_TXT_URL, config.LABEL_TXT_PATH, quiet=False)

    gdd.download_file_from_google_drive(file_id=config.LABEL_TXT_URL,
                                        dest_path=config.LABEL_TXT_PATH,
                                        unzip=False)
    
def get_labels(config=config):
    #TODO: save images in different folders named as the labels to make accessing them faster
    labels_map = {}

    with open(config.LABEL_TXT_PATH) as file:
        data = file.readlines()

    for line in data:
        split_line = line.split()
        label = int(split_line[1])
        filepath = os.path.join(config.DATASET_FOLDER_IMG, split_line[0])

        if label in labels_map:
            labels_map[int(split_line[1])].append(filepath)
        else:
            labels_map[int(split_line[1])] = [filepath]

    return labels_map


def get_pairs(config=config):
    """
    :return: {"train": [], "validation": [], "test": []} containing a each a list [[v1,v2,v3,v4]...] of pair.txt
    """

    # download and split pairs into train, validation and test
    folder_list = glob.glob(path.join(config.DATASET_FOLDER_IMG, "*.txt"))
    if len(folder_list) == 0:
        print("downloading training pairs")
        target_filepath = path.join(".", config.PAIR_TXT_TRAIN_PATH)
#         gdown.download(config.PAIR_TXT_TRAIN_URL, target_filepath, quiet=False)

        gdd.download_file_from_google_drive(file_id=config.PAIR_TXT_TRAIN_URL,
                                        dest_path=target_filepath,
                                        unzip=False)
    
        print("downloading test pairs ")
        target_filepath = path.join(".", config.PAIR_TXT_VALID_PATH)
#         gdown.download(config.PAIR_TXT_VALID_URL, target_filepath, quiet=False)

        gdd.download_file_from_google_drive(file_id=config.PAIR_TXT_VALID_URL,
                                        dest_path=target_filepath,
                                        unzip=False)
    
        print("splitting test pairs into validation and test pairs")
        valid_file = open(config.PAIR_TXT_VALID_PATH)
        lines = valid_file.readlines()[1:]
        valid_file.close()
        random.shuffle(lines)
        valid_file = open(config.PAIR_TXT_VALID_PATH,"w")
        for item in lines[0:int(len(lines)/2)]:
            valid_file.write("%s" % item)
        valid_file.close()
        test_file = open(config.PAIR_TXT_TEST_PATH, "w+")
        for item in lines[int(len(lines) / 2):]:
            test_file.write("%s" % item)
        test_file.close()


    train_file = open(config.PAIR_TXT_TRAIN_PATH)
    valid_file = open(config.PAIR_TXT_VALID_PATH)
    test_file = open(config.PAIR_TXT_TEST_PATH)

    pairmap = {"train": [], "valid": [], "test": []}

    lines_train = train_file.readlines()[1:]
    train_file.close()
    for line in lines_train:
        pairmap["train"].append(line.split("\t"))

    lines_valid = valid_file.readlines()
    train_file.close()
    for line in lines_valid:
        pairmap["valid"].append(line.split("\t"))
        random.shuffle(pairmap["valid"])

    lines_test = test_file.readlines()
    test_file.close()
    for line in lines_test:
        pairmap["test"].append(line.split("\t"))
        random.shuffle(pairmap["test"])

    return pairmap

def get_dataset_filename_map(min_val=2, max_val=-1):
    """
        Function used to return a map key-value
        Key: Name of the person
        Value: Image path list 
        This function mu
        
        Arguments: 
        min_val= minim amount of element that the Value list must contain. Default: 2
    """
    FOLDER_LIST = glob.glob(path.join(config.DATASET_FOLDER_IMG, "*"))
    if len(FOLDER_LIST) == 0:
        raise Exception(
            "place the dataset folder into the main project folder and update the config file correspondingly or download it first with donwload_dataset()")

    result = {}
    total_size = 0
    for folder_path in FOLDER_LIST:
        person_name = os.path.basename(os.path.normpath(folder_path))
        image_list = glob.glob(path.join(folder_path, "*"+config.DATASET_IMAGE_EXTENSION))
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
