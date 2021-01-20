from os import path
import os
import tarfile
import config_lfw
import glob
from src.tools import gdrive_helper
import random
import zipfile
import shutil
from collections import defaultdict
from torchvision import transforms
from src.tools.image_preprocess import FaceAlignTransform, ImageAugmentation, ToNumpy
import urllib
from tqdm import tqdm


# TARGET_FOLDER = path.join(".", config.DATASET_FOLDER_IMG)
FOLDER_LIST = []


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def _download_zip(url, zip_filename, target_dir):
    zip_path = os.path.join(target_dir, zip_filename)
    if zip_filename not in os.listdir(target_dir):
        print('\ndownloading zip file...')

        with DownloadProgressBar(unit='B', unit_scale=True,
                                 miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, zip_path, reporthook=t.update_to)
    else:
        print('Dir is not empty')

    return zip_path


def dataset_download_targz(config=config_lfw):
    folder_list = glob.glob(path.join(config.DATASET_FOLDER_IMG, "*"))
    if len(folder_list) != 0:
        print("Dataset already downloaded")
        return

    target_filepath = path.join(".")
    _download_zip(config.DATASET_URL, config.DATASET_ZIP_NAME, target_filepath)
    target_filepath = path.join(".", config.DATASET_ZIP_NAME)

    with tarfile.open(target_filepath) as tar:
        tar.extractall(path=config.DATASET_MAIN_FOLDER_NAME)
    os.remove(target_filepath)


def dataset_gdrive_download(config=config_lfw, url=None):
    folder_list = glob.glob(path.join(config.DATASET_FOLDER_IMG, "*"))
    if len(folder_list) != 0:
        print("Dataset already downloaded")
    else:
        target_filepath = path.join(".", config.DATASET_ZIP_NAME)

        if url is not None:
            gdrive_helper.download_file_from_google_drive(url, target_filepath)
        else:
            gdrive_helper.download_file_from_google_drive(config.DRIVE_URL, target_filepath)

        if target_filepath.split('.')[-1] == 'rar':
            unrar_file(target_filepath, config.DATASET_MAIN_FOLDER_NAME)
        else:
            zip_file = zipfile.ZipFile(target_filepath, "r")
            zip_file.extractall(path=config.DATASET_MAIN_FOLDER_NAME)
            zip_file.close()

        os.remove(target_filepath)

    # get the labels txt
    if hasattr(config, "LABEL_TXT_URL") and len(config.LABEL_TXT_URL) > 0:
        if os.path.exists(config.LABEL_TXT_PATH):
            print("Labels already downloaded")
            return

        gdrive_helper.download_file_from_google_drive(config.LABEL_TXT_URL, config.LABEL_TXT_PATH)

def unrar_file(src, dst):
    from unrar import rarfile
    rar = rarfile.RarFile(src)
    rar.extractall(dst)

def get_labels(config=config_lfw, in_folders=False):
    """
    Function used to return a map key-value
        Key: Name of the person
        Value: Image path list
        This function mu
    :param config: dataset configuration
    :param in_folders: whether the samples are stored dir or in sub-dir.
        Ex if true: config.DATASET_FOLDER_IMG -> Label1 -> image1
                                                        -> image2...
                                              -> Label2 -> imag1...
    :return:
    """
    labels_map = defaultdict(list)

    with open(config.LABEL_TXT_PATH) as file:
        data = file.readlines()

    for line in data:
        split_line = line.split()
        label = int(split_line[1])
        if in_folders:
            filepath = os.path.join(config.DATASET_FOLDER_IMG, str(label), split_line[0])
        else:
            filepath = os.path.join(config.DATASET_FOLDER_IMG, split_line[0])

        labels_map[label].append(filepath)

    return labels_map


def get_pairs(config=config_lfw):
    """
    :return: {"train": [], "validation": [], "test": []} containing a each a list [[v1,v2,v3,v4]...] of pair.txt
    """

    # download and split pairs into train, validation and test
    folder_list = glob.glob(path.join(config.DATASET_FOLDER_IMG, "*.txt"))
    if len(folder_list) == 0:
        print("downloading training pairs")
        target_filepath = path.join(".", config.PAIR_TXT_TRAIN_PATH)
        #         gdown.download(config.PAIR_TXT_TRAIN_URL, target_filepath, quiet=False)

        gdrive_helper.download_file_from_google_drive(config.PAIR_TXT_TRAIN_URL, target_filepath)

        print("downloading test pairs ")
        target_filepath = path.join(".", config.PAIR_TXT_VALID_PATH)
        #         gdown.download(config.PAIR_TXT_VALID_URL, target_filepath, quiet=False)

        gdrive_helper.download_file_from_google_drive(config.PAIR_TXT_VALID_URL, target_filepath)

        print("splitting test pairs into validation and test pairs")
        valid_file = open(config.PAIR_TXT_VALID_PATH)
        lines = valid_file.readlines()[1:]
        valid_file.close()
        random.shuffle(lines)
        valid_file = open(config.PAIR_TXT_VALID_PATH, "w")
        for item in lines[0:int(len(lines) / 2)]:
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


def get_dataset_filename_map(config=config_lfw, min_val=2, max_val=-1):
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
        image_list = glob.glob(path.join(folder_path, "*" + config_lfw.DATASET_IMAGE_EXTENSION))
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


def save_images_in_folders(config=None):
    """
    saves images in config.DATASET_FOLDER_IMG in subfolders with names as their labels
    :param config:
    :return:
    """
    print('\nsaving images in folders')
    dataset_url = config.DATASET_FOLDER_IMG

    with open(config.LABEL_TXT_PATH) as file:
        data = file.readlines()

    for line in data:
        split_line = line.split()
        label = int(split_line[1])
        image_name = split_line[0]
        src_url = os.path.join(config.DATASET_FOLDER_IMG, image_name)

        if not os.path.exists(src_url):
            continue

        dst_url_dir = os.path.join(dataset_url, str(label))
        if not os.path.exists(dst_url_dir):
            os.mkdir(dst_url_dir)

        dst_url = os.path.join(dst_url_dir, image_name)

        shutil.copyfile(src_url, dst_url)
        os.remove(src_url)
    print('images to folders completed\n')


def get_list_of_indices(dataset):
    """

    :param dataset:
    :return: list of list with every index corresponding to the class labels and the sublists containing the ids
    of the images corresponding to that label.
    """
    ddict = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        ddict[label].append(idx)

    list_of_indices_for_each_class = list(ddict.values())

    return list_of_indices_for_each_class


def get_pre_transforms(input_shape):
    """get transforms before applying image augmentation"""
    return transforms.Compose([
        transforms.Resize((input_shape[1], input_shape[2])),
    ])

def get_augmentations():
    """get image augmentation transforms"""
    return transforms.Compose([
                ToNumpy(),
                ImageAugmentation.getImageAug().augment_image,
            ])

def get_transforms(input_shape, mode='train'):
    """ get finishing transforms"""
    if mode == 'train':
        return transforms.Compose([
            transforms.ToTensor(),
        ])
    elif mode == 'val' or mode == 'test':
        return transforms.Compose([
            get_pre_transforms(input_shape),
            transforms.ToTensor(),
        ])
    elif mode == 'inference':
        print('Images not Augmented', mode)
        return transforms.Compose([
            transforms.Resize((input_shape[1], input_shape[2])),
            FaceAlignTransform(FaceAlignTransform.ROTATION),
            transforms.ToTensor(),

        ])