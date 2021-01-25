from os import path

# DATASET_URL = ""
DATASET_MAIN_FOLDER_NAME = path.join(".", "data", "dataset")
DATASET_FOLDER_IMG = path.join(DATASET_MAIN_FOLDER_NAME, "img_align_celeba")
DATASET_IMAGE_EXTENSION = ".jpg"
DATASET_ZIP_NAME = "img_align_celeba.tgz"
DRIVE_URL = r"https://drive.google.com/uc?id=19WCblcbXjfqAQRmA7x0OtOAs6rMVvrnn"
LABEL_TXT_URL = "https://drive.google.com/uc?id=1jeEflo6J1siudrzKJGVMsJvZGoc6Pj_g"
LABEL_TXT_PATH = path.join(DATASET_FOLDER_IMG, "identity_CelebA.txt")