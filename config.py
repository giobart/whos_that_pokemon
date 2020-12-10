from os import path

# Dataset
DATASET_URL = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
DATASET_MAIN_FOLDER_NAME = path.join(".", "data", "dataset")
DATASET_FOLDER_IMG = path.join(DATASET_MAIN_FOLDER_NAME, "lfw")
DATASET_IMAGE_EXTENSION = ".jpg"
DATASET_ZIP_NAME = "lfw.tgz"
DRIVE_URL = ""

# trained model data, please update .gitignore accordingly
MODEL_DATA_PATH = "model_data/"
LANDMARK_MODEL_DLIB_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
