import dblib
import os
from config import MODEL_DATA_PATH, LANDMARK_MODEL_DLIB_URL
import gdown
import numpy as np
from PIL import Image


def image_deep_alignment(img):
    # convert image to np array
    img = np.array(img)

    # initialize detector
    face_detector, shape_predictor = get_detectors()

    # get face shape
    detections = face_detector(img, 1)

    if len(detections) > 0:
        detected_face = detections[0]
        img_shape = shape_predictor(img, detected_face)
        img = dlib.get_face_chip(img, img_shape, size=img.shape[0])

    return Image.fromarray(img)


def get_detectors():
    # initialize and return detector
    face_detector = dlib.get_frontal_face_detector()

    # check required file exists in the home/.deepface/weights folder
    if os.path.isfile(os.path.join(".", MODEL_DATA_PATH)) != True:
        print("Shape prediction model missing, downloading it now!")

        url = LANDMARK_MODEL_DLIB_URL
        output = os.path.join(".", MODEL_DATA_PATH, "dlib", url.split("/")[-1])

        gdown.download(url, output, quiet=False)

        zipfile = bz2.BZ2File(output)
        data = zipfile.read()
        newfilepath = output[:-4]  # discard .bz2 extension
        open(newfilepath, 'wb').write(data)

    shape_predictor = dlib.shape_predictor(
        os.path.join(".", MODEL_DATA_PATH, "dlib", "shape_predictor_5_face_landmarks.dat"))

    return face_detector, shape_predictor
