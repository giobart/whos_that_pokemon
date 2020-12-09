import dlib
import cv2
import os
from config import MODEL_DATA_PATH, LANDMARK_MODEL_DLIB_URL
import gdown
import numpy as np
from PIL import Image
import bz2
import math
import scipy.misc

LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]
CHIN_INDEX = 9
ALPHA_SHIFT = 10
LEFT_EYE_POS = lambda w, h: (w / 3 + ALPHA_SHIFT, h / 3 + ALPHA_SHIFT)
RIGHT_EYE_POS = lambda w, h: ((2 * w / 3) - ALPHA_SHIFT, h / 3 + ALPHA_SHIFT)
CHIN_POS = lambda w, h: ((w / 2) + ALPHA_SHIFT, (3 * h / 4) + ALPHA_SHIFT)


def extract_eye_center(shape, eye_indices):
    points = list(map(lambda i: shape.part(i), eye_indices))
    xs = map(lambda p: p.x, points)
    ys = map(lambda p: p.y, points)
    return sum(xs) // 6, sum(ys) // 6


def angle_between_2_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    tan = (y2 - y1) / (x2 - x1)
    return np.degrees(np.arctan(tan))


def extract_chin(shape, CHIN_INDEX):
    part = shape.part(CHIN_INDEX)
    return part.x, part.y


def get_rotation_matrix(p1, p2):
    angle = angle_between_2_points(p1, p2)
    x1, y1 = p1
    x2, y2 = p2
    xc = (x1 + x2) // 2
    yc = (y1 + y2) // 2
    M = cv2.getRotationMatrix2D((xc, yc), angle, 1)
    return M


def get_affine_transform_matrix(p1, p2, p3, d1, d2, d3):
    pts1 = np.float32([list(p1), list(p2), list(p3)])
    pts2 = np.float32([list(d1), list(d2), list(d3)])

    return cv2.getAffineTransform(pts1, pts2)


def get_translation_matrix(p1, w, h):
    x1, y1 = p1
    tx = (w / 3) - x1
    ty = (h / 3) - y1
    if abs(tx) > w / 3:
        # too big translation, lets avoid it
        tx = 0
        ty = 0
    return np.float32([[1, 0, tx], [0, 1, ty]])


def get_scaling_factor(p1, p2, w):
    x1, y1 = p1
    x2, y2 = p2
    d_x1 = math.sqrt(abs(x1 - x2) ** 2 + abs(y1 - y2) ** 2)
    d_final = (2 * w / 3) - w / 3
    return d_final / d_x1


def image_deep_alignment(img, transform_kind):
    # convert image to np array
    img = np.array(img)

    # initialize detector
    face_detector, shape_predictor = get_detectors()

    # get face shape
    detections = face_detector(img, 1)

    transformed = img
    if len(detections) > 0:
        detected_face = detections[0]
        img_shape = shape_predictor(img, detected_face)

        left_eye = extract_eye_center(img_shape, LEFT_EYE_INDICES)
        right_eye = extract_eye_center(img_shape, RIGHT_EYE_INDICES)
        chin = extract_chin(img_shape, CHIN_INDEX)
        w = img.shape[0]
        h = img.shape[1]

        if transform_kind == FaceAlignTransform.ROTATION:
            rotation = get_rotation_matrix(left_eye, right_eye)
            translation = get_translation_matrix(left_eye, w, h)
            translated = cv2.warpAffine(img, translation, img.shape[:2], flags=cv2.INTER_CUBIC)
            transformed = cv2.warpAffine(translated, rotation, img.shape[:2], flags=cv2.INTER_CUBIC)
        elif transform_kind == FaceAlignTransform.AFFINE:
            matrix = get_affine_transform_matrix(
                left_eye, right_eye, chin,
                LEFT_EYE_POS(w, h), RIGHT_EYE_POS(w, h), CHIN_POS(w, h)
            )
            transformed = cv2.warpAffine(img, matrix, img.shape[:2], flags=cv2.INTER_CUBIC)

    return Image.fromarray(transformed)


def get_detectors():
    # initialize and return detector
    face_detector = dlib.get_frontal_face_detector()

    # check required file exists in the home/.deepface/weights folder
    if not os.path.isfile(os.path.join(".", MODEL_DATA_PATH, "dlib", "shape_predictor_68_face_landmarks.dat")):
        print("Shape prediction model missing, downloading it now!")

        url = LANDMARK_MODEL_DLIB_URL
        output_dir = os.path.join(".", MODEL_DATA_PATH, "dlib")
        output = os.path.join(output_dir, url.split("/")[-1])

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        gdown.download(url, output, quiet=False)

        zipfile = bz2.BZ2File(output)
        data = zipfile.read()
        newfilepath = output[:-4]  # discard .bz2 extension
        open(newfilepath, 'wb').write(data)

    shape_predictor = dlib.shape_predictor(
        os.path.join(".", MODEL_DATA_PATH, "dlib", "shape_predictor_68_face_landmarks.dat"))

    return face_detector, shape_predictor


class FaceAlignTransform(object):
    """
    Align the face
    """

    ROTATION = "r"
    AFFINE = "a"

    def __init__(self, kind=AFFINE):
        self.transform_kind = kind

    def __call__(self, img):
        return image_deep_alignment(img, self.transform_kind)
