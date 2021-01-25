import cv2
from facenet_pytorch import MTCNN
import numpy as np
from PIL import Image
import math
from imgaug import augmenters as iaa
import torch

ALPHA_SHIFT = 10
LEFT_EYE_POS = lambda w, h: (w / 3 + ALPHA_SHIFT, h / 3 + ALPHA_SHIFT)
RIGHT_EYE_POS = lambda w, h: ((2 * w / 3) - ALPHA_SHIFT, h / 3 + ALPHA_SHIFT)
NOSE_POS = lambda w, h: ((w / 2) + ALPHA_SHIFT, (3 * h / 5) + ALPHA_SHIFT)


def angle_between_2_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    tan = (y2 - y1) / (x2 - x1)
    return np.degrees(np.arctan(tan))


def get_rotation_matrix(p1, p2):
    angle = angle_between_2_points(p1, p2)
    x1, y1 = p1
    x2, y2 = p2
    xc = (x1 + x2) // 2
    yc = (y1 + y2) // 2
    M = cv2.getRotationMatrix2D((xc, yc), angle, 1)
    return M


def image_deep_alignment(img, transform_kind="crop", precomputed_detection=None, precomputed_landmarks=None,
                         compute_landmarks=True):
    # convert image to np array
    img = np.array(img)

    detections = None
    landmarks = None

    # compute bounding box and landmarks
    if precomputed_detection is None or precomputed_landmarks is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # initialize detector
        face_detector = MTCNN(device=device)
        face_detector.select_largest = True
        detections, probs, landmarks = None, None, None
        # detect landmark points
        if not compute_landmarks:
            detections, probs = face_detector.detect(img, landmarks=False)
        else:
            detections, probs, landmarks = face_detector.detect(img, landmarks=True)
    else:
        detections = precomputed_detection
        landmarks = precomputed_landmarks

    transformed = img

    if detections is not None:

        x, y, x2, y2 = int(detections[0][0]), int(detections[0][1]), int(detections[0][2]), int(detections[0][3])
        h = img.shape[0]
        w = img.shape[1]

        # rotation transformation
        if transform_kind == FaceAlignTransform.ROTATION:
            left_eye = landmarks[0][0]
            right_eye = landmarks[0][1]
            nose = landmarks[0][4]
            rotation = get_rotation_matrix(left_eye, right_eye)
            transformed = cv2.warpAffine(img, rotation, img.shape[:2], flags=cv2.INTER_CUBIC)

        # crop the bounding boxes and expand the box by a factor of 1/3
        elif transform_kind == "crop":
            y = y - int((y2 - y) * 1 / 3)
            if y < 0:
                y = 0
            y2 = y2 + int((y2 - y) * 1 / 3)
            if y2 > h:
                y2 = h - 1
            x = x - int((x2 - x) * 1 / 3)
            if x < 0:
                x = 0
            x2 = x2 + int((x2 - x) * 1 / 3)
            if x2 > w:
                x2 = w - 1
            return Image.fromarray(img[y:y2, x:x2, :]), detections, landmarks

    return Image.fromarray(transformed), detections, landmarks


class FaceAlignTransform(object):
    """
    Align the face by crop only (SIMPLE kind) or crop and rotation (ROTATION kind)
    """

    ROTATION = "r"  # crop and rotation
    SIMPLE = "a"    # crop only

    def __init__(self, shape, kind="a"):
        self.shape = shape
        self.kind = kind

    def __call__(self, img):
        return self.crop_and_resize(img)

    def crop_and_resize(self, img):
        detections = None
        landmarks = None
        if self.kind is FaceAlignTransform.SIMPLE:
            img, _, _ = image_deep_alignment(img, compute_landmarks=False)
        else:
            img, detections, landmarks = image_deep_alignment(img)

        # create a square image and center the cropped face
        old_size = img.size
        ratio = float(self.shape) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        img = img.resize(new_size, Image.ANTIALIAS)
        new_im = Image.new("RGB", (self.shape, self.shape))
        new_im.paste(img, ((self.shape - new_size[0]) // 2,
                           (self.shape - new_size[1]) // 2))
        img = new_im

        # rotate the image
        if self.kind is FaceAlignTransform.ROTATION:
            img, _, _ = image_deep_alignment(img, transform_kind="r", precomputed_detection=detections,
                                             precomputed_landmarks=landmarks)
        return img


class ToNumpy(object):
    """
    Align the face
    """

    def __init__(self):
        pass

    def __call__(self, img):
        return np.array(img)


class ImageAugmentation:
    @staticmethod
    def getImageAug():
        return iaa.Sequential([
            iaa.GaussianBlur(sigma=(0, 1.0)),
            iaa.Fliplr(0.5),
            iaa.AddToBrightness((-30, 30)),
        ])
