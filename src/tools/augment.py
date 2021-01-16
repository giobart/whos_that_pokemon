from imgaug import augmenters as iaa
import numpy as np


class ImgAugTransformation:
    ### 25% Chance to add GaussianBlur, randomly flips, rotates, adjusts brightness
    def __init__(self):
        self.aug = iaa.Sometimes(0.25, iaa.Sequential([
            iaa.GaussianBlur(sigma=(0, 1.0)),
            iaa.Fliplr(0.5),
            iaa.AddToBrightness((-30, 30)),
            iaa.MultiplyBrightness((0.5, 1.5)),
        ]))

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_images(img)

def getImageAug():
    return iaa.Sequential([
        iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 1.0))),
        iaa.Sometimes(0.25, iaa.Fliplr(0.5)),
        iaa.Sometimes(0.25, iaa.Affine(rotate=(-10, 10), mode='symmetric')),
        iaa.Sometimes(0.25, iaa.AddToBrightness((-30, 30))),
        iaa.Sometimes(0.25, iaa.MultiplyBrightness((0.5, 1.5)))
    ])

# def randomflip(image):
#
#     return image.RandomHorizontalFlip((0.5), transforms.ToTensor(),])


# class GaussianBlur(object):
#     def __init__(self, mean, std):
#         self.std = std
#         self.mean = mean
#
#     def __call__(self, img):
#         image = np.array(img)
#         image_blur = cv2.GaussianBlur(image, self.kernel_size, self.std)
#         return Image.fromarray(image_blur)


# class multiAugment(traindataset):
# adds random rotation up to 15 degrees, colorjitter, horizontal flip, and blur

# transform = transforms.Compose([
#           transforms.RandomRotation(15),
#           transforms.ColorJitter(brightness=0, contrast = 0, saturation = 0, hue = 0)
#          transforms.RandomHorizontalFlip(),
#          transforms.GaussianBlur(kernel_size, sigma=(0.1, 2.0))
#      ])
#  returns transform
