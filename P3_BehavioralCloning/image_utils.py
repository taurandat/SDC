import cv2
import numpy as np

import scipy.misc
from scipy.ndimage import rotate
from scipy.stats import bernoulli


def crop(image, top_percent, bottom_percent):
    """
    Crops an image according to the given parameters
    """
    assert 0 <= top_percent < 0.5, 'top_percent should be between 0.0 and 0.5'
    assert 0 <= bottom_percent < 0.5, 'top_percent should be between 0.0 and 0.5'

    top = int(np.ceil(image.shape[0] * top_percent))
    bottom = image.shape[0] - int(np.ceil(image.shape[0] * bottom_percent))

    return image[top:bottom, :]


def resize(image, new_dim):
    """
    Resize a given image according the the new dimension
    """
    return scipy.misc.imresize(image, new_dim)


def random_flip(image, steering_angle, flipping_prob=0.5):
    """
    Based on the outcome of an coin flip, the image will be flipped.
    If the flipping is applied, the steering angle will be negated.
    """
    head = bernoulli.rvs(flipping_prob)
    if head:
        return np.fliplr(image), -1 * steering_angle
    else:
        return image, steering_angle


def random_gamma(image):
    """
    An alternative method for changing the brightness of training images.
    Source: http://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
    """
    gamma = np.random.uniform(0.4, 1.5)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def random_shear(image, steering_angle, shear_range=200):
    """
    Shear a given image according to the given parameters.
    """
    rows, cols, ch = image.shape
    dx = np.random.randint(-shear_range, shear_range + 1)
    random_point = [cols / 2 + dx, rows / 2]
    pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
    pts2 = np.float32([[0, rows], [cols, rows], random_point])
    dsteering = dx / (rows / 2) * 360 / (2 * np.pi * 25.0) / 6.0
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows), borderMode=1)
    steering_angle += dsteering

    return image, steering_angle


def random_rotation(image, steering_angle, rotation_amount=15):
    """
    Rotate a given image according to the given parameters.
    """
    angle = np.random.uniform(-rotation_amount, rotation_amount + 1)
    rad = (np.pi / 180.0) * angle
    return rotate(image, angle, reshape=False), steering_angle + (-1) * rad


def min_max(data, a=-0.5, b=0.5):
    """
    Normalize the data base on the min-max method.
    """
    data_max = np.max(data)
    data_min = np.min(data)

    return a + (b - a) * ((data - data_min) / (data_max - data_min))
