import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import bernoulli

import image_utils

# Some useful constants
DRIVING_LOG_FILE = './data/driving_log.csv'
IMG_PATH = './data/'
STEERING_COEFFICIENT = 0.229


def generate_new_image(image, steering_angle, top_crop_percent=0.35, bottom_crop_percent=0.1,
                       resize_dim=(64, 64), do_shear_prob=0.9):
    """
    Get a new randomized image base on a given image, according to the given parameters.
    """
    head = bernoulli.rvs(do_shear_prob)

    if head == 1:
        image, steering_angle = image_utils.random_shear(image, steering_angle)

    image = image_utils.crop(image, top_crop_percent, bottom_crop_percent)

    image, steering_angle = image_utils.random_flip(image, steering_angle)

    image = image_utils.random_gamma(image)

    image = image_utils.resize(image, resize_dim)

    return image, steering_angle


def get_next_image_files(batch_size=64):
    """
    The simulator records three images (namely: left, center, and right) at a given time
    However, when we are picking images for training we randomly (with equal probability)
    one of these three images and its steering angle.
    """
    data = pd.read_csv(DRIVING_LOG_FILE)
    num_of_img = len(data)
    rnd_indices = np.random.randint(0, num_of_img, batch_size)

    image_files_and_angles = []
    for index in rnd_indices:
        rnd_image = np.random.randint(0, 3)
        if rnd_image == 0:
            img = data.iloc[index]['left'].strip()
            angle = data.iloc[index]['steering'] + STEERING_COEFFICIENT
            image_files_and_angles.append((img, angle))
        elif rnd_image == 1:
            img = data.iloc[index]['center'].strip()
            angle = data.iloc[index]['steering']
            image_files_and_angles.append((img, angle))
        else:
            img = data.iloc[index]['right'].strip()
            angle = data.iloc[index]['steering'] - STEERING_COEFFICIENT
            image_files_and_angles.append((img, angle))

    return image_files_and_angles


def generate_next_batch(batch_size=64):
    """
    This generator yields the next training batch

    :param batch_size:
        Number of training images in a single batch

    :return:
        A tuple of features and steering angles as two numpy arrays
    """
    while True:
        X_batch = []
        y_batch = []
        images = get_next_image_files(batch_size)
        for img_file, angle in images:
            raw_image = plt.imread(IMG_PATH + img_file)
            raw_angle = angle
            new_image, new_angle = generate_new_image(raw_image, raw_angle)
            X_batch.append(new_image)
            y_batch.append(new_angle)

        assert len(
            X_batch) == batch_size, 'len(X_batch) == batch_size should be True'

        yield np.array(X_batch), np.array(y_batch)
