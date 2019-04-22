from  skimage import io
from skimage import filters
import matplotlib.pyplot as plt
import os
from skimage import transform
import numpy as np
from src.utils import *
from sklearn.model_selection import train_test_split


image_width = 36
image_height = 36


def transform_chars_for_training(char_img):
    """Performs binary thresholding and returns a 1D numpy array of image"""

    char_img = transform.resize(char_img, (image_height, image_width))
    thresh = filters.threshold_mean(char_img)
    char_img = char_img > thresh
    # plot_image(char_img)
    char_img = char_img.reshape((1, (image_height * image_width)))

    return char_img


def transform_char_set_for_training(folder_path):
    """Takes in folder and returns list of all binary imgs in that folder"""

    imgs = []

    for file in os.listdir(folder_path):
        img = io.imread(folder_path + '/' + file, as_grey=True)
        img = transform_chars_for_training(img)

        imgs.append(img)

    return imgs


def create_training_set(folder_paths):
    """Takes in list of folders of characters, return training and testing set"""

    feats, labels = False, False

    for folder in folder_paths:
        char_file = int(folder[-3:])
        imgs = transform_char_set_for_training(folder)

        X = np.array([[img] for img in imgs])
        y = np.array([[char_file]] * len(imgs))
        feats = np.concatenate((feats, X), axis=0) if feats is not False else X
        labels = np.concatenate((labels, y), axis=0) if labels is not False else y

    feats = np.reshape(feats, (feats.shape[0], feats.shape[-1]))

    return feats, labels


