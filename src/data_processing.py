from  skimage import io
from skimage import filters
import matplotlib.pyplot as plt
import os
from skimage import transform
import numpy as np
from src.utils import *
from sklearn.model_selection import train_test_split


image_width = 28
image_height = 28

b_file_path = '/home/jsw/Workspace/accounts/data/English/Img/GoodImg/Bmp/Sample038/'
e_file_path = '/home/jsw/Workspace/accounts/data/English/Img/GoodImg/Bmp/Sample041/'
o_file_path = '/home/jsw/Workspace/accounts/data/English/Img/GoodImg/Bmp/Sample051/'
r_file_path = '/home/jsw/Workspace/accounts/data/English/Img/GoodImg/Bmp/Sample054/'


def transform_chars_for_training(char_img):
    """Performs binary thresholding and returns a 1D numpy array of image"""

    char_img = transform.resize(char_img, (image_height, image_width))
    thresh = filters.threshold_mean(char_img)
    char_img = char_img > thresh
    # plot_image(char_img)
    char_img = char_img.reshape((1, (image_height * image_width)))

    return char_img


def transform_char_set_for_training(folder_path):

    imgs = []

    for file in os.listdir(folder_path):
        img = io.imread(folder_path + file, as_grey=True)
        img = transform_chars_for_training(img)

        imgs.append(img)

    return imgs


e_imgs = transform_char_set_for_training(e_file_path)
b_imgs = transform_char_set_for_training(b_file_path)
o_imgs = transform_char_set_for_training(o_file_path)
r_imgs = transform_char_set_for_training(r_file_path)


def create_training_set():
    """Takes in transformed 1D numpy array of character and labels"""

    b_X = np.array([[b] for b in b_imgs])
    e_X = np.array([[e] for e in e_imgs])
    o_X = np.array([[o] for o in o_imgs])
    r_X = np.array([[r] for r in r_imgs])

    b_y = np.array([[0]] * len(b_imgs))
    e_y = np.array([[1]] * len(e_imgs))
    o_y = np.array([[2]] * len(o_imgs))
    r_y = np.array([[3]] * len(r_imgs))

    feats = np.concatenate((b_X, e_X, o_X, r_X), axis=0)
    labels = np.concatenate((b_y, e_y, o_y, r_y), axis=0)

    feats = np.reshape(feats, (feats.shape[0], feats.shape[-1]))

    X_train, X_test, y_train, y_test = train_test_split(
        feats, labels, test_size=0.33, random_state=42, stratify=labels)

    return X_train, X_test, y_train, y_test

