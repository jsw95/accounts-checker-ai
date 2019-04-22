from skimage import io
from skimage import transform
import matplotlib.pyplot as plt
from skimage.filters import sobel, threshold_otsu
import matplotlib.patches as mpatches
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
import numpy as np
import os

image_width = 36
image_height = 36


def split_characters(box_img):

    sobel_img = sobel(box_img)
    thresh = threshold_otsu(sobel_img)
    bw = closing(sobel_img > (1.2 * thresh), square(1))  # Unstable

    cleared = clear_border(bw)
    label_image = label(cleared)

    chars = []

    for idx, region in enumerate(regionprops(label_image)):

        if region.area >= 100:
            minr, minc, maxr, maxc = region.bbox
            char = box_img[minr:maxr, minc:maxc]

            scaled_char = transform.resize(char, (image_height, image_width))

            io.imsave('../images/chars/char' + str(idx) + '.jpg', scaled_char)
            chars.append(scaled_char)

    return chars


def generate_features_from_character_set():

    for file in os.listdir("../images/chars/"):

        char = io.imread("../images/chars/" + file, as_gray=True)

        char_fv = np.zeros((1, (image_height * image_width)), dtype='float64')

        scaled_char = transform.resize(char, (image_height, image_width))

        char_x = scaled_char.reshape((1, (image_height * image_width)))

        char_fv = np.vstack((char_fv, char_x))
        char_fv = np.delete(char_fv, 0, 0)

        with open('../data/chars/' + file[:-4] + '.txt', 'a') as f_out:
            np.savetxt(f_out, char_fv, fmt='%.6f')
