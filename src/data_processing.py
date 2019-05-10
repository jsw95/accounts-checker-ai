import os
import string
import torch
import numpy as np
from skimage import transform, io
from skimage.filters import sobel, threshold_otsu, threshold_mean
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.segmentation import clear_border

image_width = 36
image_height = 36


def resize_img(img, img_size):
    (ht, wt) = img_size

    (h, w) = img.shape
    fx = w / wt
    fy = h / ht
    f = max(fx, fy)
    new_size = (max(min(ht, int(h / f)), 1),
                max(min(wt, int(w / f)), 1))  # scale according to f
    new_img = transform.resize(img, new_size)

    # filling blank portions
    target = np.ones([ht, wt])
    target[0:new_size[0], 0:new_size[1]] = new_img

    return target


def crop_text_in_box(box_img):
    sobel_img = sobel(box_img)
    thresh = threshold_otsu(sobel_img)
    bw = closing(sobel_img > (0.8 * thresh), square(3))  # Unstable

    cleared = clear_border(bw)
    label_image = label(cleared)

    words = []

    for idx, region in enumerate(regionprops(label_image)):

        if region.area >= 10:
            minr, minc, maxr, maxc = region.bbox
            word = box_img[minr:maxr, minc:maxc]
            words.append(word)

    return words


def find_boxes(img):
    thresh = threshold_otsu(img)

    bw = closing(img > (1.16 * thresh), square(1))  # Unstable

    cleared = clear_border(bw)

    label_image = label(cleared)

    box_locations = []

    for region in regionprops(label_image):
        # take regions with large enough areas
        if region.area >= 5000:

            minr, minc, maxr, maxc = region.bbox

            loc = region.centroid

            box_img = img[minr:maxr, minc:maxc]
            loc_with_img = (loc, box_img)
            box_locations.append(loc_with_img)

    return box_locations


def binary_threshold(img):
    thresh = threshold_mean(img)
    img = img > thresh

    return img

def transform_imgs_for_training(img):
    """Performs binary thresholding and returns a 1D numpy array of image"""

    img = transform.resize(img, (image_height, image_width))
    thresh = threshold_mean(img)
    img = img > thresh
    img = img.reshape((1, (image_height * image_width)))

    return img


# all_letters = string.ascii_letters + " .,;'\""
# n_letters = len(all_letters)
# char_dict = {}
# for idx, char in enumerate(all_chars):
#     enc = [0.] * len(all_chars)
#     enc[idx] = 1
#     char_dict[char] = enc
def letterToIndex(letter):
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

#
# def encode_word(word):
#     flatten = lambda l: [item for sublist in l for item in sublist]
#
#     word_encoded = np.array(flatten([char_dict[i] for i in word]))
#     word_encoded = torch.from_numpy(word_encoded)
#     print(word_encoded.size())
#
#     return word_encoded


# a = lineToTensor("t")
# print(a.shape)