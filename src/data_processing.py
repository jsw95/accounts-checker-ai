import string

import numpy as np
import torch
from skimage import transform
from skimage import io
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

            # io.imsave('../images/chars/char' + str(idx) + '.jpg', scaled_char)
            chars.append(scaled_char)

    return chars


def transform_imgs_for_training(img):
    """Performs binary thresholding and returns a 1D numpy array of image"""

    img = transform.resize(img, (image_height, image_width))
    thresh = threshold_mean(img)
    img = img > thresh
    img = img.reshape((1, (image_height * image_width)))

    return img


# all_letters = string.ascii_letters + " .,;'\""
# n_letters = len(all_letters)
#
#
# def letter_to_index(letter):
#     char_dict = {}
#     for idx, char in enumerate(all_letters):
#         enc = [0.] * len(all_letters)
#         enc[idx] = 1
#         char_dict[char] = enc


# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
# def word_to_tensor(line):
#     tensor = torch.zeros(len(line), 1, n_letters)
#     for li, letter in enumerate(line):
#         tensor[li][0][letter_to_index(letter)] = 1
#     return tensor


def generate_char_dict():
    all_letters = string.ascii_letters + " .,;'\""
    char_dict = {}
    for idx, char in enumerate(all_letters):
        char_dict[char] = idx
    return char_dict


def word_to_tensor(word, char_dict):
    enc = [char_dict[c] for c in word]

    return torch.Tensor(enc).to(torch.int64)


