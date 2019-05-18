import string
from src.utils import plot_image

import os
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


def transform_image_for_training(img):
    img = resize_img(img, (128, 128))
    img = binary_threshold(img)

    return img


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
    bw = closing(sobel_img > (1.0 * thresh), square(1))  # Unstable

    cleared = clear_border(bw)
    label_image = label(cleared)

    chars = []

    for idx, region in enumerate(regionprops(label_image)):

        if region.area >= 100:
            minr, minc, maxr, maxc = region.bbox
            if abs((minr - maxr) - (minc - maxc)) < 120:  # unstable check for long thin boxes
                char = box_img[minr:maxr, minc:maxc]
                # plot_image(char)
                chars.append(char)

    return chars



def generate_char_dict():
    all_letters = string.ascii_letters + " .,;'\""
    char_dict = {}
    for idx, char in enumerate(all_letters):
        char_dict[char] = idx
    return char_dict


def generate_char_wip():
    boxes_path = "/home/jack/Workspace/data/accounts/images/boxes"
    boxes = [io.imread(f"{boxes_path}/{f}") for f in os.listdir(boxes_path)[11:15]]
    char_list = []
    for box in boxes:
        chars = split_characters(box)
        for char in chars:
            char_list.append(transform_image_for_training(char))

    return char_list
