import os
import numpy as np
from skimage import transform, io
from skimage.filters import sobel, threshold_otsu
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
