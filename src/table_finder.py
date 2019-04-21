from skimage import io
import numpy as np
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage.filters import sobel, sobel_h, sobel_v, scharr, prewitt, threshold_otsu
import matplotlib.patches as mpatches
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square


# img = io.imread("/home/jsw/Workspace/accounts/images/account_template.jpg", as_gray=True)
img = io.imread("/home/jsw/Workspace/accounts/images/easy_template.jpg", as_gray=True)


def draw_red_boxes(img):

    thresh = threshold_otsu(img)
    bw = closing(img > (1.16 * thresh),  square(1))

    cleared = clear_border(bw)

    label_image = label(cleared)

    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    ax[0].imshow(img, cmap='gray')


    for region in regionprops(label_image):

        if region.area >= 5000:
            minr, minc, maxr, maxc = region.bbox

            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=0.5)
            ax[0].add_patch(rect)


    ax[1].imshow(img)

    ax[0].set_axis_off()
    plt.tight_layout()
    plt.show()


def find_boxes(img):
    # apply threshold
    thresh = threshold_otsu(img)

    bw = closing(img > (1.16 * thresh), square(1))  # Unstable

    # remove artifacts connected to image border
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


def save_box_images(box_locations):

    sorted_box_locations = sorted(box_locations, key=lambda x: [x[0][0], x[0][1]])

    for idx, box in enumerate(sorted_box_locations):
        print(idx)
        box_img = box[1]

        plt.imshow(box_img)
        io.imsave("images/box" + str(idx) + '.jpg', box_img)

        # plt.show()


# bl = find_boxes(img)
# save_box_images(bl)

