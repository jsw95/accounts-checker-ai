import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.segmentation import clear_border
from skimage.transform import hough_transform, hough_line, hough_line_peaks, probabilistic_hough_line
from matplotlib import cm
from skimage.feature import canny
from pytesseract import image_to_string

import numpy as np

def check_for_name(l):
    def check(pos):
        while l[pos] == '11':
            pos -= 1
            check(pos)

        return l[pos]

    return [check(i) for i in range(len(l))]


def return_coords_table(boxes):
    box_map = {str(loc): img for loc, img in boxes}

    sorted_coords = sorted(boxes, key=lambda x: [x[0][0], x[0][1]])

    cols_string = [image_to_string(box_map[str(i[0])]) for i in sorted_coords]
    all_cols = [cols_string[i::6][:24] for i in range(6)]

    df = pd.DataFrame(all_cols).T
    df = df[df.columns[::-1]]
    df.columns = df.iloc[0]
    df = df.drop(df.index[0])

    return df


def plot_image(img):
    plt.imshow(img, cmap='gray')
    plt.show()


def draw_red_boxes(img):
    thresh = threshold_otsu(img)
    bw = closing(img > (1.16 * thresh), square(1))

    cleared = clear_border(bw)

    label_image = label(cleared)

    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    ax[0].imshow(img, cmap='gray')

    for region in regionprops(label_image):

        if region.area >= 5000:
            minr, minc, maxr, maxc = region.bbox

            rect = Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=0.5)
            ax[0].add_patch(rect)

    ax[1].imshow(img)
    ax[0].set_axis_off()
    plt.tight_layout()
    plt.show()


def hough_lines(img):
    edges = canny(img, 2, 1, 25)
    lines = probabilistic_hough_line(edges, threshold=10, line_length=5,
                                     line_gap=3)

    # Generating figure 1

    # Generating figure 2
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(img, cmap=cm.gray)
    ax[0].set_title('Input image')

    ax[1].imshow(edges, cmap=cm.gray)
    ax[1].set_title('Canny edges')

    ax[2].imshow(edges * 0)
    for line in lines:
        p0, p1 = line
        ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
    ax[2].set_xlim((0, img.shape[1]))
    ax[2].set_ylim((img.shape[0], 0))
    ax[2].set_title('Probabilistic Hough')

    for a in ax:
        a.set_axis_off()

    plt.tight_layout()
    plt.show()
