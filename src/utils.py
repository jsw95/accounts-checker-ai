import matplotlib.pyplot as plt
import pandas as pd
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.segmentation import clear_border


def check_for_name(l):
    def check(pos):
        while l[pos] == '11':
            pos -= 1
            check(pos)

        return l[pos]

    return [check(i) for i in range(len(l))]


def return_coords_table(coords):
    sorted_coords = sorted(coords, key=lambda x: [x[0][0], x[0][1]])

    all_cols = [sorted_coords[i::6][:24] for i in range(6)]

    data = {str(i): all_cols[i] for i in range(6)}

    df = pd.DataFrame(all_cols).T
    df = df[df.columns[::-1]]
    df.columns = ['Col1', 'Col2', 'Col3', 'Col4', 'Col5', 'Col6']

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

            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=0.5)
            ax[0].add_patch(rect)

    ax[1].imshow(img)
    ax[0].set_axis_off()
    plt.tight_layout()
    plt.show()
