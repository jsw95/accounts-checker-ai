from skimage import io
from skimage import transform
import matplotlib.pyplot as plt
from skimage.filters import sobel, sobel_h, sobel_v, scharr, prewitt, threshold_otsu
import matplotlib.patches as mpatches
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
import numpy as np
import os

img = io.imread("../images/boxes/box23.jpg", as_gray=True)

print(img.shape)

# plt.imshow(img, cmap='gray')
# plt.show()

image_width = 28
image_height = 28




def split_characters(box_img):

    sobel_img = sobel(box_img)

    thresh = threshold_otsu(sobel_img)

    bw = closing(sobel_img > (1.2 * thresh), square(1))  # Unstable

    # remove artifacts connected to image border
    cleared = clear_border(bw)

    label_image = label(cleared)

    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    ax[0].imshow(box_img, cmap='gray')

    chars = []

    for idx, region in enumerate(regionprops(label_image)):

        if region.area >= 100:
            minr, minc, maxr, maxc = region.bbox

            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=0.5)

            char = box_img[minr:maxr, minc:maxc]


            scaled_char = transform.resize(char, (image_height, image_width))

            io.imsave('../images/chars/char' + str(idx) + '.jpg', scaled_char)
            chars.append(scaled_char)

            ax[0].add_patch(rect)

    ax[1].imshow(scaled_char, cmap='gray')
    plt.show()

    return chars



char = io.imread("../images/chars/char25.jpg", as_gray=True)

def generate_features_from_character_set():

    for file in os.listdir("../images/chars/"):
        char = io.imread("../images/chars/" + file, as_gray=True)

        # char_label = 'w'
        char_fv = np.zeros((1, (image_height * image_width)), dtype='float64')

        scaled_char = transform.resize(char, (image_height, image_width))

        char_x = scaled_char.reshape((1, (image_height * image_width)))
        # tmp_digit_data = np.hstack((char_label, char_x[0, :]))

        char_fv = np.vstack((char_fv, char_x))
        char_fv = np.delete(char_fv, 0, 0)

        with open('../data/chars/' + file[:-4] + '.txt', 'a') as f_out:
            np.savetxt(f_out, char_fv, fmt='%.6f')



generate_features_from_character_set()



# chars = split_characters(img)

