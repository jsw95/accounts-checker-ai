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

# img = sobel(img)

# apply threshold
# thresh = scharr(img)
thresh = threshold_otsu(img)
# thresh = prewitt(img)
bw = closing(img > (1.16 * thresh),  square(1))

# remove artifacts connected to image border
cleared = clear_border(bw)

# label image regions
label_image = label(cleared)

fig, ax = plt.subplots(1, 2, figsize=(10, 6))
ax[0].imshow(img, cmap='gray')

box_locations = []

for region in regionprops(label_image):
    # take regions with large enough areas
    if region.area >= 5000:
        # draw rectangle around segmented coins


        minr, minc, maxr, maxc = region.bbox
        top_left_corner = (minr, minc)
        box_locations.append(top_left_corner)

        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=0.5)
        ax[0].add_patch(rect)


minr, minc, maxr, maxc = (191, 897, 257, 1038)
minr, minc, maxr, maxc = (1240, 2, 1310, 723)

box_filtered = img[minr:maxr, minc:maxc]

io.imsave("box.jpg", box_filtered)


b = io.imread("box.jpg", as_gray=True)

ax[1].imshow(b)

ax[0].set_axis_off()
plt.tight_layout()
plt.show()