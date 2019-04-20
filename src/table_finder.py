import skimage as si
from skimage import io
from skimage import morphology
from skimage.exposure import histogram
from skimage.feature import canny
import numpy as np
from skimage.color import label2rgb

from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage.filters import sobel, sobel_h, sobel_v, scharr, prewitt



# img = io.imread("/home/jsw/Workspace/accounts/images/account_template.jpg", as_gray=True)
img = io.imread("/home/jsw/Workspace/accounts/images/easy_template.jpg", as_gray=True)

# print(img.shape)
#
#
# # hist, hist_centers = histogram(img)
#
# # fig, axes = plt.subplots(1, 2, figsize=(8, 3))
# # axes[0].imshow(img, cmap=plt.cm.gray, interpolation='nearest')
# # axes[0].axis('off')
# # axes[1].plot(hist_centers, hist, lw=2)
# # axes[1].set_title('histogram of gray values')
#
#
# edge_sobel = sobel(img)
#
#
# markers = np.zeros_like(edge_sobel)
# markers[img < 0.72] = 1
# markers[img >= 0.72] = 2
#
# # fig, ax = plt.subplots(figsize=(4, 3))
# # ax.imshow(markers, cmap=plt.cm.nipy_spectral, interpolation='nearest')
# # ax.set_title('markers')
# # ax.axis('off')
#
# segmentation = morphology.watershed(edge_sobel, markers)
#
# # fig, ax = plt.subplots(figsize=(4, 3))
# # ax.imshow(segmentation, cmap=plt.cm.gray, interpolation='nearest')
# # ax.set_title('segmentation')
# # ax.axis('off')
#
# segmentation = ndi.binary_fill_holes(segmentation - 1)
# labeled_coins, _ = ndi.label(segmentation)
# image_label_overlay = label2rgb(labeled_coins, image=img)
#
# fig, axes = plt.subplots(1, 2, figsize=(8, 3))
# axes[0].imshow(img, cmap=plt.cm.gray, interpolation='nearest')
# axes[0].contour(segmentation, [0.5], linewidths=1.2, colors='y')
# axes[1].imshow(image_label_overlay, interpolation='nearest')
#
# for a in axes:
#     a.axis('off')
#
# plt.tight_layout()
#
# plt.show()

# plt.show()

# print(img[::1])
# plt.imshow(sobel(img), cmap='gray', interpolation='nearest')
# plt.show()
# print(img)

# img = rgb2gray(img)

# edge_sobel_h = sobel_h(img)
# edge_sobel_v = sobel_v(img)

# edges1 = canny(edge_sobel)
# edges2 = canny(img, sigma=3)

# fig, ax = plt.subplots(1, 3)
#
# ax[0].imshow(edge_sobel, cmap='gray')
# # ax[1].imshow(edges1, cmap='gray')
# # ax[2].imshow(edges2, cmap='gray')
#
#
# plt.show()


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb



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
# image_label_overlay = label2rgb(label_image, image=img)

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.imshow(img, cmap='gray')
# ax[1].imshow(label_image, cmap='gray')
# ax[2].imshow(sobel(img), cmap='gray')
for region in regionprops(label_image):
    # take regions with large enough areas
    if region.area >= 5000:
        # draw rectangle around segmented coins
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=0.5)
        ax.add_patch(rect)

ax.set_axis_off()
plt.tight_layout()
plt.show()