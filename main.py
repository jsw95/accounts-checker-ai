from skimage import io, img_as_ubyte
from src.data_processing import *
from src.utils import *
from pytesseract import image_to_string
from src.dataset_management import *
if __name__ == "__main__":

    base = "/home/jack/Workspace/data/accounts/images/templates/"
    file = "full.jpg"

    img = io.imread(base + file, as_grey=True)
    img = img_as_ubyte(img)
    # bimg = io.imread(base + '../boxes/box0.jpg', as_grey=True)

    boxes = [i for loc, i in find_boxes(img)]
    print(len(boxes))

    for i, im in enumerate(boxes):
        print(image_to_string(im))