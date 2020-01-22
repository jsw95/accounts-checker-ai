from skimage import io, img_as_ubyte
from src.data_processing import *
from src.utils import *
from pytesseract import image_to_string
from src.dataset_management import *

if __name__ == "__main__":

    base = "/home/jack/Workspace/accounts-checker-ai/data/"
    file = "img1.jpg"

    img = io.imread(base + file, as_gray=True)
    img = img_as_ubyte(img)

    boxes = find_boxes(img)
    box_table = return_coords_table(boxes)

def image_to_json(img):
    img = img_as_ubyte(img)

    boxes = find_boxes(img)
    box_table = return_coords_table(boxes)

    return box_table.to_json()

    # box_map = {str(loc): img for loc, img in boxes}
    #
    # boxes = [i for loc, i in find_boxes(img)]
    # print(len(boxes))

    # for i, im in enumerate(boxes):
    #     print(image_to_string(im))
    #
    #
    # box_table = return_coords_table(boxes)

