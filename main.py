from skimage import io, img_as_ubyte
from src.data_processing import *
from src.utils import *
from pytesseract import image_to_string
from src.dataset_management import *
if __name__ == "__main__":

    base = "/home/jack/Workspace/data/accounts/images/templates/"
    file = "base.jpg"

    img = io.imread(base + file, as_grey=True)
    img = img_as_ubyte(img)


    boxes = find_boxes(img)

    box_map = {str(loc): img for loc, img in boxes}


    box_table = return_coords_table(boxes)

