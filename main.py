from src.utils import *
from src.table_finder import find_boxes
from skimage import io


if __name__ == "__main__":

    # img = io.imread("/home/jsw/Workspace/accounts/images/account_template.jpg", as_gray=True)

    img = io.imread("/home/jsw/Workspace/accounts/images/easy_template.jpg", as_gray=True)

    boxes = find_boxes(img)

    coords_table = return_coords_table(boxes)
    coords_table.to_csv("data/image_table.csv")
    print(coords_table)