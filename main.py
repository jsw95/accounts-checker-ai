from skimage import io
from src.data_processing import *
from src.utils import *
from src.dataset_management import *
if __name__ == "__main__":

    print([f for f in os.listdir("/home/jwells/data/accounts/training/")])

    # base_data_path = "/home/jwells/data/accounts/"
    #
    # img = io.imread("/home/jwells/data/accounts/template2.jpg", as_gray=True)
    # # draw_red_boxes(img)
    #
    # boxes = find_boxes(img)
    #
    # for idx, box in enumerate(boxes):
    #     box = box[1]
    #     box = binary_threshold(box)
    #     box = resize_img(box, (125, 800))
    #     io.imsave(f"{base_data_path}training/{idx}.jpg", box)
