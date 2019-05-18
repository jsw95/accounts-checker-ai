from skimage import io
from src.data_processing import *
from src.utils import *
from src.dataset_management import *
if __name__ == "__main__":

    boxes_path = "/home/jack/Workspace/data/accounts/images/boxes"
    boxes = [io.imread(f"{boxes_path}/{f}") for f in os.listdir(boxes_path)[10:20]]
    # box = boxes[0]
    # plot_image(box)
    for box in boxes:

        # plot_image(box)
        # print(box)
        # print(np.std(box))
        # print(np.max(box) - np.min(box))
        # print()
        try:
        #     # img = crop_text_in_box(box)
        #     # plot_image(img)
        #     # plot_image(box)
        #     # print(np.std(box))
            chars = split_characters(box)
            for char in chars:
                plot_image(char)
        #     c = transform_image_for_training(chars[0])
        #     plot_image(c)
        #
        except Exception as e:
            print(e)
            pass

