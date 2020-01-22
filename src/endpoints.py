from flask_restful import Resource, request

from src.data_processing import *
from src.dataset_management import *
from src.utils import *


class ImageToJson(Resource):
    def post(self):
        data = request.json
        img = data['img']
        img = img_as_ubyte(img)

        boxes = find_boxes(img)
        box_table = return_coords_table(boxes)

        return box_table.to_json()
