from flask import Flask
from flask_restful import Resource, Api
from src.endpoints import ImageToJson
from flask_restful import reqparse

# parser = reqparse.RequestParser()
# parser.add_argument('img')
# args = parser.parse_args()

app = Flask(__name__)
api = Api(app)


api.add_resource(ImageToJson, '/ImgConverter' )

if __name__ == '__main__':
    app.run(debug=True)