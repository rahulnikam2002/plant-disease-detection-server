import io
from flask import Flask, request
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
import os
import requests
from tensorflow.keras.preprocessing import image
from flask import request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
model = load_model('best_model.h5')

plant = {
    0: "Apple___Apple_scab",
    1: "Apple___Black_rot",
    2: "Apple___Cedar_apple_rust",
    3: "Apple___healthy",
    4: "Blueberry___healthy",
    5: "Cherry_(including_sour)___Powdery_mildew",
    6: "Cherry_(including_sour)___healthy",
    7: "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    8: "Corn_(maize)___Common_rust_",
    9: "Corn_(maize)___Northern_Leaf_Blight",
    10: "Corn_(maize)___healthy",
    11: "Grape___Black_rot",
    12: "Grape___Esca_(Black_Measles)",
    13: "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    14: "Grape___healthy",
    15: "Orange___Haunglongbing_(Citrus_greening)",
    16: "Peach___Bacterial_spot",
    17: "Peach___healthy",
    18: "Pepper,_bell___Bacterial_spot",
    19: "Pepper,_bell___healthy",
    20: "Potato___Early_blight",
    21: "Potato___Late_blight",
    22: "Potato___healthy",
    23: "Raspberry___healthy",
    24: "Soybean___healthy",
    25: "Squash___Powdery_mildew",
    26: "Strawberry___Leaf_scorch",
    27: "Strawberry___healthy",
    28: "Tomato___Bacterial_spot",
    29: "Tomato___Early_blight",
    30: "Tomato___Late_blight",
    31: "Tomato___Leaf_Mold",
    32: "Tomato___Septoria_leaf_spot",
    33: "Tomato___Spider_mites Two-spotted_spider_mite",
    34: "Tomato___Target_Spot",
    35: "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    36: "Tomato___Tomato_mosaic_virus",
    37: "Tomato___healthy",
}


@app.route('/')
def index_view():
    result = {'food': True}
    return result


def read_image(filename):
    response = requests.get(filename)
    response.raise_for_status()
    f = io.BytesIO(response.content)
    img = load_img(f, target_size=(256, 256))
    x = image.img_to_array(img)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    return x


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    data = request.get_json()
    foodName = data['food']
    if request.method == 'POST':
        imgURL = foodName
        img = read_image(imgURL)
        pred = np.argmax(model.predict(img))
        print(pred)
        # 'fruit' , 'prob' . 'user_image' these names we have seen in predict.html.
        result = {'food': plant[pred], 'img': imgURL}
        return result
    else:
        return "Unable to read the file. Please check file extension"


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=8000)
