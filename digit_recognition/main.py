from flask import Flask, render_template, request, jsonify
import base64
import os

import neural_network as nn
from image_utils import image_to_mnist


app = Flask(__name__)

IMAGE_PATH = "image/image.png"


nn.load_weights()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.json["image"]

    # Remove:
    # data:image/png;base64,...
    data = data.split(",", 1)[1]

    image_bytes = base64.b64decode(data)

    os.makedirs(
        os.path.dirname(IMAGE_PATH),
        exist_ok=True
    )

    with open(IMAGE_PATH, "wb") as f:
        f.write(image_bytes)

    # Existing preprocessing pipeline
    X = image_to_mnist(IMAGE_PATH)

    # Existing neural network
    result = nn.predict(X)

    prediction = int(result[0])

    return jsonify({
        "prediction": prediction
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )