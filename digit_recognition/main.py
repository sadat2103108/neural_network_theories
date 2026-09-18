from flask import Flask, render_template, request, jsonify
import base64
import os


import neural_network as nn
from image_utils import image_to_mnist

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(BASE_DIR, "image", "image.png")

nn.load_weights()


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json["image"]

    if "," in data:
        data = data.split(",", 1)[1]

    image_bytes = base64.b64decode(data)

    if len(image_bytes) == 0:
        return jsonify({
            "error": "Received empty image"
        }), 400

    os.makedirs(
        os.path.dirname(IMAGE_PATH),
        exist_ok=True
    )

    with open(IMAGE_PATH, "wb") as f:
        f.write(image_bytes)

    print("Saved image:", IMAGE_PATH)
    print("Image size:", len(image_bytes), "bytes")

    # -----------------------------------
    # IMAGE -> (1, 784)
    # -----------------------------------

    try:
        X = image_to_mnist(
            IMAGE_PATH,
            debug_directory=os.path.join(BASE_DIR, "debug")
        )

    except (ValueError, FileNotFoundError) as error:
        return jsonify({
            "error": str(error)
        }), 400

    print("X type:", type(X))
    print("X shape:", X.shape)
    print("X dtype:", X.dtype)
    print("X minimum:", X.min())
    print("X maximum:", X.max())

    # -----------------------------------
    # MODEL PREDICTION
    # -----------------------------------

    result = nn.predict(X)

    print("Raw prediction:", result)

    prediction = int(result[0])

    return jsonify({
        "prediction": prediction
    })


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False
    )