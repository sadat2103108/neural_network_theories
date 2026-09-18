# Digit Recognition (Camera to MNIST)

A small **web app** that reads a handwritten digit through your camera and
predicts which digit (0-9) it is.

The model is **not trained here**. This app reuses a fully connected neural
network that I had already built and trained from scratch in my MNIST theory
notebook, and only does inference (forward propagation) with those saved
weights.

---

## Pretrained weights

The trained weights used by this app come from:

**neural_network_theories -> re_learn_from_scratch_python/THEORY_FCNN_MNIST**

https://github.com/sadat2103108/neural_network_theories/tree/main/re_learn_from_scratch_python/THEORY_FCNN_MNIST

The file `weights.npz` in this folder is a copy of the trained weights from
that notebook (`mnist_theory.ipynb`). It stores four arrays:

| Array | Meaning                    |
| ----- | -------------------------- |
| `W1`  | Hidden layer weights       |
| `b1`  | Hidden layer biases        |
| `W2`  | Output layer weights       |
| `b2`  | Output layer biases        |

`neural_network.py` only loads these arrays and runs a forward pass
(`ReLU` hidden layer, `softmax` output, then `argmax`). No training happens in
this project.

---

## How it works

1. The browser opens the camera (rear/environment camera when available) and
   shows a square guide box on top of the video.
2. When you press **Capture & Predict**, the browser crops the video to that
   guide box and sends it to the server as a base64 PNG via `POST /predict`.
3. The server saves the image and converts it into an MNIST-style input with
   `image_utils.image_to_mnist`:
   - illumination flattening to cancel shadows and uneven lighting,
   - Otsu plus adaptive thresholding to find the pen strokes,
   - repair of small broken strokes and removal of noise,
   - grouping of nearby stroke parts into one digit,
   - resize to a 20x20 region placed and centered (by center of mass) on a
     black 28x28 canvas,
   - normalization to the `0.0 - 1.0` range and flattening to shape `(1, 784)`.
4. `neural_network.predict` runs the pretrained network on that input and the
   predicted digit is returned to the page and displayed.

---

## Project structure

```
digit_recognition/
├── main.py              # Flask app and /predict endpoint
├── neural_network.py    # Loads weights.npz and runs forward propagation
├── image_utils.py       # Camera image -> MNIST-style (1, 784) input
├── weights.npz          # Pretrained weights from the theory notebook
├── templates/
│   └── index.html       # Camera UI (capture, guide box, prediction)
├── image/               # Last captured image saved by the server
└── debug/               # Optional intermediate preprocessing images
```

---

## Requirements

```
pip install flask numpy opencv-python
```

---

## Run

```bash
cd digit_recognition
python main.py
```

Then open:

```
http://127.0.0.1:5000
```

Use this exact `127.0.0.1` address so the browser allows camera access
(`getUserMedia` only works on a secure context, and localhost counts as one).

---

## Usage tips

- Write the digit **thick and dark** on **white paper**.
- Keep the digit **inside the red guide box** and fill most of it.
- Avoid strong shadows across the paper; the preprocessing removes a lot of
  uneven lighting, but good light still helps.
- If a prediction looks wrong, check the images in `debug/` to see what the
  network actually received (`05_final_mnist.png` is the final 28x28 input).
