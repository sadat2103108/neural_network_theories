# Neural Networks From Scratch

A learning-by-building project: reimplementing neural networks from first principles,
**without TensorFlow or PyTorch** — no autograd, no `fit()`, no hidden magic. Every forward
pass, every derivative and every weight update is written by hand.

The project follows one person's actual learning path: start with a single artificial neuron,
build a network as a real graph of connected neurons in C++, then relearn everything in Python
(vectorized perceptrons), and finally arrive at the matrix formulation of a fully connected
network trained on **MNIST** — with the theory written out notebook-by-notebook along the way.

---

## The Learning Path

| Phase | What was explored | Where |
|-------|-------------------|-------|
| 1 | A single **perceptron**, taught logical **OR / AND** (and shown to fail on XOR) | `re_learn_from_scratch_python/perceptron/1_perceptron_learns_OR.py` |
| 2 | A tiny **MLP (2 hidden neurons)**, taught logical **XOR** — and why it sometimes fails | `re_learn_from_scratch_python/perceptron/2_perceptrons_learns_XOR.py` |
| 3 | The same XOR task **fixed**: `tanh` + BCE + 3 hidden neurons | `re_learn_from_scratch_python/perceptron/3_fixed_perceptron_learns_XOR.py` |
| 4 | The **graph-of-neurons** network in C++, generalized to any layer shape, trained on MNIST | `from_scratch_cpp/neural_network_scratch_raw.cpp`, `.../neural_network_scratch.cpp` |
| 5 | The **matrix formulation** of a fully connected network, MNIST at ~97.5% test accuracy, with full theory notes | `re_learn_from_scratch_python/THEORY_FCNN_MNIST/mnist_theory.ipynb` |
| 6 | The trained weights put to use: a **Flask web app** that recognizes camera-captured digits | `digit_recognition/` |

The same ideas are implemented twice from two different mental models:

- **C++ — object/graph model:** each `Neuron` is an independent object that owns its own
  weight vector, bias, activation, and cached `z`/`a`/`delta`. The network is a container of
  layers of these neuron objects, wired together by looping over them. Nothing is a matrix.
- **Python — matrix model:** the whole layer is three NumPy arrays (`W`, `b`) and backprop
  collapses into a handful of matrix multiplications. This is what frameworks do internally.

---

## Repository Layout

```
mnist_from_scratch/
├── README.md
├── .gitignore
├── data/                                  # MNIST IDX files (raw binary)
│   ├── train-images-idx3-ubyte            # 60,000 training images (28x28)
│   ├── train-labels-idx1-ubyte
│   ├── t10k-images-idx3-ubyte             # 10,000 test images
│   └── t10k-labels-idx1-ubyte
│
├── from_scratch_cpp/                      # Phase 4: graph-of-neurons network
│   ├── neural_network_scratch_raw.cpp     # first full draft
│   └── neural_network_scratch.cpp         # polished / generalized version
│
├── re_learn_from_scratch_python/          # Phases 1-3 + 5: relearning in Python
│   ├── perceptron/
│   │   ├── 1_perceptron_learns_OR.py
│   │   ├── 2_perceptrons_learns_XOR.py
│   │   └── 3_fixed_perceptron_learns_XOR.py
│   └── THEORY_FCNN_MNIST/
│       ├── mnist_theory.ipynb             # theory notes + full FCNN pipeline
│       └── weights.npz                    # saved trained parameters
│
└── digit_recognition/                     # Phase 6: camera web app (inference only)
│   ├── main.py                            # Flask server + /predict endpoint
│   ├── neural_network.py                  # loads weights.npz, forward pass only
│   ├── image_utils.py                     # camera image -> MNIST-style (1, 784)
│   ├── weights.npz                        # copy of the trained weights from Part 3
│   ├── templates/
│   │   └── index.html                     # camera UI (guide box, capture, prediction)
│   ├── image/                             # last captured image
│   └── debug/                             # intermediate preprocessing images
```

---

## Part 1 — C++: a Network as a Graph of Neurons

**Files:** `from_scratch_cpp/neural_network_scratch_raw.cpp`, `from_scratch_cpp/neural_network_scratch.cpp`

This was the first attempt, built on the intuition that a network is literally a graph of
individual neurons, each acting on its own.

### `class Neuron`

Each neuron is an independent unit holding everything it needs:

- `vector<double> W` — one weight per input
- `double b` — bias
- `Activation activation` — its own activation function
- `double z` — cached pre-activation (needed for the derivative)
- `double a` — cached output
- `double delta` — cached error signal for backprop

Methods: `forward(X)`, `computeOutputDelta(y)`, `computeHiddenDelta(weightedDeltaSum)`,
`updateWeights(input)`, plus getters for `weightAt`, `getDelta`, `getOutput`.

### `class NN`

- Built from a `vector<pair<layer_size, Activation>>` and an input size, so any depth/width
  can be constructed without editing code.
- `forwardAll(X)` runs the input through every layer, keeping all intermediate activations.
- `train(X, label)` runs the full manual backpropagation:
  1. Forward pass.
  2. Build the target vector `Y` (one-hot from the integer label).
  3. Output-layer deltas: `delta = lossDerivative(a, y) * activationDerivative(z)`.
  4. Hidden-layer deltas in reverse: `delta = (Σ w_next · delta_next) * activationDerivative(z)`.
  5. Update every weight and bias with `w -= LR * delta * input`.
- `predict(X)` returns `argmax` of the output layer.

### Activation functions

An `enum Activation { NONE, SIGMOID, RELU, TANH, SOFTMAX }` with `activate()` and
`activationDerivative()` written out for each — the derivatives by hand, no library.

### MNIST loader (written from the raw bytes)

- `readInt()` reads a 4-byte big-endian unsigned integer (MNIST stores headers big-endian,
  while x86 is little-endian, so the bytes are manually swapped).
- `loadImages()` / `loadLabels()` parse the IDX header (magic number, count, rows, cols),
  validate the magic numbers (`2051` for images, `2049` for labels), and normalize pixels
  from `0–255` to `0.0–1.0`.

### The two built-in experiments

- `test()` — tiny XOR network (`2 → 4 tanh → 2 sigmoid`), trained for 20,000 epochs.
- `test2()` — MNIST: `784 → 128 ReLU → 10 sigmoid`, 5 epochs, single-image (online) updates,
  reporting train and test accuracy.

**Result:** the C++ network reaches **~63.3% test accuracy** on MNIST (per the commit history).
The low accuracy compared to the Python notebook is a direct lesson in why batching, better
loss/activation pairing (softmax + cross-entropy) and vectorization matter — the C++ version
does one image at a time and uses sigmoid outputs with a per-neuron squared-error-style delta.

### Build & run

The loader uses the relative path `../data/...`, so run the binary from inside `from_scratch_cpp/`:

```bash
cd from_scratch_cpp
g++ -O2 -o nn neural_network_scratch.cpp
./nn
```

`test2()` is the active experiment in `main()`. Uncomment `test()` in `main()` to run the XOR
demo instead.

> **Note:** `neural_network_scratch_raw.cpp` and `neural_network_scratch.cpp` currently hold the
> same generalized implementation — the `_raw` file preserves the first complete version and the
> other is the cleaned-up final. The design described above is the graph/independent-neuron idea
> that both files share.

---

## Part 2 — Python: A Single Perceptron, Then the Smallest XOR Network

**Files:** `re_learn_from_scratch_python/perceptron/`

This is a fresh re-derivation of the same concepts in Python, days later, starting from the
smallest possible unit.

### `1_perceptron_learns_OR.py` — one neuron

- A `Neuron` class with He-initialized weights and a step activation (`1 if z > 0 else 0`).
- Manual stochastic gradient descent written out per weight — no matrices, no loops over
  abstractions:
  `W[i] -= LR * (y_hat - y) * X[i]`, `b -= LR * (y_hat - y)`.
- Targets logical **OR** by default (switch the target list to `[0,0,0,1]` for **AND**).
- The comments record the key realization: a single perceptron draws one straight line, so it
  solves linearly separable OR/AND but **cannot** solve XOR.

### `2_perceptrons_learns_XOR.py` — the smallest network

- Vectorized NumPy network: `2 → 2 (ReLU) → 1 (sigmoid)`, MSE-style backprop, 100,000 epochs.
- The long header comment is the real lesson: the code is *theoretically* correct but only
  works *most* of the time, because
  - ReLU can kill a neuron (`z < 0` → permanently 0), and with only 2 hidden neurons a single
    dead neuron is fatal;
  - random initialization can land in a poor local minimum;
  - seeds like `42` happen to work.
- Also notes the alternative fixes explored: more hidden neurons, `tanh` instead of ReLU,
  and a cross-entropy derivative instead of MSE.

### `3_fixed_perceptron_learns_XOR.py` — the fix, parameterized

- Same task, with the fixes applied: hidden layer uses `tanh`, output uses `sigmoid`, the loss
  is Binary Cross-Entropy, and the derivative of the loss w.r.t. the pre-activation collapses
  to the elegant `dL/dZ2 = A2 - Y`.
- Layer sizes are now **variables** (`NUM_HIDDEN_LAYER_1_NEURONS`, etc.), and the hidden layer
  is given 3 neurons instead of 2, which reliably escapes the bad-initialization trap.
- Comment notes the memorizable identities used: `d(tanh z)/dz = 1 - tanh²(z)` (written as
  `1 - A1**2` since `A1 = tanh(Z1)`), and BCE + sigmoid ⇒ `A - Y`.

**Run any of them:**

```bash
cd re_learn_from_scratch_python/perceptron
python3 1_perceptron_learns_OR.py
python3 2_perceptrons_learns_XOR.py
python3 3_fixed_perceptron_learns_XOR.py
```

---

## Part 3 — Matrix Formulation + Theory: Fully Connected Network on MNIST

**File:** `re_learn_from_scratch_python/THEORY_FCNN_MNIST/mnist_theory.ipynb`
**Saved model:** `re_learn_from_scratch_python/THEORY_FCNN_MNIST/weights.npz`

The notebook is written as a guided theory document (markdown explanations interleaved with the
code that implements them), organized into five phases.

### Network structure

```
Input (784) → Dense 128 → ReLU → Dense 10 → Softmax → 10-class prediction
W1: (784, 128)   b1: (128,)
W2: (128, 10)    b2: (10,)
```

### PHASE 1 — Load the data

- `load_images()` / `load_labels()` read the IDX binary format with `np.frombuffer`.
  The 16-byte header is read as four big-endian `>u4` integers (magic, count, rows, cols),
  the pixel bytes as `uint8`, then reshaped `(N, 28, 28)`.
- The notes explain endianness, why `uint8` doesn't need it, and how the flat byte buffer
  becomes a 3-D array.
- Uses `matplotlib` to visualize a sample digit, and checks per-digit frequency
  (`np.bincount`) to confirm the dataset is roughly balanced.

### PHASE 2 — Validation

- Builds a **stratified validation split**: 1,000 images per digit (10,000 total) are held out
  for validation; the remaining 50,000 are used for training.
- Important detail documented in the notebook: shuffle an **index array** and apply it to both
  `X` and `y`, so features and labels stay aligned.
- Pixels are scaled to `0–1` (`float32`) and each `28×28` image is flattened to a `784` vector.

### PHASE 3 — Forward pass

- `init_params()` uses **He initialization**: `randn * sqrt(2 / fan_in)`, biases at zero.
  The notes explain the normal distribution and the `sqrt(2/n)` scaling.
- `relu()`, `deriv_relu()`, and a numerically stable `softmax()` (subtract the row max before
  exponentiating) — with notes on broadcasting, `axis=1`, and `keepdims`.
- `forward_prop()` computes `Z1 = X@W1 + b1`, `A1 = relu(Z1)`, `Z2 = A1@W2 + b2`,
  `A2 = softmax(Z2)`, working on batches of 64.
- The notebook walks through the exact tensor shapes for one example vs. a full batch, and
  explains NumPy broadcasting (`(64,128) + (128,)`).

### PHASE 4 — Backpropagation

- Targets are **one-hot** encoded (`one_hot()`).
- Loss is **Categorical Cross-Entropy**; the notes show it collapsing to `-log(A2[correct])`
  for a single one-hot example, averaged over the batch.
- The key gradient identity is used: `dL/dZ2 = (A2 - Y) / batch_size`.
- `back_prop()` computes:
  - `dZ2 = (A2 - Y)/B`, `dW2 = A1ᵀ @ dZ2`, `db2 = sum(dZ2, axis=0)`
  - `dZ1 = (dZ2 @ W2ᵀ) * deriv_relu(Z1)`, `dW1 = Xᵀ @ dZ1`, `db1 = sum(dZ1, axis=0)`
  - with notes about why each transpose is needed to match parameter shapes.
- `train_and_save_weights()` runs 30 epochs of mini-batch gradient descent (batch 64,
  LR 0.05), printing loss / train accuracy / validation accuracy each epoch, then saves the
  parameters with `np.savez(..., W1, b1, W2, b2)`.

### PHASE 5 — Use the model

- `load_weights()` restores `weights.npz` so the model can be reused **without retraining**.
- Evaluates on the 10,000-image test set and runs a few sample predictions.

### Recorded results

Training log from the notebook (30 epochs, batch 64, LR 0.05):

```
Epoch  1/30 | Loss: 0.4806 | Train Acc: 0.8719 | Val Acc: 0.9051
Epoch 10/30 | Loss: 0.1030 | Train Acc: 0.9714 | Val Acc: 0.9608
Epoch 20/30 | Loss: 0.0559 | Train Acc: 0.9852 | Val Acc: 0.9699
Epoch 30/30 | (loss continues to fall, val acc ~0.97)
Test Accuracy: 0.9754
```

**~97.5% test accuracy** with a two-layer network written entirely by hand.

**Run it:**

```bash
cd re_learn_from_scratch_python/THEORY_FCNN_MNIST
jupyter notebook mnist_theory.ipynb
```

The notebook loads data from `../../data/`, i.e. the repository-root `data/` folder, and writes
`weights.npz` into the same directory it lives in.

---

## Part 4 — Inference App: Camera Digit Recognition

**Files:** `digit_recognition/`
**Pretrained weights:** `digit_recognition/weights.npz`

A Flask web app that recognizes digits captured with a camera. It does **no training** — it loads
the `weights.npz` trained in Part 3 and only runs forward propagation. The weights are a copy of
the ones from my theory notebook:

**neural_network_theories → re_learn_from_scratch_python/THEORY_FCNN_MNIST**

https://github.com/sadat2103108/neural_network_theories/tree/main/re_learn_from_scratch_python/THEORY_FCNN_MNIST

### Files

- `main.py` — Flask server. Serves the page, accepts `POST /predict` (base64 PNG), saves the
  capture, converts it, and returns the predicted digit as JSON.
- `neural_network.py` — loads `W1, b1, W2, b2` from `weights.npz` and runs the same forward pass
  as the notebook (`X@W1+b1 → ReLU → @W2+b2 → softmax → argmax`), inference only.
- `image_utils.py` — turns a camera photo into an MNIST-style `(1, 784)` float array.
- `templates/index.html` — the camera UI: video preview, red guide box, capture button, and the
  `fetch` call to `/predict`.

### How it works

1. The browser opens the camera (`getUserMedia`, environment camera when available) and overlays
   a square guide box on the video.
2. **Capture & Predict** crops the video to that guide box, exports a base64 PNG, and `POST`s it
   to `/predict`.
3. The server decodes and saves the image, then `image_to_mnist()`:
   - flattens illumination (divide by a heavily blurred copy) to cancel shadows and uneven light;
   - thresholds with Otsu plus an adaptive threshold, keeping only genuinely dark pixels;
   - closes small gaps (`_repair_broken_strokes`) and removes tiny noise components;
   - groups nearby stroke parts and picks the most central/complete digit (`_find_digit_box`);
   - resizes to a 20×20 region on a black 28×28 canvas and centers it by center of mass;
   - scales to `0.0–1.0` and flattens to `(1, 784)`.
4. `nn.predict(X)` returns the digit, which the page displays.

### Run

```bash
cd digit_recognition
pip install flask numpy opencv-python
python main.py
```

Open **http://127.0.0.1:5000** (use this exact address, since `getUserMedia` only works in a
secure context and localhost counts as one).

### Debug images

Each request writes the intermediate stages to `digit_recognition/debug/` — `01_gray`,
`02_flattened`, `03_before_repair`, `04_after_repair`, and `05_final_mnist.png`, the exact 28×28
array the network saw. This is the fastest way to debug a wrong prediction.

---

## Concepts Covered End-to-End

- The artificial neuron: weighted sum + bias + activation
- Perceptron learning rule / stochastic gradient descent
- Linear separability — why OR/AND work and XOR needs a hidden layer
- Activation functions and their derivatives: step, sigmoid, tanh, ReLU, softmax
- Dead ReLU, local minima, and the role of weight initialization
- Loss functions: squared error, MSE, Binary/Categorical Cross-Entropy
- Manual backpropagation via the chain rule (both per-neuron and matrix forms)
- One-hot encoding, `argmax` prediction
- He initialization, numerically stable softmax, mini-batch training
- The MNIST IDX binary format, endianness, and a from-bytes data loader
- Vectorization and NumPy broadcasting
- Saving/loading trained weights and separating training from inference
- Why the matrix formulation is both simpler and faster than the graph-of-neurons form
- Deploying a trained network for inference in a web app, and preparing noisy camera photos
  (illumination flattening, Otsu + adaptive thresholding, stroke repair, centering) into the
  MNIST input format

## Requirements

- **C++**: any compiler with C++17 (`g++`); no external libraries.
- **Python**: `numpy`, `matplotlib`, and Jupyter (for the notebook). The perceptron scripts
  need only `numpy` (plus the standard-library `random`/`math` for the first one).
- **Web app** (`digit_recognition/`): `flask` and `opencv-python` in addition to `numpy`.

```bash
pip install numpy matplotlib jupyter
pip install flask numpy opencv-python
```

## Data

The raw MNIST IDX files are tracked under `data/` and are read directly by both the C++ loader
and the notebook — no downloading or CSV conversion step. The `weights.npz` file is a trained
artifact (~815 KB) that can be loaded to skip training.
