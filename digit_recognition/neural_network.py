import numpy as np


# ============================================================
# Model weights
# ============================================================

__W1 = None
__b1 = None
__W2 = None
__b2 = None


# ============================================================
# Activation functions
# ============================================================

def __softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))

    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def __ReLU(z):
    return np.maximum(0, z)


# ============================================================
# Load model weights
# ============================================================

def load_weights(filename="weights.npz"):
    global __W1, __b1, __W2, __b2

    data = np.load(filename)

    __W1 = data["W1"]
    __b1 = data["b1"]
    __W2 = data["W2"]
    __b2 = data["b2"]


# ============================================================
# Forward propagation
# ============================================================

def __forward_prop(X):
    Z1 = X @ __W1 + __b1
    A1 = __ReLU(Z1)

    Z2 = A1 @ __W2 + __b2
    A2 = __softmax(Z2)

    return Z1, A1, Z2, A2


# ============================================================
# Prediction
# ============================================================

def predict(X):
    if __W1 is None:
        raise RuntimeError("Model weights are not loaded. Call load_weights() first.")

    _, _, _, A = __forward_prop(X)

    return np.argmax(A, axis=1)
