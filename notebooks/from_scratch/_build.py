"""Generator for the ``notebooks/from_scratch/`` series.

These notebooks teach neural networks from first principles by *implementing*
each piece in NumPy before showing how PyTorch wraps the same thing. They are
self-contained — no dependency on the project's ``ccdp`` package except in
the final notebook which compares the from-scratch approach to the real
production pipeline.

Re-run after editing to regenerate the .ipynb files:

    python notebooks/from_scratch/_build.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Cell factories
# ---------------------------------------------------------------------------


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True) or [""],
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True) or [""],
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# ---------------------------------------------------------------------------
# Shared bootstrap cell. CIFAR-10 is downloaded on demand via torchvision (only
# used for the dataset; the *models* are pure NumPy). Stays cached across runs.
# ---------------------------------------------------------------------------

BOOTSTRAP = code('''\
# === Bootstrap (safe to re-run) ===
# Installs minimal deps and downloads tiny CIFAR-10 subset for any notebook
# that needs it. The from-scratch notebooks deliberately avoid importing the
# project package so they run anywhere (Colab, plain Python, Jupyter).
import os, sys, subprocess
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "numpy", "matplotlib", "pillow", "torchvision"], check=True)
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
print("numpy", np.__version__)
''')


def load_cifar_subset_cell() -> dict:
    """Re-usable cell that produces (X_train, y_train, X_test, y_test) at
    32×32×3 in NumPy with values in [0, 1]. We use torchvision only for the
    dataset download; everything else is NumPy."""
    return code('''\
# Load a tiny CIFAR-10 subset.
# torchvision is used ONLY to fetch the dataset bytes — no torch tensors,
# no torch.nn. We immediately convert to NumPy.
from torchvision import datasets
from pathlib import Path
DATA_ROOT = Path("./_cifar_cache")
DATA_ROOT.mkdir(exist_ok=True)

train_ds = datasets.CIFAR10(root=DATA_ROOT, train=True,  download=True)
test_ds  = datasets.CIFAR10(root=DATA_ROOT, train=False, download=True)

def to_numpy(ds, n):
    X = np.stack([np.asarray(ds[i][0], dtype=np.float32) / 255.0 for i in range(n)])
    y = np.array([ds[i][1] for i in range(n)], dtype=np.int64)
    return X, y

X_train, y_train = to_numpy(train_ds, 2000)
X_test,  y_test  = to_numpy(test_ds,  500)
CLASSES = ("plane","car","bird","cat","deer","dog","frog","horse","ship","truck")
print("X_train", X_train.shape, "y_train", y_train.shape)
print("X_test ", X_test.shape,  "y_test ", y_test.shape)
''')


# ===========================================================================
# Notebook 00 — Math foundations
# ===========================================================================

NB_00 = notebook([
    md("""\
# 00 — Math foundations

This is the only math notebook. Everything that follows assumes you understand the four ideas here:

1. **Vectors and matrices** — a CNN is matrix multiplications glued together.
2. **Dot product** — every neuron computes one of these.
3. **Derivatives and the chain rule** — backpropagation *is* the chain rule.
4. **Gradient** — the direction "downhill" in a loss landscape.

We'll keep it surgical: introduce a concept, write it in NumPy, see it work.
"""),
    BOOTSTRAP,
    md("""\
## 1. Vectors

A vector is just an ordered list of numbers. Geometrically, it's an arrow from the origin.

$$\\vec{v} = \\begin{bmatrix} 2 \\\\ 3 \\end{bmatrix}$$

In NumPy:
"""),
    code('''\
v = np.array([2.0, 3.0])
print("vector:", v)
print("length (Euclidean):", np.linalg.norm(v))  # √(2² + 3²) = √13 ≈ 3.606
'''),
    md("""\
## 2. Dot product — the atom of neural nets

The dot product of two vectors of the same length is

$$\\vec{a} \\cdot \\vec{b} = \\sum_i a_i b_i$$

Geometrically: how aligned the two arrows are. **A single neuron computes a dot product of weights and inputs.**

```
neuron output (pre-activation) = w · x + b
                                  ^^^^^
                                  dot product
```
"""),
    code('''\
w = np.array([0.5, -1.0, 2.0])  # neuron's weights
x = np.array([1.0,  2.0, 0.5])  # an input
b = 0.1                          # bias
pre_activation = w @ x + b       # `@` is matmul; for 1-D it's dot product
print("w · x + b =", pre_activation)
'''),
    md("""\
## 3. Matrices and matrix multiplication

A matrix is a 2-D grid of numbers. **A whole layer of neurons is one matrix multiplication.**

If you have a layer with $n$ inputs and $m$ output neurons, its weight matrix is shape $(m, n)$. For one input vector $x$ of shape $(n,)$:

$$h = W \\cdot x + \\vec{b}$$

`h` has shape $(m,)$ — one number per neuron.
"""),
    code('''\
W = np.array([[ 0.5, -1.0,  2.0],   # neuron 0
              [ 1.0,  0.0,  1.5],   # neuron 1
              [-0.2,  0.3,  0.8]])  # neuron 2
b = np.array([0.1, 0.0, -0.5])
x = np.array([1.0, 2.0, 0.5])
h = W @ x + b
print("layer output:", h)
'''),
    md("""\
## 4. Derivatives

A derivative tells you how fast a function changes. For $f(x) = x^2$, the derivative is $f'(x) = 2x$.

Why neural nets care: the **loss** is a function of all the weights. To reduce the loss, we need to know "if I nudge this weight a little, does the loss go up or down?" That's a derivative.
"""),
    code('''\
# Numerically verify d/dx (x^2) = 2x at x=3 → 6.
def f(x): return x**2
x = 3.0
h = 1e-6
slope = (f(x+h) - f(x-h)) / (2*h)
print(f"numerical derivative at x=3: {slope:.6f}   (exact: 6)")
'''),
    md("""\
## 5. Chain rule — the entire reason backprop works

If $y = g(f(x))$ then

$$\\frac{dy}{dx} = \\frac{dy}{df} \\cdot \\frac{df}{dx}$$

A neural net is a long composition: $\\text{loss}(\\text{layer}_N(\\text{layer}_{N-1}(\\dots(\\text{layer}_1(x)))))$. To compute the derivative of the loss with respect to any weight, you multiply the local derivatives along the chain. That is backprop. There is no other magic.
"""),
    code('''\
# Example: y = (3x + 1)^2.  dy/dx = 2(3x+1) · 3 = 6(3x+1)
# At x=2: dy/dx = 6 · 7 = 42.
def g(x): return 3*x + 1
def f(g): return g**2
x = 2.0; h = 1e-6
print("numerical:", (f(g(x+h)) - f(g(x-h))) / (2*h), "  exact: 42")
'''),
    md("""\
## 6. Gradient — derivative for many variables at once

If your loss depends on a million weights, the **gradient** is just the vector of all those partial derivatives. NumPy lets us compute these efficiently because matrix multiplication and broadcasting handle the bookkeeping.

We're done with prerequisites. Next: load real images, then build a neuron from these four ideas.
"""),
])


# ===========================================================================
# Notebook 01 — Data preprocessing
# ===========================================================================

NB_01 = notebook([
    md("""\
# 01 — Data preprocessing

The model is downstream. Most real-world bugs are in this notebook's worth of code: bad dtype, wrong channel order, forgotten normalization, wrong image size.

## Roadmap
1. Image as a NumPy array
2. dtype and value range — the most common bug
3. Channel order: RGB vs BGR vs CHW vs HWC
4. **Image size — decision table for classifier vs detector vs regressor**
5. Normalization: why, and how to compute the constants
6. Augmentation: random flip, crop, color jitter in NumPy
7. One-hot encoding (classifier) vs scalar target (regressor)
"""),
    BOOTSTRAP,
    md("""\
## 1. Image as a NumPy array

A color image is a 3-D array: `(height, width, channels)`. Each channel value is typically 0..255 (uint8) or 0..1 (float).
"""),
    code('''\
from PIL import Image
import numpy as np

# Make a synthetic 4×6 RGB image: red square on white background
arr = np.full((4, 6, 3), 255, dtype=np.uint8)
arr[1:3, 1:3] = [255, 0, 0]
img = Image.fromarray(arr, mode="RGB")
print("shape:", arr.shape)
print("first pixel (R,G,B):", arr[0, 0])
img.resize((180, 120))   # display upscale
'''),
    md("""\
## 2. dtype + value range (the most common silent bug)

| Where | Typical dtype | Typical range |
|---|---|---|
| Disk (jpg/png) | uint8 | 0..255 |
| Just-loaded NumPy | uint8 | 0..255 |
| **Model input** | **float32** | **0..1 or normalised** |
| Model output | float32 | depends on layer |

**Always convert to float and divide by 255** before feeding a model. uint8 arithmetic wraps around silently — `np.uint8(200) + np.uint8(100) == 44`.
"""),
    code('''\
u = np.array([200, 100], dtype=np.uint8)
print("uint8 add:", u.sum())               # 44 — wraparound!
f = u.astype(np.float32) / 255.0
print("float add:", f.sum())               # 1.176 — correct
'''),
    md("""\
## 3. Channel order: RGB vs BGR, HWC vs CHW

- **PIL / matplotlib / torchvision** use **HWC + RGB**.
- **OpenCV** uses **HWC + BGR**.
- **PyTorch tensors** use **CHW + RGB**.

Mixing these is the #2 most common bug — model trained with one order tested with another quietly drops 20+ accuracy points.
"""),
    code('''\
# Convert HWC → CHW (NumPy axis transpose)
hwc = np.zeros((32, 32, 3), dtype=np.float32)
chw = hwc.transpose(2, 0, 1)                 # axes: 2→0, 0→1, 1→2
print("HWC:", hwc.shape, " → CHW:", chw.shape)
'''),
    md("""\
## 4. Image-size decision rules

You **never** feed the raw image. You pick a target size based on the *task*:

| Task | Standard input size | Why |
|---|---|---|
| Image classifier (ResNet-style) | **224 × 224** | Pretrained ImageNet weights use this. Smaller loses fine detail; larger costs 4x memory per doubling. |
| Object detector (YOLO-style) | **640 × 640** | Bigger so small objects survive. Must be divisible by 32 (the network's downsampling stride). |
| Pixel-precise (segmentation) | 512–1024 | Need spatial detail in the output. |
| Image regressor (cost, age, …) | 224 × 224 | Same as classifier; the head is a single scalar instead of N classes. |

**Output-size formula for a conv layer:**

$$H_\\text{out} = \\left\\lfloor \\frac{H_\\text{in} + 2P - K}{S} \\right\\rfloor + 1$$

where $K$ is kernel size, $S$ is stride, $P$ is padding. Worked example: a 224×224 image through a 7×7 conv with stride 2 and padding 3:

$$\\left\\lfloor \\frac{224 + 6 - 7}{2} \\right\\rfloor + 1 = 112$$
"""),
    code('''\
def conv_output_size(H_in, K, S, P):
    return (H_in + 2*P - K) // S + 1

print("first conv of ResNet50:", conv_output_size(224, K=7, S=2, P=3))   # 112
print("after maxpool 3×3 s2 p1:", conv_output_size(112, K=3, S=2, P=1))  # 56
'''),
    md("""\
## 5. Normalization — why and how

Networks train better when inputs are roughly mean 0, std 1. Two reasons:

1. **Activations stay in the useful range** of sigmoid / tanh / softmax.
2. **Gradient magnitudes are stable** across layers — avoids vanishing/exploding gradients.

For ImageNet-pretrained models, the standard constants are computed once over millions of training images:

```
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

For your own dataset, compute them from the train split.
"""),
    code('''\
# Compute CIFAR-10-style normalization constants from a small batch.
imgs = np.random.rand(100, 32, 32, 3).astype(np.float32)   # fake
mean = imgs.mean(axis=(0, 1, 2))
std  = imgs.std(axis=(0, 1, 2))
print("mean per channel:", mean.round(3), "  std:", std.round(3))
'''),
    md("""\
## 6. Augmentation — three useful ones in 10 lines of NumPy
"""),
    code('''\
def random_horizontal_flip(img, p=0.5):
    return img[:, ::-1] if np.random.rand() < p else img

def random_crop_and_resize(img, crop_size, out_size):
    """Crop a random crop_size×crop_size region then upscale (nearest) to out_size."""
    H, W = img.shape[:2]
    y = np.random.randint(0, H - crop_size + 1)
    x = np.random.randint(0, W - crop_size + 1)
    crop = img[y:y+crop_size, x:x+crop_size]
    # Nearest-neighbour upscale; for production prefer LANCZOS via PIL.
    factor = out_size / crop_size
    rows = (np.arange(out_size) / factor).astype(int)
    cols = (np.arange(out_size) / factor).astype(int)
    return crop[np.ix_(rows, cols)]

def color_jitter(img, strength=0.1):
    return np.clip(img + np.random.uniform(-strength, strength, size=3), 0, 1)

# Quick demo on a synthetic image
fig, axes = plt.subplots(1, 4, figsize=(12, 3))
src = np.random.rand(32, 32, 3).astype(np.float32)
for ax, fn, t in zip(axes,
                      [lambda x: x, random_horizontal_flip,
                       lambda x: random_crop_and_resize(x, 24, 32),
                       color_jitter],
                      ["original", "h-flip", "random crop+resize", "color jitter"]):
    ax.imshow(fn(src)); ax.set_title(t); ax.axis("off")
plt.show()
'''),
    md("""\
## 7. Targets: classifier vs regressor

The model's *output head* and the *loss function* depend on what you want it to predict.

| Task | Target shape | Final layer | Loss |
|---|---|---|---|
| Multi-class (one of N) | scalar int $\\in [0, N)$ | N logits | Softmax + cross-entropy |
| Multi-label (any subset of N) | length-N vector of 0/1 | N logits | N independent BCEs |
| Regression (scalar) | one float | 1 linear output | MSE or MAE |
| Detection (boxes + classes) | list of (cls, bbox) | per-grid-cell predictions | sum of regression + classification |
"""),
    code('''\
# One-hot encode an integer label
def one_hot(y, num_classes):
    out = np.zeros(num_classes, dtype=np.float32)
    out[y] = 1.0
    return out

print(one_hot(3, num_classes=10))   # plane=0 ... dog=5 ... so index 3 = cat → [0,0,0,1,0,0,0,0,0,0]
'''),
    md("""\
**Next:** build a single neuron, then a 2-layer MLP, and train it on a toy problem.
"""),
])


# ===========================================================================
# Notebook 02 — Perceptron to MLP
# ===========================================================================

NB_02 = notebook([
    md("""\
# 02 — From perceptron to MLP (with manual backprop)

## Roadmap
1. The single neuron (perceptron) — math and code
2. Activation functions — sigmoid, ReLU, tanh (each with its derivative)
3. Loss functions — MSE vs cross-entropy
4. Manual forward + backward pass for a **2-layer MLP**
5. Gradient descent — actually train the thing
6. Run on a tiny CIFAR subset and watch the loss drop
"""),
    BOOTSTRAP,
    md("""\
## 1. The single neuron

A neuron is a dot product followed by a non-linear activation:

$$y = \\sigma(w \\cdot x + b)$$

Without the non-linearity, stacking neurons does **nothing** — composition of linear functions is still linear. The non-linearity is what makes deep networks expressive.
"""),
    code('''\
def neuron(x, w, b, activation):
    return activation(x @ w + b)

# A "or-gate-ish" neuron with sigmoid activation
def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))

w = np.array([5.0, 5.0]); b = -2.5
for x in [[0,0], [0,1], [1,0], [1,1]]:
    print(x, "→", round(float(neuron(np.array(x), w, b, sigmoid)), 3))
'''),
    md("""\
## 2. Activation functions (and why we need their derivatives)

| Name | Formula | Derivative | Use |
|---|---|---|---|
| **Sigmoid** | $\\sigma(z) = \\frac{1}{1+e^{-z}}$ | $\\sigma(z)(1-\\sigma(z))$ | Binary output |
| **Tanh** | $\\tanh(z)$ | $1 - \\tanh^2(z)$ | Hidden, classical |
| **ReLU** | $\\max(0, z)$ | 0 if z<0 else 1 | Hidden, modern default |
| **Softmax** | $e^{z_i} / \\sum_j e^{z_j}$ | (computed via cross-entropy) | Multi-class final |
"""),
    code('''\
def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))
def sigmoid_grad(z): s = sigmoid(z); return s * (1 - s)
def relu(z): return np.maximum(0, z)
def relu_grad(z): return (z > 0).astype(z.dtype)
def tanh(z): return np.tanh(z)
def tanh_grad(z): return 1 - np.tanh(z) ** 2

# Plot the three
zs = np.linspace(-5, 5, 200)
fig, axes = plt.subplots(1, 3, figsize=(12, 3))
for ax, (fn, gn, name) in zip(axes, [(sigmoid, sigmoid_grad, "sigmoid"),
                                       (tanh, tanh_grad, "tanh"),
                                       (relu, relu_grad, "ReLU")]):
    ax.plot(zs, fn(zs), label=name)
    ax.plot(zs, gn(zs), label="derivative", linestyle="--")
    ax.set_title(name); ax.legend(); ax.grid(True)
plt.show()
'''),
    md("""\
## 3. Loss functions

A loss function tells you "how wrong" the model is on a training example. The training loop nudges weights to make this smaller.

**MSE (regression):** $\\mathcal{L} = \\frac{1}{N} \\sum_i (\\hat{y}_i - y_i)^2$

**Cross-entropy (classification):** $\\mathcal{L} = -\\sum_i y_i \\log \\hat{y}_i$

Cross-entropy combined with softmax has a beautifully simple gradient: $\\hat{y} - y$. This is why every classifier uses this combination.
"""),
    code('''\
def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)   # numerical stability
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def cross_entropy(y_true_onehot, y_pred):
    # add small eps for numerical safety
    return -float((y_true_onehot * np.log(y_pred + 1e-9)).sum(axis=-1).mean())

# example
probs = softmax(np.array([[2.0, 0.1, 0.2]]))
print("softmax:", probs.round(3), "sum:", probs.sum())
print("CE if true class=0:", cross_entropy(np.array([[1,0,0]]), probs))
'''),
    md("""\
## 4. The 2-layer MLP — implement forward AND backward by hand

Two linear layers with a ReLU between them, ending in softmax + cross-entropy. We will derive every gradient and verify against NumPy.

Architecture: `input (D)  →  Linear(D, H)  →  ReLU  →  Linear(H, C)  →  Softmax  →  CE loss`
"""),
    code('''\
class MLP:
    """2-layer MLP, pure NumPy, with manual backprop."""

    def __init__(self, in_dim, hidden, out_dim, lr=0.1, seed=0):
        rng = np.random.default_rng(seed)
        # He init for ReLU layer; small Gaussian for the second.
        self.W1 = rng.standard_normal((in_dim, hidden)).astype(np.float32) * np.sqrt(2.0/in_dim)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rng.standard_normal((hidden, out_dim)).astype(np.float32) * 0.01
        self.b2 = np.zeros(out_dim, dtype=np.float32)
        self.lr = lr

    def forward(self, X):
        # X: (N, D)
        self.X = X
        self.z1 = X @ self.W1 + self.b1            # (N, H)
        self.a1 = np.maximum(0, self.z1)           # ReLU
        self.z2 = self.a1 @ self.W2 + self.b2      # (N, C)
        self.p  = softmax(self.z2)                 # (N, C)
        return self.p

    def loss(self, Y_onehot):
        return -float((Y_onehot * np.log(self.p + 1e-9)).sum(axis=-1).mean())

    def backward(self, Y_onehot):
        # Softmax + CE gradient is simply (p - y) / N
        N = self.X.shape[0]
        dz2 = (self.p - Y_onehot) / N                  # (N, C)
        dW2 = self.a1.T @ dz2                          # (H, C)
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.W2.T                          # (N, H)
        dz1 = da1 * (self.z1 > 0)                      # ReLU' = step
        dW1 = self.X.T @ dz1                           # (D, H)
        db1 = dz1.sum(axis=0)
        # SGD step
        self.W1 -= self.lr * dW1; self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2; self.b2 -= self.lr * db2

print("MLP class compiled.")
'''),
    md("""\
## 5. Train on CIFAR-10 (tiny, flattened to a vector)

A flat MLP on raw image pixels is **not** the right architecture for images — that's why NB 03+04 exist. But it converges enough to demonstrate the training loop.
"""),
    load_cifar_subset_cell(),
    code('''\
# Flatten 32x32x3 images to vectors of 3072, take 2 classes (cat, dog) for speed
mask_tr = (y_train == 3) | (y_train == 5)
mask_te = (y_test  == 3) | (y_test  == 5)
X_tr = X_train[mask_tr].reshape(-1, 32*32*3)
y_tr = (y_train[mask_tr] == 5).astype(np.int64)   # 1 if dog, 0 if cat
X_te = X_test[mask_te].reshape(-1, 32*32*3)
y_te = (y_test[mask_te] == 5).astype(np.int64)

# Center the data — helps the MLP a lot
mean = X_tr.mean(axis=0); std = X_tr.std(axis=0) + 1e-8
X_tr = (X_tr - mean) / std
X_te = (X_te - mean) / std

Y_tr = np.eye(2)[y_tr].astype(np.float32)

mlp = MLP(in_dim=32*32*3, hidden=128, out_dim=2, lr=0.05)
hist = []
for epoch in range(20):
    mlp.forward(X_tr)
    L = mlp.loss(Y_tr)
    mlp.backward(Y_tr)
    p_te = mlp.forward(X_te)
    acc_te = (p_te.argmax(axis=-1) == y_te).mean()
    hist.append((L, acc_te))
    print(f"epoch {epoch+1:2d}  train_loss={L:.4f}  test_acc={acc_te:.3f}")

plt.figure(figsize=(8,3))
plt.plot([h[0] for h in hist], label="train loss")
plt.twinx().plot([h[1] for h in hist], color="orange", label="test acc")
plt.title("MLP on flattened CIFAR (cat vs dog)"); plt.show()
'''),
    md("""\
## 6. Why this isn't enough

A flattened image loses all spatial structure. Two pixels that are neighbours in the image are arbitrarily far apart in the flat vector — the MLP has to *learn* that they're related. Convolution gives the model that knowledge for free.

**Next:** implement `conv2d` from scratch.
"""),
])


# ===========================================================================
# Notebook 03 — Convolution from scratch
# ===========================================================================

NB_03 = notebook([
    md("""\
# 03 — Convolution from scratch

## Roadmap
1. Definition of 2-D convolution
2. Implement `conv2d` in 30 lines of NumPy
3. Hand-crafted kernels (edge, blur, sharpen) — same operator, different weights
4. Padding, stride, dilation — the three knobs
5. **Output-size formula** with worked examples
6. Pooling (max, average) and why it exists
7. The receptive field
"""),
    BOOTSTRAP,
    md("""\
## 1. Definition

2-D cross-correlation of image $I$ with kernel $K$ at position $(y, x)$:

$$(I * K)(y, x) = \\sum_{i=0}^{K_h-1} \\sum_{j=0}^{K_w-1} I(y+i, x+j) \\cdot K(i, j)$$

"Slide the kernel over the image, multiply element-wise, sum." Repeat at every position.

(Pedantic note: math literature calls this *cross-correlation*; deep-learning practice calls it *convolution*. We follow the deep-learning convention.)
"""),
    code('''\
def conv2d(img, kernel, padding=0, stride=1):
    """Single-channel 2-D convolution from scratch.

    img:    (H, W) float array
    kernel: (kh, kw) float array
    Returns: (Hout, Wout) float array
    """
    if padding > 0:
        img = np.pad(img, padding, mode="constant", constant_values=0)
    H, W = img.shape
    kh, kw = kernel.shape
    Hout = (H - kh) // stride + 1
    Wout = (W - kw) // stride + 1
    out = np.zeros((Hout, Wout), dtype=np.float32)
    for y in range(Hout):
        for x in range(Wout):
            patch = img[y*stride:y*stride+kh, x*stride:x*stride+kw]
            out[y, x] = (patch * kernel).sum()
    return out

print("conv2d ready")
'''),
    md("""\
## 2. Hand-crafted kernels

The same `conv2d` does completely different things depending on the kernel values.
"""),
    code('''\
# Build a tiny test image: vertical line down the middle
img = np.zeros((9, 9), dtype=np.float32)
img[:, 4] = 1.0

sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)
blur    = np.ones((3, 3), dtype=np.float32) / 9.0
sharpen = np.array([[ 0, -1,  0],
                    [-1,  5, -1],
                    [ 0, -1,  0]], dtype=np.float32)

fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for ax, (k, t) in zip(axes, [(None, "input"),
                              (sobel_x, "sobel-x (vertical edge)"),
                              (blur, "blur"),
                              (sharpen, "sharpen")]):
    if k is None:
        ax.imshow(img, cmap="gray"); ax.set_title(t)
    else:
        ax.imshow(conv2d(img, k, padding=1), cmap="RdBu"); ax.set_title(t)
    ax.axis("off")
plt.show()
'''),
    md("""\
**The whole point of training a CNN:** instead of you hand-picking these kernel values, the network *learns* them by gradient descent. Early layers tend to learn edge / color detectors that look very much like Sobel.

## 3. Padding, stride, dilation

| Knob | Effect | Use |
|---|---|---|
| **Padding** P | Adds P pixels of zeros around the input. Lets you preserve spatial size after conv. | "Same padding": $P = (K-1)/2$ keeps $H_\\text{out} = H_\\text{in}$ when stride 1. |
| **Stride** S | Move the kernel S steps each time. Downsamples by factor S. | Used instead of pooling in modern CNNs (ResNet uses stride-2 convs). |
| **Dilation** D | Spread kernel cells D pixels apart. Larger receptive field, same params. | Semantic segmentation (DeepLab). |

## 4. Output-size formula

$$H_\\text{out} = \\left\\lfloor \\frac{H_\\text{in} + 2P - K}{S} \\right\\rfloor + 1$$
"""),
    code('''\
def conv_output_size(H, K, P, S):
    return (H + 2*P - K) // S + 1

for (H, K, P, S, expected) in [
    (32, 3, 1, 1, 32),    # same conv
    (32, 3, 0, 1, 30),    # no padding
    (32, 3, 1, 2, 16),    # strided conv = downsample
    (224, 7, 3, 2, 112),  # ResNet50's first conv
]:
    got = conv_output_size(H, K, P, S)
    print(f"H={H} K={K} P={P} S={S} → {got}  ({'OK' if got==expected else f'expected {expected}'})")
'''),
    md("""\
## 5. Pooling

A pooling layer downsamples by taking the max (or average) over each KxK window. No learnable parameters. Two reasons it exists:

1. **Translation invariance** — small shifts of an object don't change the output.
2. **Spatial reduction** — fewer pixels = cheaper subsequent layers.

Modern CNNs (ResNet, EfficientNet) often use stride-2 convs *instead* of pooling, but it's still everywhere.
"""),
    code('''\
def max_pool(img, k=2, s=2):
    H, W = img.shape
    Hout = (H - k) // s + 1
    Wout = (W - k) // s + 1
    out = np.zeros((Hout, Wout), dtype=np.float32)
    for y in range(Hout):
        for x in range(Wout):
            out[y, x] = img[y*s:y*s+k, x*s:x*s+k].max()
    return out

x = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10,11,12],
              [13,14,15,16]], dtype=np.float32)
print("input:\\n", x)
print("\\nmax_pool 2x2, stride 2:\\n", max_pool(x, k=2, s=2))
'''),
    md("""\
## 6. Receptive field — what one output pixel "sees"

After $L$ stacked 3×3 convs (stride 1), one output pixel depends on a $(2L+1) \\times (2L+1)$ region of the input.

After 1 conv: 3×3. After 2 convs: 5×5. After 10 convs: 21×21.

This is why **depth helps**: deep stacks of small kernels see a large region without the parameter cost of one giant kernel.

**Next:** stack convs into a tiny CNN, train it on CIFAR, then add the residual trick.
"""),
])


def _write(name: str, nb: dict) -> None:
    path = HERE / name
    with path.open("w") as f:
        json.dump(nb, f, indent=1)
        f.write("\n")
    print(f"wrote {path.relative_to(HERE.parent.parent)}")


# More notebooks defined in the continuation file below. We split for
# readability; both files are imported and combined in main().
def main() -> None:
    from notebooks.from_scratch._build_part2 import NB_04, NB_05, NB_06, NB_07, NB_08, NB_09
    _write("00_math_foundations.ipynb",       NB_00)
    _write("01_data_preprocessing.ipynb",     NB_01)
    _write("02_perceptron_to_mlp.ipynb",      NB_02)
    _write("03_convolution_from_scratch.ipynb", NB_03)
    _write("04_tiny_cnn_to_resnet.ipynb",     NB_04)
    _write("05_yolo_from_scratch.ipynb",      NB_05)
    _write("06_optimisation_and_training.ipynb", NB_06)
    _write("07_metrics_and_diagnostics.ipynb",   NB_07)
    _write("08_epoch_visualization.ipynb",       NB_08)
    _write("09_pytorch_contrast.ipynb",          NB_09)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(HERE.parent.parent))
    main()
