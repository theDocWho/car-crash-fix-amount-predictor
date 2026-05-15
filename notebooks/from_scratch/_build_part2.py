"""Continuation of `notebooks/from_scratch/_build.py` — notebooks 04 through 09.

Split into two files for readability. Imported and dispatched from _build.py.
"""

from __future__ import annotations

from notebooks.from_scratch._build import BOOTSTRAP, load_cifar_subset_cell, code, md, notebook


# ===========================================================================
# Notebook 04 — Tiny CNN, then add the residual trick
# ===========================================================================

NB_04 = notebook([
    md("""\
# 04 — Tiny CNN → ResNet block

We'll stack the conv2d from NB 03 into a real (tiny) CNN, train it on CIFAR-10, then add a **residual connection** and see why ResNet50 can stack 50 layers deep without falling apart.

## Roadmap
1. Multi-channel conv (RGB in, many feature maps out) — the actual building block
2. Build a tiny CNN: 2 conv blocks + FC head
3. Why deep nets break: the **vanishing gradient** problem (demonstrated)
4. The **residual block** — 5 lines of code, decade-defining idea
5. How ResNet50 scales this to 50 layers

We'll use NumPy for forward passes and gradient bookkeeping but lean on PyTorch's `autograd` for the actual gradients here — implementing multi-channel conv backprop in NumPy is 200 lines of index-arithmetic and obscures the *idea*. This is the right time to introduce why PyTorch exists.
"""),
    BOOTSTRAP,
    md("""\
## 1. Multi-channel conv — the real building block

The conv from NB 03 was single-channel. A real conv layer has:

- **Input:** `(C_in, H, W)` — e.g. RGB image: C_in=3
- **Kernel:** `(C_out, C_in, kH, kW)` — `C_out` filters, each spanning all `C_in` input channels
- **Output:** `(C_out, H', W')`

Each output channel is the sum of separately-convolving each input channel with its own kernel slice, plus a bias.
"""),
    code('''\
import numpy as np
def multi_channel_conv2d(x, W, b, padding=1):
    """x: (C_in,H,W); W: (C_out,C_in,k,k); b: (C_out,)"""
    C_in, H, Wd = x.shape
    C_out, _, k, _ = W.shape
    xp = np.pad(x, [(0,0),(padding,padding),(padding,padding)])
    Hout = H + 2*padding - k + 1
    Wout = Wd + 2*padding - k + 1
    out = np.zeros((C_out, Hout, Wout), dtype=np.float32)
    for o in range(C_out):
        for y in range(Hout):
            for X in range(Wout):
                patch = xp[:, y:y+k, X:X+k]
                out[o, y, X] = (patch * W[o]).sum() + b[o]
    return out

x = np.random.randn(3, 8, 8).astype(np.float32)
W = np.random.randn(16, 3, 3, 3).astype(np.float32) * 0.1
b = np.zeros(16, dtype=np.float32)
y = multi_channel_conv2d(x, W, b, padding=1)
print("input:", x.shape, " → output:", y.shape, "  (3 RGB → 16 feature maps)")
'''),
    md("""\
## 2. Tiny CNN architecture

Two conv blocks (conv → ReLU → pool), then flatten, then a dense head. Small enough to train on CPU in minutes.

```
(3, 32, 32) → conv(16) → ReLU → maxpool 2     → (16, 16, 16)
            → conv(32) → ReLU → maxpool 2     → (32,  8,  8)
            → flatten                          → (2048,)
            → dense(10)                        → (10,)
            → softmax + cross-entropy
```

For training I'll switch to PyTorch (autograd) but the architecture is exactly what we built above — same operations, PyTorch just handles the gradient bookkeeping.
"""),
    code('''\
import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc    = nn.Linear(32 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))     # (B, 16, 16, 16)
        x = self.pool(F.relu(self.conv2(x)))     # (B, 32,  8,  8)
        x = x.flatten(1)                         # (B, 2048)
        return self.fc(x)                         # (B, 10)

print(TinyCNN())
print("param count:", sum(p.numel() for p in TinyCNN().parameters()))
'''),
    load_cifar_subset_cell(),
    code('''\
# Train the TinyCNN for a few epochs. Pure-NumPy training would take much longer
# because of conv backprop; that 200-line implementation is what `autograd` saves.
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

def to_chw(X):  # (N,H,W,C) → (N,C,H,W)
    return np.transpose(X, (0, 3, 1, 2))

Xt = torch.tensor(to_chw(X_train)).float()
yt = torch.tensor(y_train)
Xv = torch.tensor(to_chw(X_test)).float()
yv = torch.tensor(y_test)

model = TinyCNN().to(device)
optim = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
crit  = nn.CrossEntropyLoss()

BS = 64
loss_hist, acc_hist = [], []
for ep in range(8):
    model.train()
    perm = torch.randperm(len(Xt))
    for i in range(0, len(Xt), BS):
        idx = perm[i:i+BS]
        xb, yb = Xt[idx].to(device), yt[idx].to(device)
        optim.zero_grad()
        loss = crit(model(xb), yb)
        loss.backward()
        optim.step()
    model.eval()
    with torch.no_grad():
        preds = model(Xv.to(device)).argmax(dim=-1).cpu()
    acc = (preds == yv).float().mean().item()
    loss_hist.append(loss.item()); acc_hist.append(acc)
    print(f"epoch {ep+1}  loss={loss.item():.3f}  test_acc={acc:.3f}")

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(10, 3))
ax[0].plot(loss_hist); ax[0].set_title("train loss"); ax[0].grid(True)
ax[1].plot(acc_hist);  ax[1].set_title("test acc");   ax[1].grid(True)
plt.show()
'''),
    md("""\
## 3. The vanishing-gradient problem

Naively stacking 50 conv layers does **not** work. Each layer applies its weight matrix during the forward pass, and during backprop you multiply by all those weights again — small weights compound to ~zero gradient at the early layers; large weights compound to NaN. Training the early layers stops working.

Let me show this concretely by stacking many tanh activations and watching the gradient magnitude collapse.
"""),
    code('''\
import numpy as np
np.random.seed(0)

# Toy network: chain of L tanh activations, each fed through a random weight matrix.
def gradient_norm_after_L_layers(L, dim=64):
    x = np.random.randn(dim)
    grad = np.ones(dim)  # gradient at the output
    for _ in range(L):
        W = np.random.randn(dim, dim) * 0.5  # smallish weights
        z = W @ x
        x = np.tanh(z)
        # backward: ∂tanh/∂z = 1 - tanh²
        grad = (1 - x**2) * (W.T @ grad)
    return float(np.linalg.norm(grad))

depths = [1, 5, 10, 20, 50]
for L in depths:
    print(f"L={L:3d}: ||grad|| at input = {gradient_norm_after_L_layers(L):.3e}")
'''),
    md("""\
You'll see the gradient norm collapse exponentially — by 50 layers it's effectively zero. The first layer has no signal to learn from.

## 4. The residual fix

In each block, instead of computing $y = F(x)$, compute

$$y = F(x) + x$$

The "+x" is the **skip connection**. The block now learns the *residual* $F(x)$ — the change you want to add to $x$.

Two benefits:

1. **Identity is free.** If the best $F$ is "do nothing," the network drives $F(x) \\to 0$ trivially.
2. **Gradient highway.** $\\partial y / \\partial x = \\partial F / \\partial x + 1$. That `+1` means gradient *never* fully vanishes — it has a direct path back through every skip.
"""),
    code('''\
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return F.relu(out + identity)   # ← the entire trick

# Demo: same gradient experiment, now with residual connections
import torch
def grad_norm_residual(L, dim=64):
    x = torch.randn(1, dim, 4, 4, requires_grad=True)
    blocks = nn.Sequential(*[ResidualBlock(dim) for _ in range(L)])
    y = blocks(x).sum()
    y.backward()
    return float(x.grad.norm())

# Disable gradient warning for the demo
import warnings; warnings.filterwarnings("ignore")
for L in [1, 5, 10, 20, 50]:
    print(f"residual L={L:3d}: ||grad|| = {grad_norm_residual(L):.3e}")
'''),
    md("""\
Notice the gradient norm stays in a sensible range even at 50 layers. *That* is why ResNet50 can train end-to-end where a 50-layer plain CNN cannot.

## 5. How ResNet50 scales the idea

ResNet50 = 50 weight layers organised as:

- 1 initial conv (7×7, stride 2)
- 4 stages of residual blocks (3, 4, 6, 3 blocks each)
- Each block is `1×1 conv → 3×3 conv → 1×1 conv` ("bottleneck") + skip
- Global average pool → FC(1000)

Same residual idea, repeated. No new math.

**Next:** the totally different problem of **object detection** — predicting *where* things are, not just what.
"""),
])


# ===========================================================================
# Notebook 05 — Mini YOLO from scratch
# ===========================================================================

NB_05 = notebook([
    md("""\
# 05 — Mini-YOLO from scratch

Classification: "what's in the image?" Detection: "**where** are all the things?" Different problem, different output, different loss.

## Roadmap
1. Detection problem framing
2. Bounding boxes and IoU (from scratch)
3. The grid-cell trick — YOLO's core idea
4. Non-Max Suppression in 20 lines
5. A toy YOLO on synthetic "find the bright square" data
"""),
    BOOTSTRAP,
    md("""\
## 1. The detection problem

Input: image. Output: a **variable-length** list of `(class, bounding_box, confidence)` tuples.

That variable-length-list output is the hard part. A CNN naturally produces a fixed-size tensor, not a list. YOLO's clever trick: **always predict a fixed-size grid, then post-process to a variable list.**

## 2. Bounding-box conventions

A box can be:

- **xyxy**: (x1, y1, x2, y2) corners
- **xywh**: (xc, yc, w, h) center + size
- **Pixel** or **normalised** to image size

Be ruthless about converting at the boundary of every function.
"""),
    code('''\
import numpy as np

def xywh_to_xyxy(b):
    xc, yc, w, h = b
    return xc - w/2, yc - h/2, xc + w/2, yc + h/2

def iou(a, b):
    """IoU of two boxes in xyxy form."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter + 1e-9)

# Sanity: identical boxes → 1.0, disjoint → 0.0
print(iou((0,0,4,4), (0,0,4,4)))                # 1.0
print(iou((0,0,4,4), (10,10,12,12)))            # 0.0
print(round(iou((0,0,4,4), (2,2,6,6)), 3))      # 0.143
'''),
    md("""\
## 3. The grid-cell idea

Divide the image into an $S \\times S$ grid. **Each grid cell predicts:** `(p, x, y, w, h, c_0, c_1, ..., c_{N-1})`.

- `p`: objectness — is there anything here at all?
- `(x, y, w, h)`: the box (relative to the cell)
- `c_i`: per-class probability

For a 7×7 grid with 6 classes, the model output shape is `(7, 7, 1+4+6) = (7, 7, 11)`. That's a fixed tensor — a CNN can produce it.
"""),
    code('''\
import matplotlib.pyplot as plt, matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(6, 6))
S = 7
for i in range(S+1):
    ax.axhline(i/S, color="gray", lw=0.5)
    ax.axvline(i/S, color="gray", lw=0.5)
# Pretend ground-truth box centered in cell (3, 2)
gt = (0.35, 0.28, 0.30, 0.25)  # xc, yc, w, h
x1, y1, x2, y2 = xywh_to_xyxy(gt)
ax.add_patch(patches.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor="red", lw=2))
ax.plot(gt[0], gt[1], "ro"); ax.text(gt[0]+0.01, gt[1]-0.02, "object center", color="red")
ax.set_xlim(0,1); ax.set_ylim(1,0); ax.set_aspect("equal")
ax.set_title("Each cell predicts: 'is there an object whose center is in me?'")
plt.show()
'''),
    md("""\
## 4. Non-Max Suppression (NMS)

After the model predicts boxes everywhere, many of them overlap on the same object. NMS keeps the best one and suppresses the rest:

1. Sort boxes by confidence descending.
2. Pick the top box; add to keep list.
3. Discard remaining boxes with IoU > threshold.
4. Repeat.
"""),
    code('''\
def nms(boxes, scores, iou_thresh=0.5):
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if iou(boxes[i], boxes[j]) < iou_thresh]
    return keep

# Three overlapping boxes around the same object, plus one separate one
boxes = [
    (0.10, 0.10, 0.40, 0.40),     # A — same object as B,C
    (0.12, 0.12, 0.42, 0.42),     # B
    (0.11, 0.09, 0.41, 0.43),     # C
    (0.60, 0.60, 0.90, 0.90),     # D — separate object
]
scores = [0.92, 0.85, 0.78, 0.95]
keep = nms(boxes, scores, iou_thresh=0.4)
print("kept boxes:", keep)        # → [3, 0]  (D first by score, then best of A/B/C)
'''),
    md("""\
## 5. Toy mini-YOLO on synthetic data

We'll generate 100×100 images, each with a single white square at a random location. The model learns to predict where the square is. Real YOLO is more complex (multi-scale features, focal loss, anchor-free distribution prediction), but this captures the *idea*.
"""),
    code('''\
import torch, torch.nn as nn, torch.nn.functional as F

def make_synthetic(n=400, size=64):
    imgs = np.zeros((n, 1, size, size), dtype=np.float32)
    labels = np.zeros((n, 4), dtype=np.float32)  # xc, yc, w, h (normalised)
    for k in range(n):
        w = np.random.randint(8, 20)
        h = np.random.randint(8, 20)
        x1 = np.random.randint(0, size - w)
        y1 = np.random.randint(0, size - h)
        imgs[k, 0, y1:y1+h, x1:x1+w] = 1.0
        labels[k] = [(x1 + w/2) / size, (y1 + h/2) / size, w / size, h / size]
    return imgs, labels

X_syn, y_syn = make_synthetic(400)
print("X:", X_syn.shape, " y:", y_syn.shape, " first label:", y_syn[0])

# Tiny CNN that regresses 4 numbers per image — a *simplified* detector
# (real YOLO has the grid + class outputs; we predict the single object's xywh).
class MiniDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(64, 4)

    def forward(self, x):
        return torch.sigmoid(self.head(self.body(x).flatten(1)))   # outputs in [0,1]

model = MiniDetector()
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
Xt = torch.tensor(X_syn); yt = torch.tensor(y_syn)

for ep in range(50):
    pred = model(Xt)
    loss = F.mse_loss(pred, yt)
    opt.zero_grad(); loss.backward(); opt.step()
    if (ep + 1) % 10 == 0:
        print(f"epoch {ep+1}  mse={loss.item():.4f}")

# Visualise a few predictions
import matplotlib.patches as patches
fig, axes = plt.subplots(1, 4, figsize=(12, 3))
model.eval()
with torch.no_grad():
    preds = model(Xt[:4]).numpy()
for ax, img, gt, pr in zip(axes, X_syn[:4, 0], y_syn[:4], preds):
    ax.imshow(img, cmap="gray")
    for box, color in [(gt, "green"), (pr, "red")]:
        xc, yc, w, h = box * 64
        ax.add_patch(patches.Rectangle((xc-w/2, yc-h/2), w, h,
                                        fill=False, edgecolor=color, lw=2))
    ax.set_title("green=GT  red=pred"); ax.axis("off")
plt.show()
'''),
    md("""\
**Real YOLO** scales this idea with:
- Multi-scale feature maps (predict at 80×80, 40×40, 20×20 to catch big and small objects)
- Per-cell objectness + class probabilities
- Anchor-free distribution focal loss (YOLOv8)
- Heavy augmentation (MOSAIC, MixUp)

But the *core* — grid cells predicting box+class, with IoU + NMS at inference — is exactly what we just built.

**Next:** optimisation tricks (Adam, batch norm, dropout) that make any of these models actually trainable.
"""),
])


# ===========================================================================
# Notebook 06 — Optimisation + training
# ===========================================================================

NB_06 = notebook([
    md("""\
# 06 — Optimisers, schedules, regularisation

## Roadmap
1. SGD — the simplest possible optimiser, math + code
2. SGD with **momentum**
3. **Adam** — adaptive learning rates per parameter
4. **Learning-rate schedules** — warmup, cosine, step decay
5. **Batch normalization** — what it does, why it helps
6. **Dropout** + **L2 regularisation**
7. When to use which
"""),
    BOOTSTRAP,
    md("""\
## 1. Plain SGD

$$\\theta_{t+1} = \\theta_t - \\eta \\nabla \\mathcal{L}(\\theta_t)$$

Take a step in the direction of steepest descent, with step size $\\eta$ (learning rate).

Problem: noisy gradient causes wild oscillation when the loss surface is a ravine.
"""),
    code('''\
import numpy as np
import matplotlib.pyplot as plt

# Toy 2-D loss: a ravine
def loss(p): return 0.5 * (p[0]**2 + 10 * p[1]**2)
def grad(p): return np.array([p[0], 10 * p[1]])

def sgd_path(lr=0.1, n_steps=40):
    p = np.array([5.0, 1.5])
    hist = [p.copy()]
    for _ in range(n_steps):
        p = p - lr * grad(p)
        hist.append(p.copy())
    return np.array(hist)

path = sgd_path(lr=0.1)
fig, ax = plt.subplots(figsize=(7, 4))
X, Y = np.meshgrid(np.linspace(-6, 6, 100), np.linspace(-3, 3, 100))
Z = 0.5 * (X**2 + 10*Y**2)
ax.contour(X, Y, Z, levels=20, alpha=0.5)
ax.plot(path[:, 0], path[:, 1], "o-r", markersize=3)
ax.set_title("SGD bouncing across the ravine"); ax.grid(True)
plt.show()
'''),
    md("""\
## 2. Momentum

Instead of just taking the gradient as the step, accumulate a **velocity** that's an exponential moving average of past gradients:

$$v_{t+1} = \\mu v_t + \\nabla \\mathcal{L}(\\theta_t)$$
$$\\theta_{t+1} = \\theta_t - \\eta v_{t+1}$$

`μ` (typically 0.9) damps oscillation: gradients that flip back and forth average to zero; gradients that point consistently in one direction reinforce.
"""),
    code('''\
def sgd_momentum_path(lr=0.1, mu=0.9, n_steps=40):
    p = np.array([5.0, 1.5]); v = np.zeros_like(p)
    hist = [p.copy()]
    for _ in range(n_steps):
        v = mu * v + grad(p)
        p = p - lr * v
        hist.append(p.copy())
    return np.array(hist)

paths = {"SGD": sgd_path(lr=0.1), "SGD+momentum": sgd_momentum_path(lr=0.1)}
fig, ax = plt.subplots(figsize=(7, 4))
ax.contour(X, Y, Z, levels=20, alpha=0.5)
for name, p in paths.items():
    ax.plot(p[:, 0], p[:, 1], "o-", markersize=3, label=name)
ax.legend(); ax.set_title("Momentum smooths the path"); ax.grid(True); plt.show()
'''),
    md("""\
## 3. Adam — different learning rate per parameter

Adam keeps two moving averages: first moment (mean of gradients) and second moment (mean of *squared* gradients). It divides by `sqrt(second moment)` so parameters with consistently large gradients take **smaller** steps, and rare-but-important gradients get a chance to move.

$$m_t = \\beta_1 m_{t-1} + (1-\\beta_1) g_t$$
$$v_t = \\beta_2 v_{t-1} + (1-\\beta_2) g_t^2$$
$$\\theta_{t+1} = \\theta_t - \\eta \\frac{\\hat{m_t}}{\\sqrt{\\hat{v_t}} + \\epsilon}$$

(The hats are bias corrections you need only in the first few iterations.)
"""),
    code('''\
def adam_path(lr=0.3, b1=0.9, b2=0.999, eps=1e-8, n_steps=40):
    p = np.array([5.0, 1.5])
    m = np.zeros_like(p); v = np.zeros_like(p)
    hist = [p.copy()]
    for t in range(1, n_steps + 1):
        g = grad(p)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g**2
        m_hat = m / (1 - b1**t)
        v_hat = v / (1 - b2**t)
        p = p - lr * m_hat / (np.sqrt(v_hat) + eps)
        hist.append(p.copy())
    return np.array(hist)

paths = {"SGD": sgd_path(lr=0.1),
          "SGD+momentum": sgd_momentum_path(lr=0.1),
          "Adam": adam_path(lr=0.3)}
fig, ax = plt.subplots(figsize=(7, 4))
ax.contour(X, Y, Z, levels=20, alpha=0.5)
for name, p in paths.items():
    ax.plot(p[:, 0], p[:, 1], "o-", markersize=3, label=name)
ax.legend(); ax.set_title("Adam converges with much fewer steps"); plt.show()
'''),
    md("""\
**Rule of thumb:**

| Task | First try |
|---|---|
| Brand new model, you have no idea | **Adam(3e-4)** — almost always trains *something* |
| Image classifier, plenty of data | **SGD + momentum** with cosine schedule — often gets best final accuracy |
| Transformer / language model | **AdamW** (Adam with decoupled weight decay) |

## 4. Learning-rate schedules

Constant LR is rarely optimal. Two common schedules:

- **Step decay** — multiply LR by 0.1 every 30 epochs.
- **Cosine annealing** — smoothly decrease from `lr_max` to `lr_min` over the full run.
- **Warmup** — start tiny, ramp up linearly for a few hundred steps. Critical for very large LR.
"""),
    code('''\
def cosine_schedule(step, total_steps, lr_max=1.0, lr_min=0.0):
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * step / total_steps))

steps = np.arange(0, 200)
lrs = [cosine_schedule(s, 200, lr_max=1.0) for s in steps]
plt.figure(figsize=(7, 3))
plt.plot(steps, lrs); plt.title("cosine learning-rate schedule"); plt.grid(True); plt.show()
'''),
    md("""\
## 5. Batch normalization

For each mini-batch, normalize the activations of a layer to mean 0, var 1 (per channel for conv). Then apply a learnable scale and shift:

$$\\hat{x} = \\frac{x - \\mu_B}{\\sqrt{\\sigma_B^2 + \\epsilon}}, \\quad y = \\gamma \\hat{x} + \\beta$$

Effects:

1. Stabilizes gradient magnitudes → can use a higher learning rate.
2. Adds mild regularisation noise (the batch statistics are noisy).
3. Lets you train much deeper networks.

Used everywhere in CNNs since 2015.

## 6. Dropout and L2

**Dropout** — during training, randomly zero out fraction $p$ of activations. Forces the network to not rely on any single neuron.

**L2 weight decay** — add $\\lambda \\sum w^2$ to the loss. Pulls weights toward zero, prevents huge co-adapted weights.

Both reduce overfitting. Standard amounts: dropout 0.1–0.5, L2 1e-4 for vision models.

**Next:** measure how well any of these are actually working — metrics deep-dive.
"""),
])


# ===========================================================================
# Notebook 07 — Metrics
# ===========================================================================

NB_07 = notebook([
    md("""\
# 07 — Metrics + diagnostics

Every metric in the project, derived and implemented.

## Roadmap
1. Confusion matrix (the foundation of every classification metric)
2. Precision, recall, F1
3. ROC and PR curves, AUC, AP
4. mAP — what detection papers actually report
5. RMSE, MAE, MAPE, R² (regression)
6. **When to use which** — a decision table you can keep
"""),
    BOOTSTRAP,
    md("""\
## 1. Confusion matrix

For one class:

|  | predicted positive | predicted negative |
|---|---|---|
| true positive | **TP** | FN |
| true negative | FP | TN |
"""),
    code('''\
import numpy as np
y_true = np.array([1,1,1,1,0,0,0,0,0,0])
y_pred = np.array([1,1,1,0,1,0,0,0,0,1])

TP = ((y_pred==1) & (y_true==1)).sum()
FP = ((y_pred==1) & (y_true==0)).sum()
FN = ((y_pred==0) & (y_true==1)).sum()
TN = ((y_pred==0) & (y_true==0)).sum()
print(f"TP={TP}  FP={FP}  FN={FN}  TN={TN}")
'''),
    md("""\
## 2. Precision, Recall, F1

$$\\text{precision} = \\frac{TP}{TP + FP}, \\quad \\text{recall} = \\frac{TP}{TP + FN}$$

$$F_1 = \\frac{2 P R}{P + R}$$

Precision: "when I said yes, was I right?"  Recall: "of all the yeses, how many did I catch?"
"""),
    code('''\
P = TP / max(TP + FP, 1)
R = TP / max(TP + FN, 1)
F1 = 2 * P * R / max(P + R, 1e-9)
print(f"precision={P:.3f}  recall={R:.3f}  F1={F1:.3f}")
'''),
    md("""\
## 3. Precision–recall curve and Average Precision

The model usually outputs a probability, not a hard yes/no. The threshold for "yes" affects precision and recall. **AP** is the area under the precision–recall curve as you sweep the threshold.
"""),
    code('''\
import matplotlib.pyplot as plt

# 5 predictions with their probabilities, sorted by descending prob
probs  = np.array([0.95, 0.80, 0.65, 0.50, 0.20])
labels = np.array([1,    1,    0,    1,    0])      # ground truth

# Walk the threshold from high to low
prec_list, rec_list = [], []
tp = fp = 0
N = labels.sum()
for p, lab in sorted(zip(probs, labels), key=lambda kv: -kv[0]):
    if lab == 1: tp += 1
    else:        fp += 1
    prec_list.append(tp / (tp + fp))
    rec_list.append(tp / N)

# Trapezoidal AUC of PR curve = AP
AP = float(np.trapz(prec_list, rec_list))
plt.figure(figsize=(5, 4))
plt.plot(rec_list, prec_list, "o-"); plt.xlabel("recall"); plt.ylabel("precision")
plt.title(f"AP ≈ {AP:.3f}"); plt.xlim(0,1); plt.ylim(0,1.05); plt.grid(True); plt.show()
'''),
    md("""\
**mAP** = mean of AP across all classes. `mAP@0.5` means with IoU threshold 0.5 for matching predicted boxes to ground truth.

## 4. Regression metrics
"""),
    code('''\
y_true = np.array([100, 200, 300, 400], dtype=float)
y_pred = np.array([110, 180, 330, 360], dtype=float)
err = y_pred - y_true

RMSE = np.sqrt((err**2).mean())
MAE  = np.abs(err).mean()
MAPE = (np.abs(err) / np.abs(y_true)).mean() * 100

ss_res = (err**2).sum()
ss_tot = ((y_true - y_true.mean())**2).sum()
R2 = 1 - ss_res / ss_tot

print(f"RMSE = {RMSE:.2f}")
print(f"MAE  = {MAE:.2f}")
print(f"MAPE = {MAPE:.2f} %")
print(f"R²   = {R2:.4f}")
'''),
    md("""\
## 5. Decision table — which metric should I report?

| Situation | Metric |
|---|---|
| Balanced 2-class classification | accuracy is fine |
| Imbalanced 2-class (95% negative) | **F1** or **PR-AUC** — accuracy will look great even for a useless model |
| Multi-class | macro-F1 (average across classes) |
| Multi-label | per-class F1 + macro-F1 |
| Object detection | **mAP@0.5** and **mAP@0.5:0.95** |
| Regression for end-user dollars | **MAE** + **MAPE** (interpretable) |
| Regression where outliers matter | **RMSE** |
| Anything safety-critical | recall, specifically the false-negative rate |
"""),
])


# ===========================================================================
# Notebook 08 — Visualisation
# ===========================================================================

NB_08 = notebook([
    md("""\
# 08 — Visualising training so you can debug it

You can't fix what you can't see. Five plots every training loop should produce.

## Roadmap
1. Loss curves (train vs val) — overfit detector
2. **Learning-rate vs loss** — the LR-finder trick
3. **Gradient flow** — is every layer learning?
4. **Confusion matrix heatmap** — which classes the model confuses
5. **Activation maps** — what the model "looks at"
"""),
    BOOTSTRAP,
    md("""\
## 1. Loss curves — the #1 diagnostic
"""),
    code('''\
import numpy as np, matplotlib.pyplot as plt
np.random.seed(0)
epochs = np.arange(1, 31)
train_loss = 1.2 * np.exp(-0.15 * epochs) + 0.05 * np.random.randn(30) + 0.1
val_loss   = train_loss + 0.1 + 0.005 * (epochs ** 1.5)   # diverges late
plt.figure(figsize=(8, 4))
plt.plot(epochs, train_loss, label="train")
plt.plot(epochs, val_loss, label="val")
plt.axvline(np.argmin(val_loss), linestyle="--", color="red", alpha=0.5)
plt.text(np.argmin(val_loss)+0.5, 0.5, "early stop here", color="red")
plt.legend(); plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Classic overfit"); plt.show()
'''),
    md("""\
**What you're looking for:**
- Train ↓ steadily but val flat → underfit. Try bigger model, longer training, less regularisation.
- Both ↓ then val ↑ → overfit. Early-stop at val minimum, more augmentation, more dropout.
- Both ↓ together → still fitting. Keep training.

## 2. Learning-rate finder

Sweep LR from 1e-7 to 10 over a few hundred batches, plot loss vs LR. Pick the LR where loss drops fastest (typically ~1/10 of where it diverges).
"""),
    code('''\
# Synthesise a typical "LR finder" curve to demo
lrs = np.logspace(-7, 1, 200)
loss = 1 / (1 + lrs * 100) + 0.05 * np.random.randn(200)
loss[lrs > 1] = loss[lrs > 1] * (1 + lrs[lrs > 1])    # diverges at high LR
plt.figure(figsize=(8, 4))
plt.semilogx(lrs, loss); plt.axvspan(1e-3, 1e-1, alpha=0.2, color="green")
plt.xlabel("learning rate"); plt.ylabel("loss")
plt.title("Pick the LR where loss falls fastest (green band)"); plt.grid(True); plt.show()
'''),
    md("""\
## 3. Gradient-flow plot

For each layer, plot `||grad||`. If early layers are flat at 0, you have vanishing-gradient. If late layers spike to huge values, exploding-gradient.
"""),
    code('''\
# Synthetic example — a healthy vs unhealthy gradient flow
layers = list(range(1, 13))
healthy = np.exp(-0.05 * np.arange(12)) * (1 + 0.1*np.random.randn(12))
unhealthy = np.exp(-0.4 * np.arange(12))
fig, ax = plt.subplots(1, 2, figsize=(10, 3))
ax[0].bar(layers, healthy); ax[0].set_title("healthy: gradients ~uniform"); ax[0].set_ylim(0, 1.2)
ax[1].bar(layers, unhealthy); ax[1].set_title("vanishing: early layers near zero"); ax[1].set_ylim(0, 1.2)
for a in ax: a.set_xlabel("layer index"); a.set_ylabel("‖grad‖")
plt.show()
'''),
    md("""\
## 4. Confusion-matrix heatmap

For multi-class classification, this shows you which pairs of classes the model confuses. A diagonal-heavy matrix = good model. Off-diagonal hot spots = systematic confusion you can target with more data or better features.
"""),
    code('''\
# Synthetic 10-class confusion matrix
np.random.seed(0)
cm = np.zeros((10, 10), dtype=int)
for true in range(10):
    cm[true, true] = np.random.randint(40, 60)              # most correct
    for pred in range(10):
        if pred != true:
            cm[true, pred] = np.random.randint(0, 10)
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues")
plt.colorbar(label="count")
plt.xlabel("predicted"); plt.ylabel("true"); plt.title("Confusion matrix")
for i in range(10):
    for j in range(10):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                  color="white" if cm[i, j] > 30 else "black", fontsize=8)
plt.show()
'''),
    md("""\
## 5. Activation maps — what the CNN looks at

For a trained CNN, take an intermediate feature map and visualise it. Tells you whether the conv filters have actually learned anything semantic (edges, parts) or are stuck on noise.

(Real production tooling uses Grad-CAM and friends to highlight pixels that most contributed to a particular prediction. The principle: backprop the score of one class back to the input and visualise the gradient magnitude.)
"""),
    code('''\
# Toy demo: 4 conv kernels applied to a synthetic image
img = np.zeros((32, 32), dtype=np.float32)
img[8:16, 8:24] = 1.0   # horizontal bar
img[8:24, 14:18] = 1.0  # vertical bar — forms a "T"

kernels = {
    "vertical edge":   np.array([[-1, 0, 1]] * 3),
    "horizontal edge": np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
    "corner":          np.array([[ 0,  1, 0],
                                  [ 1, -4, 1],
                                  [ 0,  1, 0]]),
    "blur":            np.ones((3, 3)) / 9,
}

def conv2d_quick(im, k):
    from scipy.signal import correlate2d  # NB: only used here for speed
    return correlate2d(im, k, mode="same")

try:
    from scipy.signal import correlate2d
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    axes[0].imshow(img, cmap="gray"); axes[0].set_title("input"); axes[0].axis("off")
    for ax, (name, k) in zip(axes[1:], kernels.items()):
        ax.imshow(conv2d_quick(img, k.astype(np.float32)), cmap="RdBu"); ax.set_title(name); ax.axis("off")
    plt.show()
except ImportError:
    print("scipy not available; skipping plot.")
'''),
    md("""\
**Next:** the same models, now in PyTorch — line-by-line contrast.
"""),
])


# ===========================================================================
# Notebook 09 — PyTorch contrast (full pipeline on CarDD-style data)
# ===========================================================================

NB_09 = notebook([
    md("""\
# 09 — Full mini-pipeline in PyTorch (project-style)

We've built every piece in NumPy. Now we'll wire up a **complete** PyTorch pipeline that mirrors what the project's main code does, side-by-side with the from-scratch equivalents.

## Roadmap
1. `Dataset` + `DataLoader` — replaces our manual batching
2. `nn.Module` — replaces our class with manual `.forward()`
3. **Autograd** — replaces 200 lines of manual backprop with one `.backward()` call
4. `Optimizer` — Adam in one line
5. Full training loop end-to-end on a CarDD-style multi-label problem
6. Comparison table: lines of code, training time, final accuracy
"""),
    BOOTSTRAP,
    md("""\
## 1. Dataset class

Compare:

**From-scratch (NB 02):** we sliced raw NumPy arrays in a Python loop.
**PyTorch:** subclass `Dataset`, implement `__len__` and `__getitem__`, hand to `DataLoader` for batching + shuffling + multi-worker loading.
"""),
    code('''\
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

class CIFARMultiClass(Dataset):
    """Wrap torchvision's CIFAR-10 and apply our own transform."""

    def __init__(self, root, train=True, n=None):
        self.ds = datasets.CIFAR10(root=root, train=train, download=True)
        self.n = n if n is not None else len(self.ds)
        self.tfm = transforms.Compose([
            transforms.ToTensor(),                               # PIL → CHW float [0,1]
            transforms.Normalize(mean=[0.49, 0.48, 0.45],
                                  std= [0.25, 0.24, 0.26]),
        ])

    def __len__(self):  return self.n
    def __getitem__(self, idx):
        img, label = self.ds[idx]
        return self.tfm(img), label

train_ds = CIFARMultiClass("./_cifar_cache", train=True, n=2000)
test_ds  = CIFARMultiClass("./_cifar_cache", train=False, n=500)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False, num_workers=0)
print("train batches:", len(train_loader), " test batches:", len(test_loader))
'''),
    md("""\
## 2. `nn.Module` — a class with weight tracking built in

Same TinyCNN we built in NB 04, with one difference: PyTorch automatically tracks every `nn.*` member as a parameter so optimisers can find them all.
"""),
    code('''\
import torch.nn as nn
import torch.nn.functional as F

class TinyCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)                  # NB 06 idea — drop-in
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.fc    = nn.Linear(32 * 8 * 8, num_classes)
        self.drop  = nn.Dropout(0.25)                    # NB 06 idea

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = self.drop(x.flatten(1))
        return self.fc(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
model  = TinyCNN().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"{n_params:,} parameters, device: {device}")
'''),
    md("""\
## 3. Autograd

Every tensor remembers the operations that produced it (the "computation graph"). Calling `.backward()` walks that graph in reverse, multiplying local derivatives — *exactly the chain rule from NB 00*, automated.

**This is the single biggest reason PyTorch exists.** Without autograd you'd write 200 lines of conv-backward code per model. With autograd it's one line.
"""),
    code('''\
optim = torch.optim.Adam(model.parameters(), lr=3e-4)
crit  = nn.CrossEntropyLoss()

print("Training loop — note how short the per-step code is:")
print("    for xb, yb in train_loader:")
print("        optim.zero_grad()")
print("        loss = crit(model(xb), yb)")
print("        loss.backward()        # ← autograd does ALL the manual work from NB 02/04")
print("        optim.step()           # ← Adam from NB 06, one call")
'''),
    md("""\
## 4. Full training loop end-to-end
"""),
    code('''\
import time
hist = {"epoch": [], "train_loss": [], "test_acc": []}

t0 = time.time()
for ep in range(8):
    model.train()
    epoch_loss = 0; n = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optim.zero_grad()
        loss = crit(model(xb), yb)
        loss.backward()
        optim.step()
        epoch_loss += loss.item() * xb.size(0); n += xb.size(0)
    train_loss = epoch_loss / n
    # Eval
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb.to(device)).argmax(dim=-1)
            correct += (preds.cpu() == yb).sum().item()
            total += len(yb)
    test_acc = correct / total
    hist["epoch"].append(ep+1); hist["train_loss"].append(train_loss); hist["test_acc"].append(test_acc)
    print(f"epoch {ep+1}  loss={train_loss:.3f}  test_acc={test_acc:.3f}")
print(f"\\ntotal: {time.time()-t0:.1f}s")
'''),
    md("""\
## 5. Plot training curves
"""),
    code('''\
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(10, 3))
ax[0].plot(hist["epoch"], hist["train_loss"]); ax[0].set_title("train loss"); ax[0].grid(True)
ax[1].plot(hist["epoch"], hist["test_acc"]);   ax[1].set_title("test acc");   ax[1].grid(True)
plt.show()
'''),
    md("""\
## 6. Comparison: NumPy vs PyTorch

| Aspect | From-scratch (NB 02–08) | PyTorch (this notebook) |
|---|---|---|
| Lines for MLP forward | ~10 (manual) | ~3 in `nn.Module` |
| Lines for MLP backward | ~15 (manual derivatives) | **0** (autograd) |
| Lines for conv2d | ~25 | ~1 (`nn.Conv2d`) |
| Lines for Adam | ~10 (math) | 1 (`torch.optim.Adam`) |
| GPU support | re-derive everything | `.to('cuda')` |
| Mini-batching | manual loop | `DataLoader` |
| Data augmentation | hand-written | `torchvision.transforms` |
| Mixed precision | infeasible | `torch.cuda.amp` decorator |
| Save / load model | pickle dict | `torch.save / load` |

**When to drop down to manual code:**
- Teaching / understanding (this whole series).
- Custom layer types not in `nn.*`.
- Research papers with novel ops.
- Constrained edge environments without PyTorch.

For 99% of real applications: use PyTorch, but **knowing what it's doing under the hood is the difference between using it and debugging it.**

## 7. Applying this to the project

The project's actual classifier ([src/ccdp/models/damage_classifier.py](../../src/ccdp/models/damage_classifier.py)) and trainer ([src/ccdp/train/train_damage_classifier.py](../../src/ccdp/train/train_damage_classifier.py)) use the *same primitives* you saw in this notebook:
- `torchvision.models.resnet50(pretrained=True)` — pre-built ResNet50 (the 50-layer residual network from NB 04, scaled up).
- `nn.BCEWithLogitsLoss(pos_weight=...)` — multi-label loss with per-class weighting.
- `torch.optim.AdamW` — Adam + decoupled weight decay (the modern default).
- Two-stage fine-tuning — freeze the backbone, train the head, then unfreeze.

You can now read that code and understand every line.
"""),
])
