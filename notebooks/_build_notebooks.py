"""One-shot generator for the beginner notebooks under ``notebooks/``.

We build `.ipynb` JSON programmatically because hand-writing notebook JSON is
error-prone and easy to break. Run once to (re)materialise the notebooks:

    python notebooks/_build_notebooks.py

The notebooks are committed; this generator stays in-tree so they can be
regenerated if we want to tweak content later.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


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
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# ---------------------------------------------------------------------------
# Shared first cells — every notebook starts with these so it runs on Colab
# (clones the repo, installs requirements, sets sys.path) AND on local clones.
# ---------------------------------------------------------------------------

COLAB_BOOTSTRAP = code('''\
# === Colab bootstrap ===
# Safe to re-run. On a local clone with `pip install -e .` already done this
# is a no-op; on Colab it clones the repo + installs deps the first time.
import os, sys, subprocess
from pathlib import Path

REPO_URL = "https://github.com/theDocWho/car-crash-fix-amount-predictor.git"
REPO_DIR = Path("car-crash-fix-amount-predictor")

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB and not REPO_DIR.exists():
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL], check=True)
if IN_COLAB:
    os.chdir(REPO_DIR)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "-r", "requirements.txt"], check=True)

# Make `ccdp` importable whether or not the package was installed editable.
src_path = Path("src").resolve()
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

print("ccdp path:", src_path)
print("running in Colab:", IN_COLAB)
''')


# ===========================================================================
# Notebook 00 — Overview
# ===========================================================================

NB_00 = notebook([
    md("""\
# 00 — Project overview

**Audience:** complete beginner. We will *not* assume you know what a neural network is.

This series of notebooks teaches the **car-crash-fix-amount-predictor** project from first principles. By the end you will be able to:

1. Explain what every model in the pipeline does and why it exists.
2. Read and modify the training code.
3. Run a small training job on Google Colab.
4. Understand every number in the evaluation report.

## The problem we are solving

> Given a photo of a damaged car, predict (a) what kind of damage it is, (b) where on the car it is, and (c) roughly how much it will cost to repair.

A real insurance assessor does this with a clipboard and 15 years of experience. We will do it with three machine-learning models stacked together.

## The pipeline at a glance

```
    [User uploads photo]
            |
            v
    +----------------+
    |  Preprocess    |   Downscale + quality score (no ML)
    +----------------+
            |
            +----------------------+----------------------+
            v                      v                      v
    +-------------+        +-------------+        +-------------+
    |  ResNet50   |        |   YOLOv8    |        |  ResNet50   |
    |  classifier |        |  detector   |        |  identifier |
    |  (what?)    |        |  (where?)   |        | (car make?) |
    +-------------+        +-------------+        +-------------+
            \\                    /                    /
             \\                  /                    /
              v                v                    v
            +------------------------------------+
            |  XGBoost cost regressor (how much?)|
            +------------------------------------+
                            |
                            v
                  +-----------------+
                  | Catalog lookup  |   Fallback if XGBoost can't trust input
                  +-----------------+
                            |
                            v
                    Final $ estimate
```

## What each notebook covers

| # | Notebook | What you'll learn |
|---|---|---|
| 01 | Data & preprocessing | Where the data comes from, what cleaning is needed, why we downscale, how to measure blur |
| 02 | ResNet50 classifier | What a CNN is, why ResNet is *residual*, two-stage transfer learning |
| 03 | YOLOv8 detector | How a model predicts boxes, IoU, NMS, mAP — all with diagrams |
| 04 | XGBoost cost regressor | What gradient boosting is, our calibration trick |
| 05 | Metrics deep dive | Precision, recall, F1, mAP, RMSE, MAE, MAPE, R² with worked examples |
| 06 | End-to-end inference | Load the production weights and run a real prediction |

Run them in order — each builds on the last.
"""),
    COLAB_BOOTSTRAP,
    md("""\
## Verify the install

The cell below imports the project and prints which models are reachable. If any line errors, the bootstrap above did not finish — re-run it.
"""),
    code('''\
# Smoke-test imports — if these all succeed, the project is wired up correctly.
import ccdp
from ccdp.preprocess import preprocess
from ccdp.data.schema import DAMAGE_TYPES
from ccdp.viz import annotate_prediction

print("DAMAGE_TYPES the models can output:", DAMAGE_TYPES)
print()
print("ccdp imported from:", Path(ccdp.__file__).parent)
'''),
    md("""\
## Visualising the data flow as a diagram

Below is the same pipeline drawn with `matplotlib`. We will redraw versions of this diagram in every notebook so you build a mental picture of how the pieces fit.
"""),
    code('''\
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")

boxes = [
    (0.5, 2,  "Image",        "#bbdefb"),
    (2.2, 2,  "Preprocess",   "#c8e6c9"),
    (4.0, 3.5,"ResNet50\\nclassifier", "#fff59d"),
    (4.0, 2,  "YOLOv8\\ndetector",    "#ffcc80"),
    (4.0, 0.5,"ResNet50\\nidentifier","#ce93d8"),
    (6.5, 2,  "XGBoost\\ncost",       "#ef9a9a"),
    (8.5, 2,  "$ estimate",          "#80cbc4"),
]
for x, y, t, c in boxes:
    ax.add_patch(patches.FancyBboxPatch((x, y-0.4), 1.4, 0.8,
                 boxstyle="round,pad=0.05", facecolor=c, edgecolor="black"))
    ax.text(x+0.7, y, t, ha="center", va="center", fontsize=10)
# arrows
for (x1, y1), (x2, y2) in [
    ((1.9, 2),(2.2,2)), ((3.6,2),(4.0,2)),
    ((3.6,2),(4.0,3.5)),((3.6,2),(4.0,0.5)),
    ((5.4,3.5),(6.5,2.2)),((5.4,2),(6.5,2)),((5.4,0.5),(6.5,1.8)),
    ((7.9,2),(8.5,2)),
]:
    ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle="->", lw=1.2))
plt.title("ccdp pipeline — what each notebook covers")
plt.show()
'''),
    md("""\
**Next:** open `01_data_and_preprocessing.ipynb` to learn where the training data comes from and why preprocessing matters before you ever touch a neural network.
"""),
])


# ===========================================================================
# Notebook 01 — Data & preprocessing
# ===========================================================================

NB_01 = notebook([
    md("""\
# 01 — Data and preprocessing

Before any model can learn, the data has to be *clean*, *consistent*, and *the right size*. This notebook walks through every preprocessing step the project performs, the math behind each, and how to verify it with code.

## Roadmap

1. The datasets we use, and why
2. The shape of one training example
3. **Why downscale?** — memory math
4. **How we downscale** — LANCZOS resampling explained
5. **Quality scoring** — variance-of-Laplacian for blur, brightness, contrast
6. Augmentation: RandAugment, MixUp, CutMix — visualised
"""),
    COLAB_BOOTSTRAP,
    md("""\
## 1. Datasets

| Dataset | Used for | Source |
|---|---|---|
| **CarDD** (Wang et al. 2023) | Damage classifier + detector training | Kaggle: `eduardo4jesus/cardd` |
| **Stanford Cars** | Car make/model identifier | Kaggle: `eduardo4jesus/stanford-cars-dataset` |

The cost target is **synthetic** — derived from a versioned parts-cost catalog × car age × a small Gaussian noise term. There is no public dataset that pairs damaged-car photos with real repair invoices, and inventing one would be dishonest. We document this trade-off in `PLAN.md §3`.

## 2. One training example

```
{
  "image":     <PIL.Image  H×W×3 RGB>,
  "damage":    ["dent", "scratch"],          # multi-label
  "bboxes":    [BBox(damage_type="dent", x_center=..., y_center=..., width=..., height=...)],
  "make":      "toyota",
  "model":     "camry",
  "year":      2015,
  "cost_usd":  812.50,                       # synthetic target
}
```
"""),
    code('''\
# Peek at the schema in code form.
from ccdp.data.schema import DAMAGE_TYPES, BBox, infer_part_from_damage
import inspect

print("Damage classes:", DAMAGE_TYPES)
print()
print("BBox fields:")
print(inspect.getsource(BBox))
'''),
    md("""\
## 3. Why downscale?

A modern phone shoots photos at ~12 megapixels. Stored as RGB uint8 that is:

$$
12{,}000{,}000 \\text{ pixels} \\times 3 \\text{ channels} \\times 1 \\text{ byte} = 36 \\text{ MB per image}
$$

For a batch of 32, that's **1.15 GB** of pixels — more than free-Colab T4 VRAM headroom after the model is loaded. And **the model never sees that resolution anyway**: ResNet50 inputs are 224×224, YOLOv8 is 640×640. So we resize once, up front, to a long edge of **1600 px**.
"""),
    code('''\
# Concrete memory comparison.
def bytes_for(w, h, c=3, dtype_bytes=1):
    return w * h * c * dtype_bytes

raw_phone   = bytes_for(4000, 3000)
downscaled  = bytes_for(1600, 1200)
classifier  = bytes_for(224, 224)
detector    = bytes_for(640, 640)

print(f"Raw phone photo (4000x3000): {raw_phone / 1e6:6.1f} MB")
print(f"Downscaled  (1600x1200):     {downscaled / 1e6:6.1f} MB")
print(f"Classifier input (224x224):  {classifier / 1e3:6.1f} KB")
print(f"Detector input (640x640):    {detector / 1e3:6.1f} KB")
'''),
    md("""\
## 4. How we downscale — LANCZOS resampling

When you shrink an image, you have to pick which pixels of the original to keep. The naive way is **nearest-neighbour**: for each output pixel, copy the closest input pixel. It's fast but creates jagged edges — bad for our models, which are trying to detect dent and scratch *edges*.

**LANCZOS** is a higher-order filter that blends multiple input pixels using the function

$$
L(x) = \\begin{cases} \\text{sinc}(x)\\,\\text{sinc}(x/a) & |x| < a \\\\ 0 & \\text{otherwise}\\end{cases}
$$

where $\\text{sinc}(x) = \\sin(\\pi x) / (\\pi x)$ and $a$ (the *kernel size*) is typically 3. The result preserves edges better than nearest-neighbour or bilinear, which is why Pillow recommends it for downsizing.

Let's see the difference visually.
"""),
    code('''\
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Build a synthetic high-frequency test pattern (alternating black/white bars).
big = np.zeros((400, 400, 3), dtype=np.uint8)
big[:, ::4] = 255  # vertical stripes every 4 pixels
img = Image.fromarray(big)

nearest = img.resize((100, 100), resample=Image.NEAREST)
bilinear = img.resize((100, 100), resample=Image.BILINEAR)
lanczos = img.resize((100, 100), resample=Image.LANCZOS)

fig, axes = plt.subplots(1, 4, figsize=(12, 4))
for ax, im, title in zip(axes, [img, nearest, bilinear, lanczos],
                          ["original 400x400", "NEAREST", "BILINEAR", "LANCZOS"]):
    ax.imshow(im); ax.set_title(title); ax.axis("off")
plt.show()
'''),
    md("""\
LANCZOS produces the smoothest stripes — the others alias into Moiré patterns. Now let's call the project's own downscaler.
"""),
    code('''\
from ccdp.preprocess.pipeline import normalize_for_inference, quality_report
from PIL import Image
import numpy as np

# Make a fake "phone photo" — 4000x3000 random noise just for size demo.
fake = Image.fromarray(np.random.randint(0, 255, (3000, 4000, 3), dtype=np.uint8))
print("input :", fake.size)
small = normalize_for_inference(fake, max_long_edge=1600)
print("output:", small.size)
print("note: long edge clamped to 1600, aspect ratio preserved.")
'''),
    md("""\
## 5. Quality scoring — is the image even usable?

If a user uploads a phone photo so blurry that even a human can't tell a dent from a reflection, the model will guess wildly and the cost estimate will be junk. So we **measure** image quality and surface it in the API response.

### Variance of the Laplacian (the standard blur metric)

The Laplacian is a 2D second-derivative operator. Apply this 3×3 kernel as a convolution to a greyscale image:

$$
\\nabla^2 = \\begin{bmatrix} 0 & 1 & 0 \\\\ 1 & -4 & 1 \\\\ 0 & 1 & 0 \\end{bmatrix}
$$

For a sharp image with crisp edges, the result has *high variance* — lots of strongly-positive and strongly-negative responses near edges. A blurry image has *low variance* because edges are smeared.

This single number — `Var(∇² image)` — is the textbook "is it blurry?" metric (Pech-Pacheco et al., 2000).

Let's reproduce it.
"""),
    code('''\
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from ccdp.preprocess.pipeline import _sharpness_score, quality_report

# Build a sharp image (edges) and a blurry copy of the same.
sharp_arr = np.zeros((256, 256), dtype=np.uint8)
sharp_arr[64:192, 64:192] = 255    # crisp white square on black
sharp = Image.fromarray(sharp_arr).convert("RGB")

blurry = sharp.filter(__import__("PIL.ImageFilter", fromlist=["GaussianBlur"]).GaussianBlur(radius=8))

print(f"Sharpness  sharp: {_sharpness_score(sharp):8.1f}")
print(f"Sharpness blurry: {_sharpness_score(blurry):8.1f}")

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(sharp); axes[0].set_title("sharp"); axes[0].axis("off")
axes[1].imshow(blurry); axes[1].set_title("blurry"); axes[1].axis("off")
plt.show()
'''),
    code('''\
# The full quality report the API returns:
quality_report(sharp)
'''),
    md("""\
## 6. Augmentation — making models robust

A model that has only ever seen well-lit, centered, crisp photos will fail on real user uploads. We *augment* the training set by randomly distorting each image differently each epoch. Three popular techniques:

- **RandAugment** — pick N random transforms (rotate, color jitter, posterize…) from a list and chain them. Reduces hyperparameter tuning.
- **MixUp** — for two images $x_i, x_j$ with labels $y_i, y_j$, train on $(\\lambda x_i + (1-\\lambda) x_j,\\ \\lambda y_i + (1-\\lambda) y_j)$ where $\\lambda \\sim \\text{Beta}(\\alpha, \\alpha)$.
- **CutMix** — paste a random rectangle from image B onto image A; mix labels by the area ratio.

The intuition: these force the model to make decisions based on *content* rather than memorising exact pixel arrangements.
"""),
    code('''\
# Visualise MixUp on two synthetic images.
import numpy as np
import matplotlib.pyplot as plt

img_a = np.zeros((128, 128, 3), dtype=np.float32); img_a[..., 0] = 1.0   # red
img_b = np.zeros((128, 128, 3), dtype=np.float32); img_b[..., 2] = 1.0   # blue

fig, axes = plt.subplots(1, 5, figsize=(14, 3))
for ax, lam in zip(axes, [1.0, 0.75, 0.5, 0.25, 0.0]):
    mix = lam * img_a + (1 - lam) * img_b
    ax.imshow(mix); ax.set_title(f"λ={lam}"); ax.axis("off")
plt.suptitle("MixUp: linear blend between two images and their labels")
plt.show()
'''),
    md("""\
**Next:** open `02_classifier_resnet50.ipynb` to learn what a convolutional network actually computes, why ResNet50 has *skip connections*, and how to train one yourself.
"""),
])


# ===========================================================================
# Notebook 02 — ResNet50 classifier
# ===========================================================================

NB_02 = notebook([
    md("""\
# 02 — The ResNet50 damage classifier

Goal: given an image, output a probability for each of the 6 damage types in `DAMAGE_TYPES`. This is a **multi-label** problem — one image can have both a dent *and* a scratch.

## Roadmap

1. What is a convolution? (with math + a tiny worked example)
2. From convolution → CNN → ResNet → ResNet50
3. The **residual** trick — why skip connections matter
4. Transfer learning: stage 1 (freeze) and stage 2 (fine-tune)
5. Loss function: binary cross-entropy for multi-label
6. **Runnable** demo-scale training cell (≤ 10 min on Colab)
7. **Optional** full-training cell (multi-hour)
"""),
    COLAB_BOOTSTRAP,
    md("""\
## 1. What is a convolution?

Imagine you have a 5×5 greyscale image and a 3×3 *kernel* (a tiny matrix of weights). You slide the kernel over the image, and at each position you compute one number:

$$
(I * K)(x, y) = \\sum_{i=-1}^{1} \\sum_{j=-1}^{1} I(x+i, y+j) \\cdot K(i, j)
$$

That's it. Convolution is just "weighted sum of a neighborhood, repeated everywhere."

The magic: **the same kernel weights are used at every spatial position**. So a kernel that learns to detect "vertical edge" detects vertical edges anywhere in the image — top-left, bottom-right, anywhere. This is called *translation equivariance* and it is why CNNs work for images.
"""),
    code('''\
import numpy as np
import matplotlib.pyplot as plt

# Vertical edge kernel — the classic "Sobel-x".
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)

# A test image: a sharp vertical edge in the middle.
img = np.zeros((9, 9), dtype=np.float32)
img[:, 4:] = 1.0

def conv2d(I, K):
    h, w = I.shape; kh, kw = K.shape
    out = np.zeros((h-kh+1, w-kw+1), dtype=np.float32)
    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            out[y, x] = (I[y:y+kh, x:x+kw] * K).sum()
    return out

response = conv2d(img, sobel_x)
print("response (3x3 sliding sum):"); print(response)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(img, cmap="gray"); axes[0].set_title("input"); axes[0].axis("off")
axes[1].imshow(response, cmap="RdBu"); axes[1].set_title("Sobel-x response"); axes[1].axis("off")
plt.show()
'''),
    md("""\
Notice the response is **near zero** everywhere except *at the edge*, where it spikes. A trained CNN learns kernels like this — but for "is this part of a dent?" instead of "is this a vertical edge?"

## 2. From convolution → CNN → ResNet50

A CNN stacks many convolutions, each followed by:
- **Activation** (ReLU: $f(x) = \\max(0, x)$ — keeps the network nonlinear)
- **Pooling** (downsample, e.g. max-pool 2×2 → halves spatial size)
- **Batch normalisation** (normalises layer outputs to mean 0, var 1; speeds training)

Early layers learn edges and color blobs. Middle layers learn textures and shapes. Late layers learn object parts.

**ResNet50** is a 50-layer CNN published by He et al. (2015). The "50" is the depth in weight layers.

## 3. The residual trick — why ResNet works at all

Naively stacking 50 layers does *not* work — training accuracy actually gets **worse** as you go deeper, because gradients vanish or explode during backprop.

ResNet's fix: in each block, instead of computing $y = F(x)$, compute

$$
y = F(x) + x
$$

The "+x" is the **skip connection** (or **residual** connection). The block now learns the *residual* $F(x) = y - x$ — the *change* you want to add to the input, not the whole output.

Why this helps:
- If the optimal $F$ is "do nothing", the network can drive $F(x) \\to 0$ trivially. Without the skip, it would have to learn the identity function across all 50 layers — much harder.
- Gradients can flow straight back through the skip path, bypassing the deep layers, which keeps gradient magnitudes alive.
"""),
    code('''\
# Visualise a residual block as a tiny diagram.
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(8, 4))
ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
ax.add_patch(patches.FancyBboxPatch((0.5, 1.5), 1.2, 1, boxstyle="round,pad=0.05",
                                     facecolor="#bbdefb"))
ax.text(1.1, 2, "x", ha="center", va="center", fontsize=12)
for x, label, color in [(3, "conv\\n3x3", "#c8e6c9"),
                         (5, "ReLU",     "#fff59d"),
                         (7, "conv\\n3x3", "#c8e6c9")]:
    ax.add_patch(patches.FancyBboxPatch((x, 1.5), 1.2, 1,
                  boxstyle="round,pad=0.05", facecolor=color))
    ax.text(x+0.6, 2, label, ha="center", va="center", fontsize=10)
# main path arrows
for a, b in [((1.7,2),(3,2)),((4.2,2),(5,2)),((6.2,2),(7,2)),((8.2,2),(9.0,2))]:
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", lw=1.2))
# skip arc
ax.annotate("", xy=(9.0, 2), xytext=(1.7, 2),
            arrowprops=dict(arrowstyle="->", lw=1.5,
                             connectionstyle="arc3,rad=-0.4", color="red"))
ax.text(5, 3.4, "skip / residual (red): y = F(x) + x",
        ha="center", color="red", fontsize=11)
ax.text(9.6, 2, "y", fontsize=12, va="center")
plt.show()
'''),
    md("""\
## 4. Transfer learning — two stages

Training a 50-layer CNN from scratch needs millions of images. We have a few thousand. So we **transfer**:

1. Download ResNet50 weights pretrained on **ImageNet** (1.3M generic photos, 1000 classes). The early layers already know edges, textures, shapes — *generic* visual features.
2. Replace only the final classification head (1000-way → 6-way for our damage types).
3. **Stage 1** — freeze the backbone, train only the head for ~5 epochs. Fast — the head has very few weights.
4. **Stage 2** — unfreeze the top half of the backbone, train end-to-end at a *much lower* learning rate. Fine-tunes the high-level features to our domain (cars).

Why two stages? If you skip stage 1 and immediately fine-tune everything, the random head produces large gradient signals that smash the carefully-tuned backbone weights. Stage 1 lets the head learn first; stage 2 then nudges everything gently.
"""),
    code('''\
# Look at the model the project actually trains.
from ccdp.models.damage_classifier import build_damage_classifier
m = build_damage_classifier(num_classes=6, pretrained=False)
print(type(m).__name__)
total = sum(p.numel() for p in m.parameters())
trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
print(f"Total params: {total:,}")
print(f"Trainable params: {trainable:,}")
'''),
    md("""\
## 5. Loss function for multi-label classification

For *single-label* (one class per image) we use **cross-entropy**. For *multi-label* (independent yes/no per class) we use **binary cross-entropy per class**:

$$
\\mathcal{L} = -\\sum_{c=1}^{C} \\big[\\,y_c \\log \\hat{y}_c + (1 - y_c) \\log (1 - \\hat{y}_c)\\,\\big]
$$

where $y_c \\in \\{0, 1\\}$ is the true label for class $c$ and $\\hat{y}_c \\in (0, 1)$ is the sigmoid-activated logit.

The intuition: each class gets its own little binary classifier ("is this a dent? yes/no") and they all contribute independently to the total loss.
"""),
    code('''\
# Demo BCE loss math by hand.
import numpy as np

# Ground truth: image has dent + scratch but no crack.
y_true = np.array([1, 1, 0])
# Model says: 0.9 dent, 0.4 scratch (too uncertain), 0.05 crack.
y_pred = np.array([0.9, 0.4, 0.05])

eps = 1e-9  # numerical guard
bce_per_class = -(y_true * np.log(y_pred + eps) + (1-y_true) * np.log(1-y_pred + eps))
print("BCE per class:", bce_per_class.round(4))
print("Total BCE   :", bce_per_class.sum().round(4))
print()
print("Notice scratch (0.4) dominates the loss — that's the class the model")
print("is least confident about, and it's the one the gradient will push hardest.")
'''),
    md("""\
## 6. Demo-scale training (runnable in ≤ 10 min on Colab)

We'll train for **just 2 epochs on a tiny synthetic subset** so you see the training loop end-to-end. This will NOT produce a competitive model — production weights come from the multi-hour run committed as v0.1.0.
"""),
    code('''\
# Synthetic mini-batch — randomly generated tensors, not real data.
# This is enough to show the optimiser stepping and the loss going down.
import torch, torch.nn as nn
from ccdp.models.damage_classifier import build_damage_classifier
from ccdp.utils import pick_device

device = pick_device()
print("device:", device)

model = build_damage_classifier(num_classes=6, pretrained=False).to(device)
optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
crit  = nn.BCEWithLogitsLoss()

# 16 fake images, 6 random binary labels each.
torch.manual_seed(0)
x = torch.randn(16, 3, 224, 224, device=device)
y = (torch.rand(16, 6, device=device) > 0.5).float()

for epoch in range(2):
    optim.zero_grad()
    logits = model(x)
    loss = crit(logits, y)
    loss.backward()
    optim.step()
    print(f"epoch {epoch+1}  loss={loss.item():.4f}")
print("\\nThe loss came down — training loop works end-to-end.")
'''),
    md("""\
## 7. Optional — full training on the real dataset

Uncomment the cell below to launch the **production** training script. This is what produced the weights packaged in the GitHub Release. On Colab T4 it takes **~3 hours**; on a free CPU instance it will not finish in a session.

You will need to upload your Kaggle API token first — see the dataset-prep section of notebook 01.
"""),
    code('''\
# Uncomment to run the real training. Keeps the demo notebook safe to "Run All".
#
# from ccdp.train.classifier import train_classifier
# train_classifier(
#     epochs_stage_1=5,
#     epochs_stage_2=15,
#     batch_size=32,
#     learning_rate_stage_1=1e-3,
#     learning_rate_stage_2=1e-4,
#     output_dir="checkpoints/classifier_full",
# )
print("(commented out — uncomment to run multi-hour training)")
'''),
    md("""\
**Next:** notebook 03 covers the **YOLOv8 detector** — how a model predicts boxes, what IoU and NMS mean, and how mAP is computed.
"""),
])


# ===========================================================================
# Notebook 03 — YOLOv8 detector
# ===========================================================================

NB_03 = notebook([
    md("""\
# 03 — The YOLOv8 damage detector

Classification answers "**what** is in this image". Detection answers "**what AND where**". The output is a list of bounding boxes, each with a class label and a confidence score.

## Roadmap

1. What is a bounding box? (xyxy vs xywh, normalised vs pixel)
2. **IoU** (Intersection-over-Union) — with diagram + math
3. **NMS** (Non-Max Suppression) — how YOLO picks the best box
4. **Anchor-free prediction** — what YOLOv8 actually outputs at every grid cell
5. **mAP** — the evaluation metric, step by step
6. Runnable demo-scale training
7. Visualise predictions on a real image
"""),
    COLAB_BOOTSTRAP,
    md("""\
## 1. Bounding-box conventions

A box can be written as:

- **xyxy**: `(x1, y1, x2, y2)` — top-left and bottom-right corners.
- **xywh**: `(xc, yc, w, h)` — center + width + height.

Both can be in **absolute pixel** coordinates or **normalised** to `[0, 1]` relative to image size.

YOLO trains and predicts in **xywh-normalised**. Our `DetectedBox.xywh_norm` follows the same convention.
"""),
    code('''\
def xywh_to_xyxy(xc, yc, w, h):
    return (xc - w/2, yc - h/2, xc + w/2, yc + h/2)

print(xywh_to_xyxy(0.5, 0.5, 0.4, 0.6))   # normalised
'''),
    md("""\
## 2. IoU — Intersection over Union

How "close" are two boxes? IoU is the standard answer:

$$
\\text{IoU}(A, B) = \\frac{|A \\cap B|}{|A \\cup B|}
$$

- 1.0 means perfect overlap.
- 0.0 means disjoint.
- 0.5 is the typical threshold for "this prediction matches that ground-truth box".
"""),
    code('''\
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def iou(a, b):
    """IoU for boxes in xyxy form."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter + 1e-9)

box_gt   = (1, 1, 5, 5)        # ground truth
box_pred = (3, 2, 7, 6)        # prediction
print(f"IoU = {iou(box_gt, box_pred):.3f}")

fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xlim(0, 8); ax.set_ylim(0, 8); ax.set_aspect("equal")
ax.add_patch(patches.Rectangle((1,1), 4, 4, fill=False, edgecolor="green", lw=2, label="ground truth"))
ax.add_patch(patches.Rectangle((3,2), 4, 4, fill=False, edgecolor="red",   lw=2, label="prediction"))
ax.legend(); ax.grid(True)
plt.title(f"IoU = {iou(box_gt, box_pred):.3f}")
plt.show()
'''),
    md("""\
## 3. NMS — Non-Max Suppression

YOLO predicts *lots* of boxes per image — often several overlapping boxes around the same dent. NMS keeps the most confident one and suppresses the rest:

```
1. Sort all predictions by confidence (descending).
2. Take the top box; add it to the keep list.
3. Discard all remaining boxes that overlap it with IoU > threshold (e.g. 0.5).
4. Repeat until no boxes left.
```

This is **per-class** — a dent box doesn't suppress a scratch box even if they overlap.
"""),
    code('''\
def nms(boxes, scores, iou_thresh=0.5):
    """Boxes: list of xyxy. Returns indices to keep."""
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if iou(boxes[i], boxes[j]) < iou_thresh]
    return keep

# 3 overlapping detections of the same dent at different confidences.
boxes = [(2,2,6,6), (2.5,2.5,6.5,6.5), (2.2,1.8,6.2,5.8)]
scores = [0.7, 0.9, 0.6]
print("kept indices:", nms(boxes, scores))
print("→ NMS keeps box index 1 (the most-confident one); discards 0 and 2.")
'''),
    md("""\
## 4. What does YOLOv8 actually predict?

YOLOv8 is **anchor-free**. The image is divided into a grid (e.g. 80×80 at one of three scales). At every grid cell, the model predicts:

- 4 numbers for the box offset (using a clever *distribution-focal-loss* encoding)
- 1 *objectness* score (is there anything here?)
- $C$ class probabilities (one per damage type)

So total output channels = `4 + 1 + C`. After NMS, the surviving cells become the final detections.
"""),
    code('''\
# Use the project's own helper to peek at what the model outputs.
from ccdp.data.schema import DAMAGE_TYPES
C = len(DAMAGE_TYPES)
print(f"Per grid cell, YOLOv8 outputs {4 + 1 + C} channels:")
print(f"  4 (box) + 1 (objectness) + {C} (one per class)")
'''),
    md("""\
## 5. mAP — mean Average Precision

The detection metric. For one class:

1. Sort all predictions by confidence descending.
2. Walk down the list. Each prediction is a **True Positive** if its IoU with an unmatched ground-truth box ≥ 0.5, else a **False Positive**.
3. After each step, compute precision $= TP / (TP + FP)$ and recall $= TP / N_\\text{gt}$.
4. Plot the precision-recall curve. The area under it is **AP** (Average Precision) for this class.
5. Average AP over all classes → **mAP**.

`mAP@0.5` means IoU threshold 0.5. `mAP@0.5:0.95` averages mAP across IoU thresholds 0.5, 0.55, …, 0.95 — much stricter.
"""),
    code('''\
# Tiny worked example: one class, 5 predictions vs 3 ground-truth boxes.
import numpy as np

preds = [
    # (conf, is_TP)
    (0.95, True),
    (0.90, True),
    (0.85, False),
    (0.60, True),
    (0.30, False),
]
N_gt = 3
tp = fp = 0
prec_list, rec_list = [], []
for conf, is_tp in preds:
    if is_tp: tp += 1
    else:     fp += 1
    prec_list.append(tp / (tp + fp))
    rec_list.append(tp / N_gt)

print("Recall :", [round(r,2) for r in rec_list])
print("Precision:", [round(p,2) for p in prec_list])

import matplotlib.pyplot as plt
plt.figure(figsize=(5,4))
plt.plot(rec_list, prec_list, marker="o")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title(f"AP (trapezoidal AUC) ≈ {np.trapz(prec_list, rec_list):.3f}")
plt.grid(True); plt.xlim(0,1); plt.ylim(0,1.05)
plt.show()
'''),
    md("""\
## 6. Demo-scale training

Ultralytics' YOLOv8 has its own trainer. Tiny epoch count below — replace `epochs=1, imgsz=320` with `epochs=50, imgsz=640` and a real dataset YAML for production. (The project's full-training entrypoint is `ccdp.train.detector`.)
"""),
    code('''\
# Tiny "does it run?" smoke test — Ultralytics has a built-in `coco8` toy
# dataset that's perfect for verifying the training loop in seconds.
from ultralytics import YOLO

model = YOLO("yolov8n.yaml")        # nano, untrained
# Comment out next line if you don't want to download the toy dataset on Colab.
# results = model.train(data="coco8.yaml", epochs=1, imgsz=320, verbose=False)

print("ultralytics imported OK. Uncomment the train line to actually train (~30s).")
'''),
    md("""\
## 7. Visualise predictions on a real image

If the production detector is available locally (after `app.py` first boot or after running the production-weights download), we can run it end-to-end and draw the boxes.
"""),
    code('''\
from pathlib import Path
from PIL import Image
from ccdp.viz import annotate_prediction

# This block only runs if production weights are present.
try:
    from ccdp.infer.variant_b import VariantBPipeline
    pipe = VariantBPipeline()
    sample = next(Path("data").rglob("*.jpg"), None) or next(Path("data").rglob("*.png"), None)
    if sample:
        img = Image.open(sample).convert("RGB")
        pred = pipe.predict(img)
        annotated = annotate_prediction(img, pred)
        annotated
    else:
        print("No sample image found under data/. Upload one to see real boxes.")
except Exception as e:
    print(f"Detector not available yet: {e}")
'''),
    md("""\
**Next:** notebook 04 covers the **XGBoost cost regressor** that ingests the classifier + detector outputs and emits dollars.
"""),
])


# ===========================================================================
# Notebook 04 — XGBoost cost regressor
# ===========================================================================

NB_04 = notebook([
    md("""\
# 04 — XGBoost cost regressor

The classifier says "dent + scratch". The detector says "two boxes covering ~6% of the image". A repair shop says "$420". The cost regressor maps the model outputs to a dollar amount.

## Roadmap

1. Why **gradient boosting** instead of another neural net
2. The math: decision tree → boosting → gradient boosting
3. Features the regressor sees
4. The **calibration** trick — how we re-price without retraining
5. Runnable training on synthetic data
"""),
    COLAB_BOOTSTRAP,
    md("""\
## 1. Why XGBoost here?

The input to the cost stage is *structured*: a 2048-d image feature vector plus categorical fields (make, body type, segment) plus bbox stats. For tabular structured data, **gradient-boosted trees** routinely beat neural networks — fewer hyperparameters, faster to train, less prone to overfit small datasets, and they give us *interpretable feature importances* for free.

XGBoost is the most battle-tested implementation. (`HistGradientBoostingRegressor` from scikit-learn is an equally good drop-in if you want a pure-sklearn variant — see the conversation context for that ablation idea.)

## 2. From decision trees → boosting → gradient boosting

### Decision tree
A flowchart: at each node, ask a yes/no question about one feature; descend until you hit a leaf with a prediction.

### Boosting
Train a sequence of **weak** trees. Each new tree focuses on the examples the previous trees got wrong. Sum all trees' predictions to get the final answer.

### Gradient boosting
The "focuses on what was wrong" is made precise: tree $t+1$ is fit to the **negative gradient** of the loss with respect to the current ensemble's predictions. For squared-error loss this is just the residual $y - \\hat y$.

Formally, at step $t$:

$$
F_{t+1}(x) = F_t(x) + \\nu \\cdot h_t(x), \\qquad h_t \\approx -\\nabla_F \\mathcal{L}(y, F_t(x))
$$

where $\\nu$ is the learning rate.
"""),
    code('''\
# Visualise boosting on a 1-D toy problem.
import numpy as np, matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

rng = np.random.default_rng(0)
x = np.linspace(0, 6, 100)
y = np.sin(x) + 0.2 * rng.standard_normal(x.shape)

X = x.reshape(-1, 1)
preds = np.zeros_like(y)
fig, axes = plt.subplots(1, 4, figsize=(15, 3.2), sharey=True)
for i, n_trees in enumerate([1, 3, 10, 50]):
    preds = np.zeros_like(y)
    for _ in range(n_trees):
        resid = y - preds
        tree = DecisionTreeRegressor(max_depth=2).fit(X, resid)
        preds += 0.3 * tree.predict(X)
    axes[i].scatter(x, y, s=8, alpha=0.4)
    axes[i].plot(x, preds, color="red")
    axes[i].set_title(f"{n_trees} trees")
plt.suptitle("Gradient boosting fits residuals tree by tree")
plt.show()
'''),
    md("""\
## 3. What features does the cost regressor see?

Variant B's feature row (training and inference) contains:

| Group | Features | Source |
|---|---|---|
| Image features | `f_0` … `f_2047` | ResNet50 backbone output |
| Car metadata | `year`, `make`, `body_type`, `segment` | User input or identifier model |
| BBox stats | counts, total area, max area, area-weighted class fractions | YOLOv8 detector |

The XGBoost model learns nonlinear interactions between *all* of these — e.g. "luxury sedan + scratch on door + low total area → $X" vs "old SUV + scratch on bumper + same area → $Y".

## 4. The calibration trick

Re-training XGBoost every time the catalog changes would be wasteful (and would lose us the production weights). Instead we add a **post-hoc calibrator**: a small affine transform applied on top of the XGBoost output to align it with a freshly-priced catalog.

```
cost_final = α · xgb_raw + β
```

α, β are fit by least squares on the calibration set. When you switch catalogs in the demo, only (α, β) get re-fit — the XGBoost model itself never moves. This is why the Gradio "Catalog manager" tab can re-price an image instantly.
"""),
    code('''\
# Tiny demo: fit (alpha, beta) by closed-form least squares.
import numpy as np

raw       = np.array([100, 200, 300, 400, 500])     # XGBoost raw predictions
priced    = np.array([150, 290, 470, 580, 720])     # what the new catalog says
X = np.vstack([raw, np.ones_like(raw)]).T
alpha, beta = np.linalg.lstsq(X, priced, rcond=None)[0]
print(f"alpha={alpha:.3f}  beta={beta:.3f}")
print("re-priced:", (alpha * raw + beta).round(1))
print("target   :", priced)
'''),
    md("""\
## 5. Runnable training on synthetic data
"""),
    code('''\
# Train a tiny XGBoost regressor end-to-end on fake structured data.
import numpy as np
import xgboost as xgb

rng = np.random.default_rng(0)
N = 400
X = rng.normal(size=(N, 10))                       # 10 features
y = (2*X[:, 0] - X[:, 1]**2 + 0.5*X[:, 2]).reshape(-1) + rng.normal(scale=0.3, size=N)

dtr = xgb.DMatrix(X[:300], label=y[:300])
dte = xgb.DMatrix(X[300:], label=y[300:])
params = {"objective": "reg:squarederror", "max_depth": 4, "eta": 0.1, "verbosity": 0}
booster = xgb.train(params, dtr, num_boost_round=80,
                    evals=[(dte, "test")], verbose_eval=20)

preds = booster.predict(dte)
print("RMSE on held-out:", np.sqrt(((preds - y[300:])**2).mean()).round(4))
'''),
    md("""\
**Next:** notebook 05 derives every evaluation metric we use — precision, recall, F1, RMSE, MAE, MAPE, R² — with worked numbers.
"""),
])


# ===========================================================================
# Notebook 05 — Metrics deep dive
# ===========================================================================

NB_05 = notebook([
    md("""\
# 05 — Metrics deep dive

Every metric the project reports, derived from scratch and verified against the project's own `ccdp.eval.metrics` implementations.

## Classification (per-class)
- **Precision** = TP / (TP + FP)
- **Recall**    = TP / (TP + FN)
- **F1**        = harmonic mean of precision and recall
- **Support**   = number of true examples of this class

## Regression (cost)
- **RMSE** — root mean squared error
- **MAE**  — mean absolute error
- **MAPE** — mean absolute percentage error
- **R²**   — coefficient of determination

## Detection
- **IoU**  (already covered in NB 03)
- **mAP** (already covered in NB 03)
"""),
    COLAB_BOOTSTRAP,
    md("""\
## Precision, Recall, F1 — derivation

For one class, build a 2×2 *confusion matrix*:

|             | predicted positive | predicted negative |
|------------:|:------------------:|:------------------:|
| true pos    | **TP**             | FN                 |
| true neg    | FP                 | TN                 |

$$
\\text{precision} = \\frac{TP}{TP + FP}, \\quad \\text{recall} = \\frac{TP}{TP + FN}
$$

Precision answers "**when the model said yes, was it right?**"
Recall answers "**of all the actual positives, how many did we catch?**"

F1 is their harmonic mean — it punishes models that get one of them very low:

$$
F_1 = 2 \\cdot \\frac{P \\cdot R}{P + R}
$$
"""),
    code('''\
import numpy as np

# 6 examples of class "dent" — model probabilities and ground truth.
# (we'll threshold at 0.5)
prob   = np.array([0.9, 0.8, 0.4, 0.2, 0.7, 0.1])
truth  = np.array([1,   1,   1,   0,   0,   0])
pred   = (prob >= 0.5).astype(int)

tp = int(((pred==1) & (truth==1)).sum())
fp = int(((pred==1) & (truth==0)).sum())
fn = int(((pred==0) & (truth==1)).sum())

precision = tp / max(tp+fp, 1)
recall    = tp / max(tp+fn, 1)
f1        = 2 * precision * recall / max(precision+recall, 1e-9)

print(f"TP={tp} FP={fp} FN={fn}")
print(f"precision={precision:.3f}  recall={recall:.3f}  F1={f1:.3f}")
'''),
    md("""\
Compare to the project's own implementation:
"""),
    code('''\
from ccdp.eval.metrics import per_class_prf
import numpy as np

# Reshape into the multi-label "[N, C]" form the function expects.
classes = ["dent"]
probs  = prob.reshape(-1, 1)
labels = truth.reshape(-1, 1).astype(float)
m = per_class_prf(probs, labels, classes)
print(m["per_class"]["dent"])
'''),
    md("""\
## Regression metrics — derivation
"""),
    code('''\
import numpy as np

y_true = np.array([100, 200, 300, 400], dtype=float)
y_pred = np.array([110, 180, 330, 360], dtype=float)
err = y_pred - y_true

rmse = np.sqrt((err**2).mean())
mae  = np.abs(err).mean()
mape = (np.abs(err) / np.abs(y_true)).mean() * 100
# R² = 1 − SS_res / SS_tot
ss_res = (err**2).sum()
ss_tot = ((y_true - y_true.mean())**2).sum()
r2 = 1 - ss_res / ss_tot

print(f"RMSE = {rmse:.2f}")
print(f"MAE  = {mae:.2f}")
print(f"MAPE = {mape:.2f} %")
print(f"R²   = {r2:.4f}")
'''),
    md("""\
**Which one should I report?** Insurance triage is most interpretable in **MAE** (dollars off, on average) and **MAPE** (% off). RMSE is more sensitive to outliers — good for catching the model when it occasionally predicts $5 instead of $500.

Verify against the project:
"""),
    code('''\
from ccdp.eval.metrics import regression_metrics
m = regression_metrics(y_true.tolist(), y_pred.tolist())
m
'''),
    md("""\
## Why F1 instead of just accuracy?

If 95% of images have NO dent, a model that always says "no dent" gets 95% accuracy but is useless. Precision/recall/F1 ignore the easy negatives and only look at how well the model handles positives. **Use F1 (or precision-recall AUC) for any class-imbalanced problem.**
"""),
])


# ===========================================================================
# Notebook 06 — End-to-end inference
# ===========================================================================

NB_06 = notebook([
    md("""\
# 06 — End-to-end inference

Putting all the pieces together: upload a real image, run the full pipeline, see boxes, see costs.

This notebook requires the **production weights** to be present locally. If you ran `app.py` once (or pointed the cells below at the GitHub Release tarball), they will be at `checkpoints/production/`. Otherwise the cell that creates the pipeline will say "checkpoint missing" and explain how to fetch them.
"""),
    COLAB_BOOTSTRAP,
    md("""\
## Fetch the production weights (once)

Same logic that runs on first boot of the HuggingFace Space.
"""),
    code('''\
# Reuse the bootstrap from app.py — safe to run inside the notebook directory.
import sys
sys.path.insert(0, ".")
import app   # this triggers the weight + catalog bootstrap as a side-effect
print("Weights at:", "checkpoints/production/")
'''),
    md("""\
## Run the full pipeline on a sample image
"""),
    code('''\
from pathlib import Path
from PIL import Image
from ccdp.preprocess import preprocess
from ccdp.viz import annotate_prediction
from ccdp.infer.variant_a import VariantAPipeline
from ccdp.infer.variant_b import VariantBPipeline

pipe_a = VariantAPipeline()
pipe_b = VariantBPipeline()

# Pick the first image you can find. Replace with your own path:
sample = next(Path("data").rglob("*.jpg"), None) or next(Path(".").rglob("*.png"), None)
print("sample:", sample)

img = Image.open(sample).convert("RGB")
img_clean, prep = preprocess(img)

pred_a = pipe_a.predict(img_clean).to_dict()
pred_b = pipe_b.predict(img_clean)

print("Variant A cost:", pred_a["cost"], pred_a["currency"])
print("Variant B cost:", pred_b.cost, pred_b.currency)
print("Detections   :", [(d.damage_type, round(d.confidence,2)) for d in pred_b.detections])

annotated = annotate_prediction(img_clean, pred_b)
annotated
'''),
    md("""\
That's everything: data in, boxes out, cost out. Modify the snippet above with your own image to verify on something not in the training set.

If you want to run the Gradio UI locally, from the repo root:

```bash
python app.py
```

Or the FastAPI server:

```bash
uvicorn ccdp.api.server:app --reload
```
"""),
])


# ---------------------------------------------------------------------------
# Write everything out
# ---------------------------------------------------------------------------

NOTEBOOKS = {
    "00_overview.ipynb":                NB_00,
    "01_data_and_preprocessing.ipynb":  NB_01,
    "02_classifier_resnet50.ipynb":     NB_02,
    "03_detector_yolov8.ipynb":         NB_03,
    "04_cost_regressor_xgboost.ipynb":  NB_04,
    "05_metrics_deep_dive.ipynb":       NB_05,
    "06_end_to_end_inference.ipynb":    NB_06,
}


def main() -> None:
    for name, nb in NOTEBOOKS.items():
        path = HERE / name
        with path.open("w") as f:
            json.dump(nb, f, indent=1)
            f.write("\n")
        print(f"wrote {path.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
