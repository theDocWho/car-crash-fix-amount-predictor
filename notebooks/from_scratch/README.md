# Neural Networks from Scratch

A 10-notebook progression that **builds** every piece of a deep-learning system in NumPy before showing how PyTorch wraps it. For complete beginners — assumes basic algebra and high-school calculus. By the end you'll be able to read the project's production training code and understand every line.

## What's different from the main `notebooks/` series

The main `notebooks/` series teaches you how *this project* works. This series teaches you **how neural networks work** — same audience, different scope.

| | Main series | From-scratch (this) |
|---|---|---|
| Goal | Use this project's code | Build the math from scratch |
| Framework | PyTorch + project code | NumPy → PyTorch only in NB 09 |
| Dataset | CarDD / Stanford Cars | CIFAR-10 (small + classic) |
| Time | ~15 min/notebook | ~30 min/notebook |

## Run order

| # | Notebook | What you build | Colab |
|---|---|---|---|
| 00 | [math_foundations](00_math_foundations.ipynb) | Vectors, dot product, derivatives, chain rule — the only prereqs | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/theDocWho/car-crash-fix-amount-predictor/blob/main/notebooks/from_scratch/00_math_foundations.ipynb) |
| 01 | [data_preprocessing](01_data_preprocessing.ipynb) | Image as array, dtype gotchas, **image-size decision rules**, normalization, augmentation, target shapes | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/theDocWho/car-crash-fix-amount-predictor/blob/main/notebooks/from_scratch/01_data_preprocessing.ipynb) |
| 02 | [perceptron_to_mlp](02_perceptron_to_mlp.ipynb) | One neuron → activations → 2-layer MLP → **manual backprop** → trains on CIFAR | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/theDocWho/car-crash-fix-amount-predictor/blob/main/notebooks/from_scratch/02_perceptron_to_mlp.ipynb) |
| 03 | [convolution_from_scratch](03_convolution_from_scratch.ipynb) | `conv2d` in 30 lines, hand-crafted kernels, **output-size formula**, pooling, receptive field | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/theDocWho/car-crash-fix-amount-predictor/blob/main/notebooks/from_scratch/03_convolution_from_scratch.ipynb) |
| 04 | [tiny_cnn_to_resnet](04_tiny_cnn_to_resnet.ipynb) | Multi-channel conv, tiny CNN, vanishing-gradient demo, **residual block** that fixes it | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/theDocWho/car-crash-fix-amount-predictor/blob/main/notebooks/from_scratch/04_tiny_cnn_to_resnet.ipynb) |
| 05 | [yolo_from_scratch](05_yolo_from_scratch.ipynb) | IoU, NMS, the grid-cell trick, mini-YOLO on synthetic data | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/theDocWho/car-crash-fix-amount-predictor/blob/main/notebooks/from_scratch/05_yolo_from_scratch.ipynb) |
| 06 | [optimisation_and_training](06_optimisation_and_training.ipynb) | SGD → momentum → Adam (all in NumPy), LR schedules, batch norm, dropout, L2 | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/theDocWho/car-crash-fix-amount-predictor/blob/main/notebooks/from_scratch/06_optimisation_and_training.ipynb) |
| 07 | [metrics_and_diagnostics](07_metrics_and_diagnostics.ipynb) | Confusion matrix, P/R/F1, AP, mAP, RMSE/MAE/R² + **which-metric decision table** | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/theDocWho/car-crash-fix-amount-predictor/blob/main/notebooks/from_scratch/07_metrics_and_diagnostics.ipynb) |
| 08 | [epoch_visualization](08_epoch_visualization.ipynb) | Loss curves, LR-finder, gradient flow, confusion heatmap, activation maps | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/theDocWho/car-crash-fix-amount-predictor/blob/main/notebooks/from_scratch/08_epoch_visualization.ipynb) |
| 09 | [pytorch_contrast](09_pytorch_contrast.ipynb) | Full mini-pipeline in PyTorch, side-by-side with NumPy code | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/theDocWho/car-crash-fix-amount-predictor/blob/main/notebooks/from_scratch/09_pytorch_contrast.ipynb) |

## Tips

- Run them **in order**. Each notebook builds on code/intuition from earlier ones.
- **Read** the markdown first, **then** run the code cell. The math motivates the implementation.
- Tweak the code. Change the learning rate, change the kernel, change the dataset size — the notebooks are meant to be poked at, not just read.

## Re-generating the notebooks

Source-of-truth is `_build.py` + `_build_part2.py`, not the ipynb JSON. To tweak content:

```bash
# edit notebooks/from_scratch/_build.py or _build_part2.py
python notebooks/from_scratch/_build.py
git add notebooks/from_scratch/*.ipynb
```

## Why NumPy and not pure Python?

NumPy *is* Python — it's the standard numerical library. Pure-pure Python with lists-of-lists would make a single CNN forward pass take minutes and bury the algorithm under index arithmetic. NumPy lets us write the math the way the textbook does. The notebooks deliberately avoid `scipy.ndimage.convolve`, `sklearn.fit`, etc. — anything that would *hide* the algorithm. NumPy is the floor.
