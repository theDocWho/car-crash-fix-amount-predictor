"""Image preprocessing transforms shared by trainers and inference.

Three callables are exported:

* :data:`IMAGENET_MEAN` / :data:`IMAGENET_STD` — normalisation statistics that
  match the ImageNet-pretrained ResNet50 weights we fine-tune from.
* :func:`eval_transform` — deterministic resize+centre-crop+normalise. Used
  for validation, test, feature extraction, and inference.
* :func:`train_transform` — augmentation pipeline with optional RandAugment.

Both transforms accept ``image_size`` (typically 224 for ResNet50). The train
transform optionally takes RandAugment parameters; setting ``num_ops=0`` falls
back to a simpler ColorJitter for backwards compatibility.
"""

from __future__ import annotations

from typing import Callable

from torchvision import transforms

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


def eval_transform(image_size: int = 224) -> Callable:
    """Return the deterministic preprocessing pipeline.

    Resize to 1.15× the target size, centre-crop down to ``image_size``, then
    convert to a normalised float tensor. The 1.15× margin matches what we use
    during training so train/eval distributions agree.
    """
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def train_transform(
    image_size: int = 224,
    randaug_num_ops: int = 0,
    randaug_magnitude: int = 9,
) -> Callable:
    """Return a training-time augmentation pipeline.

    With ``randaug_num_ops > 0`` the pipeline uses :class:`torchvision.transforms.RandAugment`
    with the given operation count and magnitude (typical values: ``num_ops=2,
    magnitude=9``). With ``num_ops=0`` it falls back to a hand-tuned ColorJitter
    that matches our pre-RandAugment baseline so old behaviour is recoverable.
    """
    pipeline = [
        transforms.Resize(int(image_size * 1.15)),
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
    ]
    if randaug_num_ops > 0:
        pipeline.append(transforms.RandAugment(
            num_ops=randaug_num_ops, magnitude=randaug_magnitude,
        ))
    else:
        pipeline.append(transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.15,
        ))
    pipeline += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(pipeline)
