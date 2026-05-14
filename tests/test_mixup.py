"""Tests for the MixUp / CutMix batch augmenters."""

from __future__ import annotations

import torch

from ccdp.train.mixup import (
    apply_mixup_or_cutmix,
    cutmix,
    mixup,
    soft_cross_entropy,
)


def _mk_batch(n: int = 8, c: int = 3, h: int = 16, w: int = 16, num_classes: int = 5):
    torch.manual_seed(0)
    x = torch.rand(n, c, h, w)
    y = torch.randint(0, num_classes, (n,))
    return x, y


def test_mixup_shape_and_soft_labels():
    x, y = _mk_batch()
    x_mix, y_soft = mixup(x, y, num_classes=5, alpha=0.4)
    assert x_mix.shape == x.shape
    assert y_soft.shape == (x.size(0), 5)
    # rows sum to ~1 (probability distribution)
    assert torch.allclose(y_soft.sum(dim=-1), torch.ones(x.size(0)), atol=1e-5)


def test_cutmix_shape_and_soft_labels():
    x, y = _mk_batch(h=32, w=32)
    x_mix, y_soft = cutmix(x, y, num_classes=5, alpha=1.0)
    assert x_mix.shape == x.shape
    assert y_soft.shape == (x.size(0), 5)
    assert torch.allclose(y_soft.sum(dim=-1), torch.ones(x.size(0)), atol=1e-5)
    # at least *some* pixels should differ from original (the pasted patch)
    assert (x_mix != x).any()


def test_mixup_disabled_returns_one_hot():
    x, y = _mk_batch()
    x_mix, y_soft = mixup(x, y, num_classes=5, alpha=0.0)
    # disabled: input unchanged, target is exact one-hot
    assert torch.equal(x_mix, x)
    assert y_soft.shape == (x.size(0), 5)
    assert ((y_soft == 0) | (y_soft == 1)).all()
    assert y_soft.argmax(dim=-1).equal(y)


def test_apply_choice_branches():
    x, y = _mk_batch()
    # prob=0 -> identity
    x_out, y_out = apply_mixup_or_cutmix(x, y, num_classes=5, prob=0.0)
    assert torch.equal(x_out, x)
    # prob=1, share=1 -> CutMix always
    x_out, y_out = apply_mixup_or_cutmix(x, y, num_classes=5,
                                          prob=1.0, cutmix_share=1.0,
                                          cutmix_alpha=1.0)
    assert y_out.shape == (x.size(0), 5)
    # prob=1, share=0 -> MixUp always
    x_out, y_out = apply_mixup_or_cutmix(x, y, num_classes=5,
                                          prob=1.0, cutmix_share=0.0,
                                          mixup_alpha=0.4)
    assert y_out.shape == (x.size(0), 5)


def test_soft_cross_entropy_matches_standard_on_one_hot():
    import torch.nn.functional as F
    torch.manual_seed(0)
    logits = torch.randn(8, 5)
    y = torch.randint(0, 5, (8,))
    y_oh = F.one_hot(y, num_classes=5).float()
    standard = F.cross_entropy(logits, y)
    soft = soft_cross_entropy(logits, y_oh)
    assert torch.allclose(standard, soft, atol=1e-6)
