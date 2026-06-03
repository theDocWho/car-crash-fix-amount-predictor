"""Guard the Gradio ``_estimate`` return arity.

The handler is wired to 6 outputs (annotated, id, variant_a, variant_b,
variant_d, json). Every return path — including the early "no image" and "no
car" exits — must produce exactly 6 values, or Gradio errors in the UI.
"""

from __future__ import annotations

from PIL import Image

import ccdp.api.demo as demo

_N_OUTPUTS = 6
_ARGS = ("Both", "USD", 0.6, 0.2, 0.3, True, "", "", None, "unknown")


def test_no_image_returns_six():
    out = demo._estimate(None, *_ARGS)
    assert len(out) == _N_OUTPUTS


def test_no_car_returns_six(monkeypatch):
    # don't load real models
    monkeypatch.setattr(demo, "_get_pipelines", lambda: {"a": None, "b": None, "d": None})

    from ccdp.identification.auto_identify import AutoIdentifyResult
    from ccdp.identification.car_gate import GateResult

    def fake_auto(image, **kwargs):
        return AutoIdentifyResult(
            has_car=False, gate=GateResult(has_car=False),
            identification=None, ml=None, note="no car",
        )

    monkeypatch.setattr("ccdp.identification.auto_identify.auto_identify", fake_auto)
    out = demo._estimate(Image.new("RGB", (64, 64)), *_ARGS)
    assert len(out) == _N_OUTPUTS
    # the annotated banner image must still be returned (not None) for the no-car case
    assert out[0] is not None
