"""ccdp.eval — Variant A vs Variant B comparison + report rendering.

Public API:
    build_comparison(variant_a_pipeline, variant_b_pipeline=None, ...)
        -> evaluates both variants on the deterministic seed=42 test split.
    Comparison
        -> the dataclass that holds everything the report needs.
    report.render_html / report.render_pdf / report.generate
        -> render to HTML and optionally PDF.
"""

from .comparison import Comparison, VariantReport, build_comparison, evaluate_variant
from . import report

__all__ = [
    "Comparison",
    "VariantReport",
    "build_comparison",
    "evaluate_variant",
    "report",
]
