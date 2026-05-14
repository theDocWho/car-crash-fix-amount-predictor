"""Render a :class:`Comparison` to HTML and (when WeasyPrint is available) PDF.

The strategy is deliberately layered:

1. We always produce an HTML report — no system dependencies, just Jinja2.
2. We additionally produce a PDF when ``weasyprint`` is importable; if it's
   not, we log a friendly note and skip the PDF. WeasyPrint needs ``pango``
   and ``cairo`` system libraries that aren't trivially available on every
   environment (e.g. plain macOS without Homebrew), so making PDF optional
   keeps the HTML path usable everywhere.

The template lives at ``reports/templates/report.html.j2`` so designers can
tweak the report without touching Python.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ccdp.eval.comparison import Comparison

REPORTS_DIR = Path("reports")
TEMPLATES_DIR = REPORTS_DIR / "templates"


def _load_template():
    """Lazy-import Jinja2 so module import stays cheap."""
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    return env.get_template("report.html.j2")


def render_html(comparison: Comparison, out_path: Optional[Path] = None) -> Path:
    """Write the HTML version of the report; always works."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    tmpl = _load_template()
    html = tmpl.render(comparison=comparison, generated_at=datetime.now(timezone.utc))
    out_path = out_path or REPORTS_DIR / f"report_{_timestamp()}.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path


def render_pdf(html_path: Path, pdf_path: Optional[Path] = None) -> Optional[Path]:
    """Convert an existing HTML report to PDF via WeasyPrint, if available.

    Returns ``None`` (and prints a hint) when WeasyPrint is not installed.
    """
    try:
        from weasyprint import HTML  # type: ignore
    except ImportError:
        print("[report] WeasyPrint not installed; skipping PDF. "
              "Install with `pip install weasyprint` and the system deps "
              "(`pango`, `cairo`, `gdk-pixbuf`) for PDF output.")
        return None
    pdf_path = pdf_path or html_path.with_suffix(".pdf")
    HTML(filename=str(html_path)).write_pdf(str(pdf_path))
    return pdf_path


def generate(comparison: Comparison, also_pdf: bool = True) -> dict:
    """Single-call convenience: render HTML, then PDF if possible. Returns paths."""
    html_path = render_html(comparison)
    pdf_path = render_pdf(html_path) if also_pdf else None
    return {"html": html_path, "pdf": pdf_path}


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
