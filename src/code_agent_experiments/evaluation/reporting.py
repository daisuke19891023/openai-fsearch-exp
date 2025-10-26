"""Report generation utilities for retrieval metrics."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

from jinja2 import Environment, PackageLoader, Template, select_autoescape

if TYPE_CHECKING:
    from code_agent_experiments.domain.models import ReportSummary


def _environment() -> Environment:
    return Environment(
        loader=PackageLoader("code_agent_experiments.evaluation", "templates"),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _render(template_name: str, summary: ReportSummary) -> str:
    env = _environment()
    template: Template = env.get_template(template_name)
    return template.render(summary=summary)


def render_markdown_report(
    summary: ReportSummary, output_path: pathlib.Path | str,
) -> pathlib.Path:
    """Render a Markdown report for the provided summary."""
    path = pathlib.Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = _render("report.md.j2", summary)
    path.write_text(content, encoding="utf-8")
    return path


def render_html_report(
    summary: ReportSummary, output_path: pathlib.Path | str,
) -> pathlib.Path:
    """Render an HTML report for the provided summary."""
    path = pathlib.Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = _render("report.html.j2", summary)
    path.write_text(content, encoding="utf-8")
    return path
