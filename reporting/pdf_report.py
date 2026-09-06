"""
InsightAI - PDF report generator.

Produces a clean PDF via ReportLab (pure text + tables, no external renderer
required). Images may be embedded as PNG bytes. If ReportLab is unavailable the
generator returns None and the caller falls back to the HTML/JSON exports.
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Dict, Optional

import pandas as pd


def generate_pdf_report(results: Dict, charts: Optional[Dict[str, bytes]] = None,
                        title: str = "InsightAI Data Analysis Report") -> Optional[bytes]:
    """Return PDF bytes or None if ReportLab can't be used."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                        TableStyle, Image, PageBreak, KeepTogether)
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    except Exception:
        return None

    charts = charts or {}
    profile = results.get("profile", {})
    quality = results.get("quality", {})
    cleaning_log = results.get("cleaning_log", [])
    insights = results.get("insights", {})

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Title"], fontSize=20, spaceAfter=8)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], spaceBefore=14, spaceAfter=6,
                        textColor=colors.HexColor("#111827"))
    normal = ParagraphStyle("normal", parent=styles["BodyText"], fontSize=9.5, leading=13)
    small = ParagraphStyle("small", parent=normal, fontSize=8, textColor=colors.grey)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, title=title, author="InsightAI",
                            leftMargin=0.8 * inch, rightMargin=0.8 * inch,
                            topMargin=0.8 * inch, bottomMargin=0.8 * inch)
    flow: list = []

    flow.append(Paragraph(title, h1))
    flow.append(Paragraph(f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", small))
    flow.append(Spacer(1, 10))

    # Overview table
    schema = profile.get("schema_summary", {})
    overview_rows = [
        ["Records", str(profile.get("rows"))],
        ["Columns", str(profile.get("columns"))],
        ["Memory", str(profile.get("memory_human"))],
        ["Missing", f"{profile.get('missing_total_pct', 0)}%"],
        ["Duplicates", str(profile.get("duplicates"))],
        ["Numeric cols", str(schema.get("numeric"))],
        ["Categorical cols", str(schema.get("categorical"))],
        ["Datetime cols", str(schema.get("datetime"))],
    ]
    t = Table(overview_rows, colWidths=[2.2 * inch, 2.2 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f9fafb")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e5e7eb")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    flow.append(Paragraph("Dataset Overview", h2))
    flow.append(t)

    flow.append(Paragraph("Data Quality", h2))
    flow.append(Paragraph(f"Score: {quality.get('score', 0)}/100 (grade {quality.get('grade', '—')})", normal))
    for i in quality.get("issues", []):
        flow.append(Paragraph(f"• {i.get('type')}: {i.get('detail')}", normal))

    if cleaning_log:
        flow.append(Paragraph("Cleaning Performed", h2))
        cle = pd.DataFrame(cleaning_log)
        cle_tbl = Table([cle.columns.tolist()] + cle.head(15).values.tolist(),
                        colWidths=[1.5 * inch, 1.6 * inch, 1.0 * inch, 2.8 * inch])
        cle_tbl.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2ff")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e5e7eb")),
        ]))
        flow.append(cle_tbl)

    flow.append(Paragraph("AI Business Insights", h2))
    flow.append(Paragraph(f"Source: {'Ollama' if insights.get('source') == 'ollama' else 'deterministic engine'}", small))
    for line in insights.get("text", "").splitlines():
        flow.append(Paragraph(line.replace("&", "&amp;").replace("<", "&lt;"), normal))

    # Embed charts.
    for name, png in list(charts.items())[:3]:
        try:
            img = Image(io.BytesIO(png))
            img.drawWidth = 6.2 * inch
            img.drawHeight = 3.2 * inch
            flow.append(KeepTogether([Paragraph(name, h2), img, PageBreak()]))
        except Exception:
            continue

    doc.build(flow)
    return buffer.getvalue()
