"""
InsightAI - HTML report generator.

Builds a single, self-contained "InsightAI Data Analysis Report" as an HTML
string with inline CSS (so it renders in a sandboxed preview and in any
browser). Charts are embedded as base64 PNGs when provided; otherwise the
report falls back to data tables.
"""

from __future__ import annotations

import base64
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd


def _fmt(value, default: str = "—") -> str:
    if value is None:
        return default
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _table_to_html(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or len(df) == 0:
        return "<p><em>No data.</em></p>"
    show = df.head(max_rows)
    cols = [str(c) for c in show.columns]
    head = "".join(f"<th>{c}</th>" for c in cols)
    body = ""
    for _, row in show.iterrows():
        cells = "".join(f"<td>{_fmt(v)}</td>" for v in row.values)
        body += f"<tr>{cells}</tr>"
    return f'<table class="table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def _png_src(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("utf-8")


def generate_html_report(results: Dict, charts: Optional[Dict[str, bytes]] = None,
                         title: str = "InsightAI Data Analysis Report") -> str:
    """Return a complete HTML report string."""
    charts = charts or {}
    profile = results.get("profile", {})
    quality = results.get("quality", {})
    cleaning_log = results.get("cleaning_log", [])
    insights = results.get("insights", {})

    css = """
    <style>
        body { font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
               color: #1f2937; background: #f9fafb; margin:0; padding:0; line-height:1.55; }
        .wrap { max-width: 960px; margin: 0 auto; padding: 24px; }
        h1 { color:#111827; font-size:1.8rem; margin:0 0 4px; }
        h2 { color:#111827; border-bottom:2px solid #e5e7eb; padding-bottom:6px; margin-top:32px; }
        h3 { color:#374151; margin-top:20px; }
        .sub { color:#6b7280; font-size:0.95rem; }
        .card-grid { display:flex; flex-wrap:wrap; gap:12px; margin:16px 0; }
        .card { background:#fff; border:1px solid #e5e7eb; border-radius:10px; padding:14px 16px; flex:1 1 200px; }
        .card .label { font-size:0.72rem; color:#6b7280; text-transform:uppercase; letter-spacing:.4px; }
        .card .value { font-size:1.5rem; font-weight:700; margin-top:6px; color:#111827; }
        .table { width:100%; border-collapse:collapse; margin:12px 0; font-size:0.86rem; background:#fff; }
        .table th { background:#f3f4f6; text-align:left; padding:8px 10px; border-bottom:2px solid #e5e7eb; }
        .table td { padding:7px 10px; border-bottom:1px solid #f3f4f6; }
        .score { font-size:2.2rem; font-weight:800; }
        .badge { display:inline-block; background:#eff6ff; color:#1d4ed8; border-radius:999px;
                 padding:2px 10px; font-size:0.78rem; font-weight:600; margin:2px; }
        img.chart { max-width:100%; border:1px solid #e5e7eb; border-radius:8px; margin:10px 0; }
        .kpi { color:#16a34a; }
        .section { background:#fff; border:1px solid #e5e7eb; border-radius:12px; padding:16px 20px; margin:16px 0; }
        pre { white-space:pre-wrap; background:#f9fafb; border:1px solid #e5e7eb; border-radius:8px; padding:12px; font-size:0.85rem; }
        footer { margin-top:40px; color:#9ca3af; font-size:0.8rem; text-align:center; }
    </style>
    """

    # KPI cards
    schema = profile.get("schema_summary", {})
    cards = [
        ("Records", _fmt(profile.get("rows"))),
        ("Columns", _fmt(profile.get("columns"))),
        ("Memory", profile.get("memory_human", "—")),
        ("Missing", f"{profile.get('missing_total_pct', 0)}%"),
        ("Duplicates", _fmt(profile.get("duplicates"))),
        ("Numeric", _fmt(schema.get("numeric"))),
        ("Categorical", _fmt(schema.get("categorical"))),
        ("Datetime", _fmt(schema.get("datetime"))),
    ]
    cards_html = "".join(
        f'<div class="card"><div class="label">{l}</div><div class="value">{v}</div></div>'
        for l, v in cards
    )

    sections: List[str] = []

    # Dataset overview
    schema_html = "".join(
        f'<span class="badge">{k}: {v}</span>' for k, v in schema.get("by_type", {}).items()
    )
    sections.append(
        f"<div class='section'><h2>Dataset Overview</h2>"
        f"<div class='card-grid'>{cards_html}</div>"
        f"<p><strong>Schema:</strong> {schema_html}</p>"
        f"<h3>Column schema</h3>"
        f"{_table_to_html(pd.DataFrame(profile.get('site_columns', []))) if profile.get('site_columns') else '<p>No schema.</p>'}"
        f"</div>"
    )

    # Data quality
    score = quality.get("score", 0)
    grade = quality.get("grade", "—")
    sub = quality.get("sub_scores", {})
    sub_html = "".join(
        f'<div class="card"><div class="label">{k}</div><div class="value">{_fmt(v)}</div></div>'
        for k, v in sub.items()
    )
    issues_html = "".join(
        f"<li><strong>{i.get('type')}:</strong> {i.get('detail')}</li>" for i in quality.get("issues", [])
    ) or "<li>None significant.</li>"
    sections.append(
        f"<div class='section'><h2>Data Quality</h2>"
        f"<div class='card-grid'><div class='card'><div class='label'>Quality Score</div>"
        f"<div class='score'>{_fmt(score)}</div><div class='sub'>Grade {grade}</div></div>"
        f"{sub_html}</div>"
        f"<h3>Detected issues</h3><ul>{issues_html}</ul></div>"
    )

    # Cleaning performed
    cle = pd.DataFrame(cleaning_log) if cleaning_log else pd.DataFrame()
    sections.append(
        f"<div class='section'><h2>Cleaning Performed</h2>"
        f"{_table_to_html(cle) if not cle.empty else '<p>No cleaning operations were required.</p>'}</div>"
    )

    # Charts
    if charts:
        charts_html = "".join(
            f"<div><h3>{name}</h3><img class='chart' src='{_png_src(data)}' /></div>"
            for name, data in charts.items()
        )
        sections.append(f"<div class='section'><h2>Key Charts</h2>{charts_html}</div>")

    # Univariate
    univ = results.get("univariate", {})
    if univ:
        univ_html = ""
        if isinstance(univ.get("numeric"), pd.DataFrame):
            univ_html += "<h3>Numeric descriptives</h3>" + _table_to_html(univ["numeric"])
        sections.append(f"<div class='section'><h2>Statistical Findings</h2>{univ_html}</div>")

    # AI insights
    insight_text = insights.get("text", "")
    src = insights.get("source", "")
    sections.append(
        f"<div class='section'><h2>AI Business Insights</h2>"
        f"<p class='sub'>Generated by {'Ollama' if src == 'ollama' else 'deterministic engine'}</p>"
        f"<pre>{insight_text}</pre></div>"
    )

    # Recommendations / limitations / next steps
    sections.append(
        f"<div class='section'><h2>Recommendations & Next Steps</h2>"
        f"<p>See the AI insights above. Domain validation and incremental refresh are "
        f"recommended before acting on conclusions.</p></div>"
    )

    # Appendix: analytics report JSON summary
    appendix = results.get("analytics_report", {})
    appendix_hint = ""
    try:
        import json
        appendix_hint = json.dumps(appendix, default=str, indent=2)[:3000]
    except Exception:
        pass
    sections.append(
        f"<div class='section'><h2>Appendix</h2><pre>{appendix_hint}</pre></div>"
    )

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8" />
<title>{title}</title>{css}</head>
<body><div class="wrap">
<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;">
  <div><h1>{title}</h1><div class="sub">Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div></div>
  <div class="sub">InsightAI — Autonomous AI Data Analyst</div>
</div>
{''.join(sections)}
<footer>InsightAI · Generated locally · Data never leaves your machine</footer>
</div></body></html>"""
    return html
