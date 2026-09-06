"""
InsightAI - Dashboard building blocks.

Pure HTML/CSS helpers used by the Streamlit app to render KPI cards, section
headings and badges. Inline styling is used so the cards render correctly inside
Streamlit (including the sandboxed preview).
"""

from __future__ import annotations

from typing import Dict, List


def kpi_card(label: str, value: str, delta: str = "", tone: str = "blue") -> str:
    """Return an HTML string for a single KPI card."""
    tones = {
        "blue": "#2563eb", "green": "#16a34a", "red": "#dc2626",
        "amber": "#d97706", "purple": "#7c3aed", "gray": "#6b7280",
    }
    color = tones.get(tone, tones["blue"])
    bg = {"blue": "#eff6ff", "green": "#f0fdf4", "red": "#fef2f2",
          "amber": "#fffbeb", "purple": "#f5f3ff", "gray": "#f9fafb"}.get(tone, "#eff6ff")
    delta_html = f'<div style="font-size:0.8rem;color:{color};font-weight:600;margin-top:4px;">{delta}</div>' if delta else ""
    return f"""
    <div style="background:{bg};border:1px solid {color}33;border-left:4px solid {color};
                border-radius:10px;padding:14px 16px;min-height:96px;box-shadow:0 1px 2px rgba(0,0,0,0.04);">
        <div style="font-size:0.75rem;color:#6b7280;text-transform:uppercase;letter-spacing:0.5px;font-weight:600;">{label}</div>
        <div style="font-size:1.6rem;font-weight:700;color:#111827;margin-top:6px;">{value}</div>
        {delta_html}
    </div>
    """


def kpi_row(cards: List[Dict]) -> str:
    """Render a responsive row of KPI cards."""
    cols = len(cards)
    base = 100 / max(cols, 1) - 1.2
    inner = "".join(
        f'<div style="flex:1 1 {base}%;min-width:150px;margin:6px;">{kpi_card(c["label"], c["value"], c.get("delta", ""), c.get("tone", "blue"))}</div>'
        for c in cards
    )
    return f'<div style="display:flex;flex-wrap:wrap;gap:6px;">{inner}</div>'


def section_header(title: str, subtitle: str = "") -> str:
    sub = f'<div style="color:#6b7280;font-size:0.9rem;margin-top:2px;">{subtitle}</div>' if subtitle else ""
    return f'<div style="margin:12px 0 8px;"><h3 style="margin:0;color:#111827;">{title}</h3>{sub}</div>'


def badge(text: str, tone: str = "blue") -> str:
    tones = {
        "blue": ("#eff6ff", "#1d4ed8"), "green": ("#f0fdf4", "#15803d"),
        "red": ("#fef2f2", "#b91c1c"), "amber": ("#fffbeb", "#b45309"),
        "gray": ("#f9fafb", "#374151"),
    }
    bg, fg = tones.get(tone, tones["blue"])
    return f'<span style="background:{bg};color:{fg};border:1px solid {fg}33;border-radius:999px;padding:2px 10px;font-size:0.78rem;font-weight:600;">{text}</span>'


def data_quality_gauge(score: float) -> str:
    """Render a data-quality score as a coloured number + bar."""
    color = "#16a34a" if score >= 80 else ("#d97706" if score >= 60 else "#dc2626")
    return f"""
    <div>
        <div style="font-size:2.4rem;font-weight:800;color:{color};">{score:.1f}<span style="font-size:1rem;color:#9ca3af;">/100</span></div>
        <div style="background:#e5e7eb;border-radius:999px;height:10px;margin-top:6px;max-width:260px;">
            <div style="height:10px;border-radius:999px;width:{max(0, min(100, score))}%;background:{color};"></div>
        </div>
    </div>
    """
