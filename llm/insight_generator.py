"""
InsightAI - AI insight generator.

Assembles the structured analytics (produced by Python) and generates business
insights via Ollama, augmented by RAG context. When Ollama is unavailable it
returns a deterministic, rule-based insight report so the application never
breaks. The LLM *only explains* numbers; Python does all calculation.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from llm.ollama_client import OllamaClient
from llm import prompts
from rag.retriever import Retriever


def build_analytics_report(
    profile: Dict,
    quality: Dict,
    cleaning_log: List[Dict],
    univariate: Dict,
    composition: List[Dict],
    distribution: List[Dict],
    comparison: List[Dict],
    relationships: Dict,
    sentiment: Optional[Dict],
    timeseries: Optional[Dict],
    anomaly: Optional[Dict],
) -> Dict:
    """Assemble a compact, JSON-safe analytics report for the LLM."""
    report: Dict = {
        "dataset": {
            "rows": profile.get("rows"),
            "columns": profile.get("columns"),
            "memory_human": profile.get("memory_human"),
            "missing_total": profile.get("missing_total"),
            "missing_total_pct": profile.get("missing_total_pct"),
            "duplicates": profile.get("duplicates"),
            "schema_summary": profile.get("schema_summary"),
        },
        "data_quality": quality,
        "cleaning_performed": cleaning_log,
        "univariate": {
            "numeric": univariate.get("numeric", []),
            "categorical_top": univariate.get("categorical", []),
        },
        "composition": composition,
        "distribution": distribution,
        "comparison": comparison,
        "relationships": relationships,
        "sentiment": sentiment,
        "time_series": timeseries,
        "anomaly": anomaly,
    }
    return report


def _safe_summary(value) -> str:
    """Truncate a value to a concise string for prompt context."""
    if isinstance(value, dict):
        return str(value)[:1500]
    return str(value)


def generate_executive_insights(
    analytics_report: Dict,
    retriever: Optional[Retriever] = None,
    ollama: Optional[OllamaClient] = None,
    rag_top_k: int = 5,
) -> Dict:
    """
    Generate structured executive insights.

    Returns a dict with keys:
      ``text``       – the raw model output (or deterministic fallback text),
      ``source``     – "ollama" or "deterministic",
      ``structured`` – parsed markdown-like sections if available.
    """
    ollama = ollama or OllamaClient()
    context = ""
    if retriever is not None:
        query = (
            "data quality, EDA, composition, distribution, comparison, correlation, "
            "time series, sentiment, anomaly detection and business insight interpretation"
        )
        context = retriever.build_prompt_context(query, top_k=rag_top_k)

    prompt = prompts.executive_insight_prompt(analytics_report, context)

    if ollama.is_available():
        try:
            text = ollama.generate(prompt, system=prompts.SYSTEM_ROLE)
            return {"text": text, "source": "ollama", "structured": _parse_sections(text)}
        except Exception as exc:
            fallback = deterministic_insights(analytics_report)
            fallback["note"] = f"Ollama generation failed: {exc}"
            return fallback
    else:
        fallback = deterministic_insights(analytics_report)
        fallback["note"] = "Ollama not available; generated deterministic insights."
        return fallback


def deterministic_insights(analytics_report: Dict) -> Dict:
    """Rule-based fallback that produces a structured report from the numbers."""
    dataset = analytics_report.get("dataset", {})
    quality = analytics_report.get("data_quality", {})
    univariate = analytics_report.get("univariate", {})
    distribution = analytics_report.get("distribution", [])
    comparison = analytics_report.get("comparison", [])
    composition = analytics_report.get("composition", [])
    relationships = analytics_report.get("relationships", {})
    sentiment = analytics_report.get("sentiment")
    timeseries = analytics_report.get("time_series")
    anomaly = analytics_report.get("anomaly")

    lines: List[str] = []
    lines.append("## Executive Summary")
    lines.append(
        f"This dataset contains {dataset.get('rows')} records and "
        f"{dataset.get('columns')} columns "
        f"(schema: {dataset.get('schema_summary', {}).get('by_type', {})}). "
        f"Data quality score is {quality.get('score', 'n/a')}/100 "
        f"(grade {quality.get('grade', 'n/a')})."
    )

    lines.append("\n## Top 5 Findings")
    findings = []

    # Composition finding.
    if composition:
        comp = composition[0]
        top = comp.get("top_categories", {})
        if top:
            top_name, top_pct = list(top.items())[0]
            findings.append(f"The largest group in '{comp.get('column')}' is "
                            f"'{top_name}' at {top_pct}% of records.")
    # Distribution (skew) finding.
    if distribution:
        d = distribution[0]
        if d.get("shape") and d.get("shape") != "symmetric":
            findings.append(f"'{d.get('column')}' is {d.get('shape')} with "
                            f"skewness {d.get('skewness')} and "
                            f"{d.get('outlier_pct')}% outliers.")
        else:
            findings.append(f"'{d.get('column')}' is roughly symmetric "
                            f"(skew {d.get('skewness')}) with std {d.get('std')}.")
    # Comparison finding.
    if comparison:
        comp = comparison[0]
        top = comp.get("top_groups", {})
        if top:
            g_name, g_val = list(top.items())[0]
            findings.append(f"'{comp.get('group')}' group '{g_name}' has the highest "
                            f"{comp.get('metric')} at {g_val}.")
    # Time-series finding.
    if timeseries and timeseries.get("available"):
        growth = timeseries.get("growth", {})
        trend = timeseries.get("trend", {})
        if growth.get("overall_growth_pct") is not None:
            findings.append(f"'{timeseries.get('value_col')}' grew by "
                            f"{growth['overall_growth_pct']}% over the period "
                            f"({trend.get('direction', 'n/a')} trend).")
    # Relationship finding.
    if relationships:
        pearson = relationships.get("pearson", {})
        strongest = pearson.get("strongest", [])
        if strongest:
            s = strongest[0]
            findings.append(f"Strongest relationship: '{s['a']}' and '{s['b']}' "
                            f"(r = {s['corr']}).")
    # Sentiment finding.
    if sentiment and sentiment.get("available"):
        summary = next(iter(sentiment["summary"].values()), {})
        if summary:
            findings.append(f"Sentiment is {summary.get('positive_pct')}% positive, "
                            f"{summary.get('negative_pct')}% negative "
                            f"(avg compound {summary.get('avg_compound')}).")

    for i, f in enumerate(findings[:5], 1):
        lines.append(f"{i}. {f}.")
    while len(findings) < 5:
        findings.append("Additional structure present; see detailed sections.")
        lines.append(f"{len(findings)}. {findings[-1]}")
        if len(findings) == 5:
            break

    lines.append("\n## Data Quality")
    for issue in quality.get("issues", []):
        lines.append(f"- {issue.get('type')}: {issue.get('detail')}")
    if not quality.get("issues"):
        lines.append("- No major data quality issues detected.")

    lines.append("\n## Trends")
    if timeseries and timeseries.get("available"):
        t = timeseries.get("trend", {})
        lines.append(f"- The {timeseries.get('value_col')} trend is "
                     f"'{t.get('direction')}' (linear R² ≈ {t.get('r2')}).")
    else:
        lines.append("- No clear time-based trend; 'no datetime column detected'.")

    lines.append("\n## Risks")
    for issue in quality.get("issues", []):
        if issue.get("type") == "missing":
            lines.append(f"- Missing values ({issue.get('detail')}) may bias downstream analysis.")
    if anomaly and anomaly.get("total_anomalies"):
        lines.append(f"- {anomaly.get('total_anomalies')} anomalous record(s) "
                     f"({anomaly.get('anomaly_pct')}%) may indicate outliers or errors.")
    if not quality.get("issues") and not (anomaly and anomaly.get("total_anomalies")):
        lines.append("- No significant risk indicators detected based on automated checks.")

    lines.append("\n## Opportunities")
    if composition:
        comp = composition[0]
        top = comp.get("top_categories", {})
        if top:
            top_name, top_pct = list(top.items())[0]
            lines.append(f"- Concentration in '{comp.get('column')}'='{top_name}' "
                         f"({top_pct}%) may offer a focused growth or risk-reduction lever.")
    lines.append("- Further segment-level analysis can reveal well-performing niches.")

    lines.append("\n## Recommendations")
    lines.append("- Prioritise the highest-impact segment identified in the comparison analysis.")
    lines.append("- Address any data quality issues highlighted above before deeper modelling.")
    lines.append("- Validate trends and relationships with domain experts before acting on them.")

    lines.append("\n## Further Questions")
    lines.append("- What drives the top-performing group in the comparison analysis?")
    lines.append("- Are the detected anomalies genuine events or data errors?")
    lines.append("- How do these findings change when the dataset is refreshed over time?")

    text = "\n".join(lines)
    return {"text": text, "source": "deterministic", "structured": _parse_sections(text)}


def _parse_sections(md: str) -> Dict[str, str]:
    """Split a markdown string into {section_title: body}."""
    sections: Dict[str, str] = {}
    current = None
    for line in md.splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            sections[current] = ""
        elif current:
            sections[current] += line + "\n"
    return sections
