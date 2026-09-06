"""
InsightAI - Report & export orchestrator.

Generates the HTML report, PDF report, machine-readable JSON results, and a CSV
analytics summary. Also writes the cleaned dataset to CSV. All outputs are saved
under ``data/exports`` and returned as bytes for the Streamlit download buttons.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from config.settings import PROJECT_ROOT
from reporting.html_report import generate_html_report
from reporting.pdf_report import generate_pdf_report


def _export_dir() -> Path:
    d = PROJECT_ROOT / "data" / "exports"
    d.mkdir(parents=True, exist_ok=True)
    return d


def export_cleaned_csv(cleaned_df: pd.DataFrame, dataset_name: str) -> Path:
    path = _export_dir() / f"cleaned_{_safe(dataset_name)}.csv"
    cleaned_df.to_csv(path, index=False)
    return path


def export_json_results(results: Dict, dataset_name: str) -> bytes:
    """Return machine-readable JSON results bytes."""
    payload = _json_safe(results)
    return json.dumps(payload, indent=2, default=str).encode("utf-8")


def export_csv_summary(results: Dict, dataset_name: str) -> bytes:
    """Return a flat CSV of key metrics/insights."""
    rows: List[Dict] = []
    profile = results.get("profile", {})
    quality = results.get("quality", {})
    schema = profile.get("schema_summary", {})
    rows.append({"section": "dataset", "metric": "records", "value": profile.get("rows")})
    rows.append({"section": "dataset", "metric": "columns", "value": profile.get("columns")})
    rows.append({"section": "dataset", "metric": "missing_pct", "value": profile.get("missing_total_pct")})
    rows.append({"section": "dataset", "metric": "duplicate_rows", "value": profile.get("duplicates")})
    rows.append({"section": "quality", "metric": "score", "value": quality.get("score")})
    rows.append({"section": "quality", "metric": "grade", "value": quality.get("grade")})
    for k, v in schema.get("by_type", {}).items():
        rows.append({"section": "schema", "metric": k, "value": v})
    for log in results.get("cleaning_log", []):
        rows.append({"section": "cleaning", "metric": log.get("operation"),
                     "value": f"{log.get('rows_affected')} rows - {log.get('reason')}"})
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def generate_all(results: Dict, dataset_name: str,
                 charts: Optional[Dict[str, bytes]] = None) -> Dict[str, bytes]:
    """Generate HTML, PDF, JSON and CSV exports; return as bytes."""
    out: Dict[str, bytes] = {}
    out["html"] = generate_html_report(results, charts=charts).encode("utf-8")
    pdf = generate_pdf_report(results, charts=charts)
    if pdf:
        out["pdf"] = pdf
    out["json"] = export_json_results(results, dataset_name)
    out["summary_csv"] = export_csv_summary(results, dataset_name)
    return out


def _safe(name: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_") or "dataset"


def _json_safe(obj):
    """Convert dataframes/numpy types to JSON-safe structures."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            return str(obj)
    if isinstance(obj, (bytes,)):
        return obj.decode("utf-8", errors="ignore")
    return str(obj) if not isinstance(obj, (str, int, float, bool)) and obj is not None else obj
