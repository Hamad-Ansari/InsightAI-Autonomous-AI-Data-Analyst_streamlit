"""
InsightAI - Prompt templates.

Builds prompts for the analysis pipeline, the RAG-augmented interpretation step,
and the structured executive insight generation. All prompts are designed so the
LLM *explains* Python-computed numbers rather than recomputing them.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

SYSTEM_ROLE = (
    "You are InsightAI, an expert senior data analyst and business consultant. "
    "You translate data analysis results into clear, accurate, useful business "
    "insights for a non-technical audience. You NEVER calculate statistics "
    "yourself - you rely strictly on the numbers provided. You never invent "
    "figures. If a value is absent, you say so. You are precise, honest about "
    "uncertainty, and you separate facts from interpretation."
)


def build_eda_context(profile: Dict) -> str:
    """Compact dataset metadata text used across prompts."""
    return (
        f"Dataset profile: {profile.get('rows')} rows x {profile.get('columns')} columns. "
        f"Memory ~{profile.get('memory_human')}. "
        f"Missing cells: {profile.get('missing_total')} "
        f"({profile.get('missing_total_pct', 0)}%). "
        f"Duplicate rows: {profile.get('duplicates')}. "
        f"Schema summary: {json.dumps(profile.get('schema_summary', {}))}."
    )


def business_context_prompt(profile: Dict, schema: Dict) -> str:
    """STEP 1 prompt - hypothesis/business understanding."""
    return f"""
You are helping to understand a dataset before analysis. Given the profile below,
infer the likely business domain, the key entities, the metrics and dimensions
present, and propose 3-5 plausible business hypotheses to test.

{json.dumps({
    "profile": {k: profile.get(k) for k in ("rows", "columns", "schema_summary")},
    "columns": [{c["name"]: c["semantic_type"]} for c in schema],
}, default=str)}

Respond with concise sections:
- Likely domain
- Key entities
- Metrics
- Dimensions
- Candidate target variable(s)
- Hypotheses (numbered)
"""


def rag_interpretation_prompt(question: str, context: str, analytics_summary: str) -> str:
    """RAG-augmented: explain analytics using retrieved methodology context."""
    return f"""
QUESTION: {question}

ANALYTICS (already computed by Python, do not recompute):
{analytics_summary}

METHODOLOGY CONTEXT (retrieved from the knowledge base):
{context}

Explain the analytical result above in simple business-friendly language,
grounding your explanation in the methodology context. Do not invent numbers.
"""


def executive_insight_prompt(analytics_report: Dict, context: str = "") -> str:
    """STEP 14 prompt - structured business insight generation."""
    data = json.dumps(analytics_report, default=str, indent=2)
    ctx = ("\nBelow is relevant analytical methodology context:\n" + context) if context else ""
    return f"""
You are InsightAI, an autonomous AI data analyst. Below is a structured analytics
report computed entirely by Python. Your job is to interpret it and produce a
structured, honest business analysis.

{ctx}

STRUCTURED ANALYTICS REPORT:
{data}

Do NOT compute or invent statistics. Only reference numbers present in the report.
Use this exact markdown structure:

## Executive Summary
## Top 5 Findings
1.
2.
3.
4.
5.
## Data Quality
## Trends
## Risks
## Opportunities
## Recommendations
## Further Questions
"""


def build_user_turn(analytics_report: Dict, context: str = "") -> str:
    return executive_insight_prompt(analytics_report, context)
