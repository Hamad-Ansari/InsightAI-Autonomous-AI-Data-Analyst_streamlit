"""
InsightAI — Autonomous AI Data Analyst.

Streamlit entry point. Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Ensure the project root is importable regardless of CWD.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

# Local modules (project root is on sys.path).
from config.settings import get_settings
from utils.logger import get_logger
from utils.validators import InsightAIError
from ingestion.loader import load_dataset, sample_dataframe
from ingestion.schema_detector import detect_schema, columns_by_type
from agent.analyst_agent import AnalystAgent
from llm.ollama_client import OllamaClient
from rag.ingest import build_index
from rag.retriever import Retriever
import visualization.charts as charts
import visualization.dashboard as dash
from agent.planner import build_plan
from analysis import comparison as cmp_analysis
from reporting.report_generator import generate_all, export_cleaned_csv

logger = get_logger()

st.set_page_config(page_title="InsightAI", layout="wide",
                   page_icon="📊", initial_sidebar_state="expanded")

# --------------------------------------------------------------------------- Cached resources


@st.cache_resource(show_spinner=False)
def get_ollama() -> OllamaClient:
    return OllamaClient()


@st.cache_resource(show_spinner=False)
def get_retriever():
    s = get_settings()
    try:
        index = build_index(s.knowledge_base_dir, s.embedding_model, s.vector_db_path)
        return Retriever(index)
    except Exception as exc:
        logger.warning("RAG index build failed: %s", exc)
        return None


@st.cache_data(show_spinner=False)
def _parse_upload(file_bytes: bytes, filename: str, max_mb: int) -> pd.DataFrame:
    """Parse an uploaded file into a DataFrame (cached)."""
    safe = filename
    tmp = Path(tempfile.gettempdir()) / safe
    tmp.write_bytes(file_bytes)
    try:
        return load_dataset(tmp, max_mb=max_mb)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


@st.cache_data(show_spinner=False)
def _run_agent(df_key: str, df: pd.DataFrame, options: dict) -> dict:
    """Run the full autonomous pipeline, cached by dataframe hash + options."""
    import hashlib
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        agent = AnalystAgent(ollama=get_ollama(), retriever=get_retriever())
        results = agent.run(
            df.copy(),
            numeric_strategy=options.get("numeric_strategy", "median"),
            categorical_strategy=options.get("categorical_strategy", "mode"),
            drop_duplicates=options.get("drop_duplicates", True),
            normalize_case=options.get("normalize_case", True),
        )
        return results


def _df_key(df: pd.DataFrame) -> str:
    try:
        import hashlib
        return hashlib.md5(
            pd.util.hash_pandas_object(df, index=True).values.tobytes()
        ).hexdigest()
    except Exception:
        return str(len(df))


# --------------------------------------------------------------------------- Small helpers


def _safe_chart(fig, name: str = "chart"):
    """Render a plotly figure, tolerating failures."""
    try:
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.warning(f"Could not render {name}: {exc}")


def _show_note(text: str):
    st.info(text)


def _section(title: str, subtitle: str = ""):
    st.markdown(dash.section_header(title, subtitle), unsafe_allow_html=True)


def _fmt_table(df: pd.DataFrame, limit: int = 30):
    if df is None or getattr(df, "empty", True):
        st.caption("No data to display.")
        return
    with st.expander("View data", expanded=False):
        st.dataframe(df.head(limit), use_container_width=True)


# --------------------------------------------------------------------------- Sidebar

SIDEBAR_PAGES = [
    "🏠 Home", "📄 Dataset Info", "🧹 Data Cleaning", "🔎 EDA",
    "🧩 Composition", "📊 Distribution", "⚖️ Comparison", "🔗 Relationship",
    "💬 Sentiment", "📈 Time Series", "🤖 AI Insights", "📚 RAG Knowledge",
    "📊 Dashboard", "📑 Final Report", "⬇️ Export", "⚙️ Settings",
]


def sidebar_settings() -> dict:
    s = get_settings()
    with st.sidebar.expander("⚙️ Analysis Settings", expanded=False):
        numeric_strategy = st.selectbox("Numeric imputation", ["median", "mean"])
        categorical_strategy = st.selectbox("Categorical imputation", ["mode", "empty"])
        drop_duplicates = st.checkbox("Remove exact duplicates", True)
        normalize_case = st.checkbox("Normalise category case", True)
        sample_frac = st.selectbox("Sampling", [1.0, 0.5, 0.25, 0.10],
                                   format_func=lambda f: f"Full" if f == 1.0 else f"{int(f*100)}%")
    with st.sidebar.expander("🔑 Ollama Settings", expanded=False):
        st.text_input("Ollama URL", value=s.ollama_base_url, key="ollama_url")
        st.text_input("Model", value=s.ollama_model, key="ollama_model")
        st.slider("Temperature", 0.0, 1.0, s.ollama_temperature, 0.05, key="temp")
        st.number_input("Max tokens", 200, 4000, s.ollama_max_tokens, 50, key="max_tokens")
    return {
        "numeric_strategy": numeric_strategy,
        "categorical_strategy": categorical_strategy,
        "drop_duplicates": drop_duplicates,
        "normalize_case": normalize_case,
        "sample_frac": sample_frac,
        "ollama_url": st.session_state.get("ollama_url", s.ollama_base_url),
        "ollama_model": st.session_state.get("ollama_model", s.ollama_model),
        "temperature": st.session_state.get("temp", s.ollama_temperature),
        "max_tokens": st.session_state.get("max_tokens", s.ollama_max_tokens),
    }


# --------------------------------------------------------------------------- Pages


def page_home(results=None):
    _section("InsightAI", "Autonomous AI Data Analyst")
    st.markdown(
        "Upload a dataset (CSV, XLSX, XLS, TSV, Parquet) and InsightAI will "
        "automatically understand, clean, explore and explain it. Everything "
        "runs **locally** — with **Ollama** for reasoning and a **RAG** knowledge "
        "base for analytical context."
    )
    results = st.session_state.get("results")
    if results:
        _render_kpi_cards(results)
        _section("Auto Analysis Summary")
        if st.button("🤖 Generate AI Insights now", key="btn_insights"):
            insights = st.session_state.setdefault("insights_shown", True)
        st.success("Dataset is loaded and analysed. Explore the sidebar to view each analysis dimension.")
        st.markdown("---")
        # Plan
        plan = results.get("plan", {})
        st.caption("Agent execution plan:")
        cols = st.columns(2)
        cols[0].markdown("**Executed**")
        cols[0].write(", ".join(plan.get("executed", [])))
        cols[1].markdown("**Skipped (not applicable)**")
        cols[1].write(", ".join(plan.get("skipped", [])) or "None")
        # Fast intro to charts
        _render_dashboard_preview(results)
    else:
        _render_welcome()


def _render_welcome():
    st.markdown("---")
    st.header("Get started")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Use your own data** — use the file uploader in the sidebar.")
    with c2:
        if st.button("✨ Load a sample dataset (sales)", key="load_sample_sales"):
            st.session_state["df"] = load_dataset(
                str(PROJECT_ROOT / "sample_data" / "sales.csv"))
            st.session_state["dataset_name"] = "sales.csv"
            st.session_state["results"] = None
            st.rerun()
    # Show expected capabilities
    st.info("Capabilities: schema detection · cleaning · EDA (composition, distribution, "
            "comparison, relationship) · sentiment · time-series · anomaly detection · "
            "RAG-augmented AI insights · dashboard · HTML/PDF/JSON/CSV reports.")


def _render_kpi_cards(results):
    profile = results.get("profile", {})
    schema = profile.get("schema_summary", {})
    quality = results.get("quality", {})
    cards = [
        {"label": "Records", "value": f"{profile.get('rows', 0):,}", "tone": "blue"},
        {"label": "Columns", "value": f"{profile.get('columns', 0)}", "tone": "blue"},
        {"label": "Data Quality", "value": f"{quality.get('score', 0)}", "delta": f"Grade {quality.get('grade', '—')}", "tone": "green"},
        {"label": "Missing", "value": f"{profile.get('missing_total_pct', 0)}%", "tone": "amber"},
        {"label": "Duplicates", "value": f"{profile.get('duplicates', 0):,}", "tone": "red"},
        {"label": "Numeric", "value": f"{schema.get('numeric', 0)}", "tone": "purple"},
        {"label": "Categorical", "value": f"{schema.get('categorical', 0)}", "tone": "gray"},
        {"label": "Date cols", "value": f"{schema.get('datetime', 0)}", "tone": "blue"},
    ]
    st.markdown(dash.kpi_row(cards), unsafe_allow_html=True)


def page_dataset_info(results):
    _section("Dataset Information")
    profile = results.get("profile", {})
    _render_kpi_cards(results)
    st.markdown("---")
    st.subheader("Dataset profile")
    meta = {
        "Rows": profile.get("rows"),
        "Columns": profile.get("columns"),
        "Memory": profile.get("memory_human"),
        "Missing cells": profile.get("missing_total"),
        "Duplicate rows": profile.get("duplicates"),
        "Schema": profile.get("schema_summary", {}).get("by_type"),
    }
    st.table(pd.DataFrame(list(meta.items()), columns=["Attribute", "Value"]))
    _section("Column Schema")
    site = profile.get("site_columns", [])
    if site:
        _fmt_table(pd.DataFrame(site))
    # Preview cleaned data
    st.subheader("Cleaned data preview")
    cd = results.get("cleaned_df")
    if cd is not None:
        st.dataframe(cd.head(20), use_container_width=True)


def page_cleaning(results):
    _section("Data Cleaning")
    log = results.get("cleaning_log", [])
    if log:
        st.success(f"{len(log)} cleaning operation(s) performed. The original data was preserved.")
        _fmt_table(pd.DataFrame(log))
    else:
        st.info("No cleaning operations were required (data was already clean).")
    # Before/after
    options = st.session_state.get("analysis_options", {})
    st.subheader("Outlier handling")
    numeric = [c for c, t in results.get("schema_types", {}).items()
               if t in {"NUMERIC", "CURRENCY", "PERCENTAGE"}]
    outliers = results.get("outlier_frame")
    if outliers is not None and not outliers.empty:
        _fmt_table(outliers, limit=40)
    else:
        st.caption("No numeric columns; outlier analysis skipped.")
    st.caption("Outliers are reported but never deleted by default. Use Settings re-run to change strategy.")


def page_eda(results):
    _section("Exploratory Data Analysis (EDA)")
    tabs = st.tabs(["Univariate", "Interactive chart builder"])
    with tabs[0]:
        univ = results.get("univariate", {})
        st.subheader("Numeric descriptives")
        _fmt_table(univ.get("numeric"), limit=40)
        st.subheader("Categorical frequencies")
        _fmt_table(univ.get("categorical"), limit=50)
        st.subheader("Text column summaries")
        _fmt_table(univ.get("text"), limit=40)
    with tabs[1]:
        _interactive_chart_builder(results)


def _interactive_chart_builder(results):
    st.caption("Pick columns and InsightAI recommends the best chart for the data types.")
    schema_types = results.get("schema_types", {})
    df = results.get("cleaned_df")
    if df is None:
        st.warning("No cleaned dataset available.")
        return
    cols = list(df.columns)
    if len(cols) < 1:
        st.warning("No columns available for charting.")
        return
    x = st.selectbox("X axis", cols, index=0, key="ib_x")
    opts_y = [c for c in cols if c != x]
    if not opts_y:
        st.warning("Add at least two columns to build an interactive chart.")
        return
    y = st.selectbox("Y axis", opts_y, index=0, key="ib_y")
    rec = charts_rec(schema_types.get(x), schema_types.get(y))
    st.caption(f"Recommended: **{rec['chart']}** — {rec['reason']}")
    chart_type = st.selectbox("Chart type", ["auto", "scatter", "bar", "line", "histogram",
                                             "box", "violin", "heatmap", "donut", "treemap", "table"],
                              key="ib_type")
    if chart_type == "table":
        st.dataframe(df[[x, y]].head(100), use_container_width=True)
        return
    agg = st.selectbox("Aggregation", ["mean", "sum", "median", "count", "min", "max"], key="ib_agg")
    color = st.selectbox("Color by (optional)", ["None"] + cols, key="ib_color")
    color_ser = df[color] if color != "None" else None
    try:
        if chart_type in {"auto", "scatter"}:
            fig = charts.scatter_chart(df, x, y, color=color_ser)
        elif chart_type in {"bar", "line"}:
            if schema_types.get(y) in {"NUMERIC", "CURRENCY", "PERCENTAGE"} and schema_types.get(x) in {"CATEGORICAL", "BOOLEAN"}:
                col = cmp_analysis.compare_groups(df, x, y, agg=agg, top_n=15)
                tmp = pd.Series(col["value"].values, index=col["group"].values)
                fig = charts.bar_chart(tmp, f"{y} by {x} ({agg})")
            elif schema_types.get(x) == "DATETIME":
                s = pd.to_numeric(df[y], errors="coerce"); ind = pd.to_datetime(df[x], errors="coerce")
                fdf = pd.DataFrame({"t": ind, "v": s}).dropna().groupby(pd.Grouper(key="t", freq="D"))["v"].mean()
                fig = charts.time_series_chart(fdf.index, fdf.values)
            else:
                fig = charts.bar_chart(df[y].astype(str).value_counts(), x)
        elif chart_type in {"histogram", "box", "violin"}:
            fig = charts.distribution_chart(df[y], kind=({"histogram": "histogram", "box": "box", "violin": "violin"}[chart_type]))
        elif chart_type == "heatmap":
            num = [c for c, t in schema_types.items() if t in {"NUMERIC", "CURRENCY", "PERCENTAGE"}]
            if len(num) >= 2:
                corr = df[num].apply(pd.to_numeric, errors="coerce").corr()
                fig = charts.correlation_heatmap(corr)
            else:
                st.warning("Need at least 2 numeric columns for a heatmap.")
                return
        elif chart_type in {"donut", "treemap"}:
            from analysis.composition import composition_by_category
            table = composition_by_category(df, x, top_n=10)
            fig = charts.composition_chart(table, kind=chart_type)
        else:
            fig = charts.bar_chart(df[y].astype(str).value_counts(), y)
        _safe_chart(fig, x)
    except Exception as exc:
        st.warning(f"Chart build failed: {exc}")


def charts_rec(xt, yt):
    from visualization.chart_selector import recommend_for_pair
    return recommend_for_pair(xt, yt)


def page_composition(results):
    _section("Composition Analysis", "'What is this data made of?'")
    comp = results.get("composition", {})
    categories = comp.get("categories", [])
    df = results.get("cleaned_df")
    if not categories or df is None:
        st.info("No categorical columns detected; composition analysis was skipped.")
        return
    cat = st.selectbox("Categorical column", categories, key="comp_cat")
    numeric = [c for c, t in results.get("schema_types", {}).items()
               if t in {"NUMERIC", "CURRENCY", "PERCENTAGE"}]
    val = st.selectbox("Weighted by metric (optional)", ["None"] + numeric, key="comp_val")
    from analysis.composition import composition_by_category
    kind = st.selectbox("Chart type", ["donut", "treemap", "stacked_bar"], key="comp_kind")
    table = composition_by_category(df, cat, value_col=None if val == "None" else val, top_n=10)
    fig = charts.composition_chart(table, kind=kind, value_col="share_pct")
    _safe_chart(fig, "Composition")
    _fmt_table(table, limit=20)


def page_distribution(results):
    _section("Distribution Analysis", "'How are values spread?'")
    dist = results.get("distribution")
    schema_types = results.get("schema_types", {})
    numeric = [c for c, t in schema_types.items() if t in {"NUMERIC", "CURRENCY", "PERCENTAGE"}]
    df = results.get("cleaned_df")
    if not numeric or df is None:
        st.info("No numeric columns detected; distribution analysis was skipped.")
        return
    col = st.selectbox("Numeric column", numeric, key="dist_col")
    kind = st.selectbox("Plot", ["histogram", "kde", "box", "violin", "ecdf"], key="dist_kind")
    fig = charts.distribution_chart(df[col], kind="histogram" if kind == "histogram" else kind)
    _safe_chart(fig, "Distribution")
    st.subheader("Distribution statistics")
    _fmt_table(dist, limit=40)


def page_comparison(results):
    _section("Comparison Analysis", "'How do groups differ?'")
    schema_types = results.get("schema_types", {})
    cat = [c for c, t in schema_types.items() if t in {"CATEGORICAL", "BOOLEAN"}]
    num = [c for c, t in schema_types.items() if t in {"NUMERIC", "CURRENCY", "PERCENTAGE"}]
    df = results.get("cleaned_df")
    if not cat or not num or df is None:
        st.info("Need at least one categorical and one numeric column for comparison.")
        return
    gc = st.selectbox("Group by (categorical)", cat, key="cmp_grp")
    vc = st.selectbox("Metric (numeric)", num, key="cmp_val")
    agg = st.selectbox("Aggregation", ["mean", "sum", "median", "count", "min", "max"], key="cmp_agg")
    grouped = cmp_analysis.compare_groups(df, gc, vc, agg=agg, top_n=15)
    fig = charts.comparison_bar(df, gc, vc, agg=agg)
    _safe_chart(fig, "Comparison")
    _fmt_table(grouped, limit=20)
    st.subheader("Top groups")
    st.dataframe(grouped.head(10), use_container_width=True)


def page_relationship(results):
    _section("Relationship Analysis", "'How are variables connected?'")
    rel = results.get("relationships", {})
    df = results.get("cleaned_df")
    schema_types = results.get("schema_types", {})
    tab1, tab2, tab3 = st.tabs(["Correlation", "Strongest pairs", "Categorical association"])
    with tab1:
        pearson = rel.get("pearson", {})
        if pearson.get("matrix"):
            _safe_chart(charts.correlation_heatmap(pd.DataFrame(pearson["matrix"])), "Heatmap")
        else:
            st.info("Need ≥2 numeric columns.")
    with tab2:
        strongest = pearson.get("strongest", [])
        if strongest:
            st.dataframe(pd.DataFrame(strongest), use_container_width=True)
        else:
            st.caption("No correlations available.")
        # scatter selector
        num = [c for c, t in schema_types.items() if t in {"NUMERIC", "CURRENCY", "PERCENTAGE"}]
        if df is not None and len(num) >= 2:
            x = st.selectbox("X", num, key="rel_x"); y = st.selectbox("Y", [c for c in num if c != x], key="rel_y")
            _safe_chart(charts.scatter_chart(df, x, y), "Scatter")
    with tab3:
        cat_pairs = rel.get("categorical", [])
        if cat_pairs:
            st.dataframe(pd.DataFrame(cat_pairs), use_container_width=True)
        else:
            st.caption("No categorical association available.")


def page_sentiment(results):
    _section("Sentiment Analysis")
    senti = results.get("sentiment")
    if not senti or not senti.get("available"):
        st.info("No text/review/comment columns detected — sentiment analysis was skipped.")
        return
    st.caption("Using local lexicon-based scoring (VADER-compatible).")
    summary = senti.get("summary", {})
    for col, s in summary.items():
        counts = s.get("counts", {})
        st.plotly_chart(charts.sentiment_donut(counts), use_container_width=True)
        st.markdown(f"**{col}** — {s['n']} texts: ")
        st.write(
            f"Positive {s['positive_pct']}% · Neutral {s['neutral_pct']}% · "
            f"Negative {s['negative_pct']}% · avg compound {s['avg_compound']}"
        )
        c1, c2 = st.columns(2)
        c1.markdown("**Top positive terms**")
        c1.write(", ".join(s.get("positive_words", [])))
        c2.markdown("**Top negative terms**")
        c2.write(", ".join(s.get("negative_words", [])))
    st.subheader("Common themes")
    st.write(", ".join(senti.get("themes", [])) or "—")


def page_timeseries(results):
    _section("Time-Series Analysis")
    ts = results.get("timeseries")
    if not ts or not ts.get("available"):
        st.info("No datetime column detected — time-series analysis was skipped.")
        return
    df = results.get("cleaned_df")
    date_col = ts.get("date_col")
    value_col = ts.get("value_col")
    st.caption(f"Index: **{date_col}**, target: **{value_col}** (frequency: {ts.get('freq')})")
    idx = ts.get("index", [])
    series = ts.get("series", [])
    ma = ts.get("rolling_data")
    if idx and series:
        fig = charts.time_series_chart(idx, series, title="Time series", ma_series=ma)
        _safe_chart(fig, "Time series")
    st.subheader("Trend & growth")
    g = ts.get("growth", {}); tr = ts.get("trend", {})
    st.metric("Overall growth", f"{g['overall_growth_pct']}%" if g.get('overall_growth_pct') is not None else "—")
    st.metric("Last period growth", f"{g['last_period_growth_pct']}%" if g.get('last_period_growth_pct') is not None else "—")
    st.write(f"Trend direction: **{tr.get('direction')}** (R² ≈ {tr.get('r2')})")
    st.subheader("Seasonality")
    seasons = ts.get("seasonality", {})
    if seasons:
        st.write(seasons)
    st.subheader("Decomposition")
    dec = ts.get("decomposition", {})
    st.caption(dec.get("reason", "Additive decomposition (trend + seasonal + residual)."))
    # Forecasting
    st.subheader("Forecast (experimental)")
    horizon = st.number_input("Horizon", 1, 30, 5, key="ts_h")
    methods = st.multiselect("Methods", ["naive", "moving_average", "exponential_smoothing", "arima"],
                             default=["naive", "moving_average"], key="ts_m")
    if methods:
        try:
            from analysis.timeseries import forecast
            ser = pd.Series(list(series), index=pd.to_datetime(idx))
            fx = forecast(ser, methods, horizon=int(horizon))
            st.caption(fx.get("note", "Forecasts are estimates, not certainties."))
            for name, f in fx.get("forecasts", {}).items():
                if f.get("values"):
                    st.write(f"**{name}** ({f.get('window')}): " +
                             ", ".join(f"{v:.2f}" for v in f["values"]))
                    st.plotly_chart(charts.forecast_chart(ser.index, ser.values,
                                                          f["values"], title=f"Forecast ({name})"),
                                    use_container_width=True)
        except Exception as exc:
            st.warning(f"Forecasting not available: {exc}")
    else:
        st.info("Select at least one forecast method.")


def page_ai_insights(results):
    _section("AI Business Insights")
    insights = results.get("insights", {})
    source = insights.get("source", "")
    if source == "ollama":
        st.success("Insights generated by **Ollama** (local LLM).")
    else:
        st.warning("Insights generated by the **deterministic engine** (Ollama is not running or unset).")
    note = insights.get("note")
    if note:
        st.caption(note)
    st.markdown("---")
    text = insights.get("text", "")
    st.markdown(text)
    # Structured sections
    structured = insights.get("structured", {})
    if structured:
        with st.expander("View structured sections"):
            for k, v in structured.items():
                st.markdown(f"**{k}**:")
                st.write(v)


def page_rag(results):
    _section("RAG Knowledge Base", "Methodology context that grounds the AI explanations.")
    ret = get_retriever()
    if ret is None:
        st.warning("RAG knowledge base unavailable (embeddings/vector store not usable).")
        return
    kb = ret.chunks
    st.write(f"Loaded **{len(kb)}** knowledge chunks from **{ret.model_name}** (backend: {ret.index.get('backend')}).")
    with st.expander("Browse knowledge base documents"):
        sources = sorted({c["source"] for c in kb})
        for s in sources:
            st.markdown(f"- `{s}`")
    st.subheader("Query the knowledge base")
    q = st.text_input("Ask about methodology (e.g. 'how should missing values be handled?')",
                      value="how to interpret correlation and skewness")
    topk = st.slider("Top-k", 1, 8, 4, key="rag_topk")
    if q:
        res = ret.search(q, top_k=topk)
        for r in res:
            st.markdown(f"**[{r['source']}]**")
            st.write(r["text"][:600])
            st.caption(f"score: {r['score']}" if r["score"] is not None else "")
            st.divider()


def page_dashboard(results):
    _section("Executive Dashboard")
    _render_kpi_cards(results)
    st.markdown("---")
    _render_dashboard_sections(results)


def _render_dashboard_preview(results):
    _section("Fast preview", "")
    _render_dashboard_sections(results, preview=True)


def _render_dashboard_sections(results, preview=False):
    df = results.get("cleaned_df")
    schema_types = results.get("schema_types", {})
    comp = results.get("composition", {})
    categories = comp.get("categories", [])
    numeric = [c for c, t in schema_types.items() if t in {"NUMERIC", "CURRENCY", "PERCENTAGE"}]
    # Composition
    if categories and df is not None:
        from analysis.composition import composition_by_category
        table = composition_by_category(df, categories[0], top_n=8)
        st.plotly_chart(charts.composition_chart(table, kind="donut"), use_container_width=True)
    # Distribution
    if numeric and df is not None:
        st.plotly_chart(charts.distribution_chart(df[numeric[0]], kind="histogram"), use_container_width=True)
    # Relationship
    rel = results.get("relationships", {}).get("pearson", {}).get("matrix")
    if rel:
        st.plotly_chart(charts.correlation_heatmap(pd.DataFrame(rel)), use_container_width=True)
    # Insights excerpt
    ins = results.get("insights", {}).get("text", "")
    if ins:
        st.markdown("---")
        st.markdown("#### AI Summary")
        st.caption(ins[:900] + ("…" if len(ins) > 900 else ""))


def page_report(results):
    _section("Final Report")
    st.caption("Generate the complete InsightAI Data Analysis Report (HTML / PDF / JSON / CSV).")
    cl = st.columns([1, 1, 1])
    if cl[0].button("Generate HTML report", key="gen_html"):
        _generate_and_store(results, formats=["html", "json", "summary_csv"])
    if st.session_state.get("report_ready"):
        st.success("Report generated. Download from the Export page or below.")
        st.markdown("---")
    # Live preview
    st.subheader("Report preview")
    if st.session_state.get("report_html"):
        st.components.v1.html(st.session_state["report_html"], height=700, scrolling=True)
    else:
        st.info("Click 'Generate HTML report' to preview.")


def _generate_and_store(results, formats=None):
    from reporting.report_generator import generate_all
    charts_map = _collect_charts(results)
    dataset_name = st.session_state.get("dataset_name", "dataset")
    try:
        exports = generate_all(results, dataset_name, charts=charts_map)
    except Exception as exc:
        exports = generate_all(results, dataset_name, charts={})
        st.warning(f"Charts omitted from report: {exc}")
    st.session_state["report_ready"] = True
    st.session_state["report_html"] = exports.get("html", b"").decode("utf-8")
    st.session_state["exports"] = exports
    st.session_state["pdf"] = exports.get("pdf")
    st.session_state["json"] = exports.get("json", b"")
    st.session_state["summary_csv"] = exports.get("summary_csv", b"")
    if "pdf" in st.session_state and st.session_state["pdf"]:
        st.success("Report generated (HTML + PDF + JSON + CSV).")


def _collect_charts(results):
    """Attempt to render key figures to PNG for the report; skip on failure."""
    out = {}
    df = results.get("cleaned_df")
    schema_types = results.get("schema_types", {})
    if df is None:
        return out
    try:
        numeric = [c for c, t in schema_types.items() if t in {"NUMERIC", "CURRENCY", "PERCENTAGE"}]
        comp = results.get("composition", {})
        categories = comp.get("categories", [])
        if categories:
            from analysis.composition import composition_by_category
            table = composition_by_category(df, categories[0], top_n=8)
            out["Composition"] = charts.figure_to_png(charts.composition_chart(table, kind="donut"))
        if numeric:
            out["Distribution"] = charts.figure_to_png(charts.distribution_chart(df[numeric[0]], kind="histogram"))
        rel = results.get("relationships", {}).get("pearson", {}).get("matrix")
        if rel:
            out["Correlation"] = charts.figure_to_png(charts.correlation_heatmap(pd.DataFrame(rel)))
    except Exception as exc:
        logger.warning("Could not export chart PNGs: %s", exc)
    return out


def page_export(results):
    _section("Export")
    df = results.get("cleaned_df")
    dataset_name = st.session_state.get("dataset_name", "dataset")
    if df is not None:
        st.download_button("⬇️ Download cleaned CSV", df.to_csv(index=False).encode("utf-8"),
                           file_name=f"cleaned_{dataset_name}.csv", mime="text/csv", key="dl_clean")
    exports = st.session_state.get("exports", {})
    if exports:
        c = st.columns(4)
        if exports.get("html"):
            c[0].download_button("⬇️ HTML report", exports["html"], file_name="insightai_report.html", mime="text/html")
        if exports.get("pdf"):
            c[1].download_button("⬇️ PDF report", exports["pdf"], file_name="insightai_report.pdf", mime="application/pdf")
        if exports.get("json"):
            c[2].download_button("⬇️ JSON results", exports["json"], file_name="insightai_results.json", mime="application/json")
        if exports.get("summary_csv"):
            c[3].download_button("⬇️ Summary CSV", exports["summary_csv"], file_name="insightai_summary.csv", mime="text/csv")
    else:
        st.info("Generate the report in the 'Final Report' page to enable downloads.")
        # Always allow downloading the machine-readable JSON of current results.
        from reporting.report_generator import export_json_results
        j = export_json_results(results, dataset_name)
        st.download_button("⬇️ Download analysis results (JSON)", j, file_name="insightai_results.json", mime="application/json")


def page_settings(results):
    _section("Settings & Environment")
    s = get_settings()
    ollama = get_ollama()
    ok = ollama.is_available()
    if ok:
        st.success(f"Ollama is running at `{ollama.base_url}`. Models: {ollama.list_models()[:6]}")
    else:
        st.warning("Ollama is not running. Start Ollama and try again. InsightAI still performs "
                   "full deterministic EDA without the LLM.")
    st.subheader("Configuration")
    cfg = {
        "OLLAMA_BASE_URL": ollama.base_url,
        "OLLAMA_MODEL": ollama.model,
        "EMBEDDING_MODEL": s.embedding_model,
        "VECTOR_DB_PATH": s.vector_db_path,
        "MAX_UPLOAD_SIZE_MB": s.max_upload_size_mb,
        "LOG_LEVEL": s.log_level,
    }
    st.json(cfg)
    st.subheader("Ollama quick commands")
    st.code("ollama pull llama3.2:3b  # install a model\nollama serve             # start the server", language="bash")
    # Rerun controls
    st.subheader("Re-run analysis")
    if st.button("🔄 Re-run analysis with current settings", key="rerun"):
        options = st.session_state.get("analysis_options", {})
        df = st.session_state.get("df")
        if df is not None:
            with st.spinner("Re-running full analysis…"):
                key = _df_key(sample_dataframe(df, options.get("sample_frac", 1.0)))
                results = _run_agent(key, sample_dataframe(df, options.get("sample_frac", 1.0)), options)
                st.session_state["results"] = results
                st.session_state["report_ready"] = False
                st.session_state["exports"] = {}
                st.rerun()


# --------------------------------------------------------------------------- Main


def main():
    s = get_settings()
    loader = st.sidebar.file_uploader(
        "Upload dataset", type=["csv", "xlsx", "xls", "tsv", "parquet"],
        help="CSV, XLSX, XLS, TSV, Parquet. Max size configurable in .env.",
    )
    # Sidebar page selector
    page = st.sidebar.radio("Navigate", SIDEBAR_PAGES, index=0)
    st.sidebar.markdown("---")
    options = sidebar_settings()
    st.session_state["analysis_options"] = options
    if not loader and "df" not in st.session_state:
        # No dataset yet
        st.session_state.setdefault("results", None)
        page_home()
        st.sidebar.caption("InsightAI — local & free. Data never leaves your device.")
        return

    # Load / refresh dataset on new upload.
    if loader is not None:
        uploaded_key = (loader.name, loader.size)
        if st.session_state.get("upload_key") != uploaded_key:
            st.session_state["upload_key"] = uploaded_key
            with st.spinner("Loading and analysing dataset…"):
                try:
                    df = _parse_upload(loader.getvalue(), loader.name, s.max_upload_size_mb)
                    st.session_state["df"] = df
                    st.session_state["dataset_name"] = loader.name
                    st.session_state["results"] = None
                    st.session_state["report_ready"] = False
                    st.session_state["exports"] = {}
                except InsightAIError as exc:
                    st.error(f"Could not load the dataset: {exc}")
                    st.stop()
                except Exception as exc:
                    st.error(f"Unexpected error loading dataset: {exc}")
                    st.stop()

    df = st.session_state.get("df")
    if df is None:
        page_home()
        return

    # Run the autonomous agent on first load / when results absent.
    if st.session_state.get("results") is None:
        with st.spinner("Running the autonomous analysis pipeline…"):
            sampled = sample_dataframe(df, options.get("sample_frac", 1.0))
            key = _df_key(sampled)
            try:
                results = _run_agent(key, sampled, options)
                st.session_state["results"] = results
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
                st.stop()

    results = st.session_state["results"]

    # Dispatch pages.
    dispatcher = {
        "🏠 Home": page_home,
        "📄 Dataset Info": page_dataset_info,
        "🧹 Data Cleaning": page_cleaning,
        "🔎 EDA": page_eda,
        "🧩 Composition": page_composition,
        "📊 Distribution": page_distribution,
        "⚖️ Comparison": page_comparison,
        "🔗 Relationship": page_relationship,
        "💬 Sentiment": page_sentiment,
        "📈 Time Series": page_timeseries,
        "🤖 AI Insights": page_ai_insights,
        "📚 RAG Knowledge": page_rag,
        "📊 Dashboard": page_dashboard,
        "📑 Final Report": page_report,
        "⬇️ Export": page_export,
        "⚙️ Settings": page_settings,
    }
    dispatcher[page](results)
    st.sidebar.caption(f"Dataset: {st.session_state.get('dataset_name', '—')}")


if __name__ == "__main__":
    main()
