"""
InsightAI - Plotly chart builders.

Every chart is a pure function returning a ``plotly.graph_objects.Figure`` so
the Streamlit UI can render it directly. Figures are interactive (hover, zoom,
legend, and downloadable via the Plotly toolbar).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

_PALETTE = px.colors.qualitative.Plotly


def figure_to_png(fig: go.Figure, scale: int = 2) -> bytes:
    """Render a plotly figure to PNG bytes. Requires kaleido; raises on failure."""
    import plotly.io as pio

    return pio.to_image(fig, format="png", scale=scale)


def _base_layout(fig: go.Figure, title: str, x_label: str = "", y_label: str = "") -> go.Figure:
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e6e6e6")
    fig.update_yaxes(showgrid=True, gridcolor="#e6e6e6")
    return fig


# ---------------------------------------------------------------- Histogram / KDE / Box / Violin
def distribution_chart(series: pd.Series, kind: str = "histogram", bins: int = 40) -> go.Figure:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return go.Figure()
    if kind == "kde":
        x = np.linspace(series.min(), series.max(), 200)
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(series.values)
        fig = go.Figure(go.Scatter(x=x, y=kde(x), mode="lines", fill="tozeroy", line=dict(width=3)))
        return _base_layout(fig, "Distribution (KDE)", "Value", "Density")
    if kind == "box":
        fig = go.Figure(go.Box(y=series, name="", boxmean=True))
        return _base_layout(fig, "Box Plot", "", "Value")
    if kind == "violin":
        fig = go.Figure(go.Violin(y=series, box_visible=True, meanline_visible=True))
        return _base_layout(fig, "Violin Plot", "", "Value")
    # histogram with KDE overlay
    hist = go.Figure()
    hist.add_trace(go.Histogram(x=series, nbinsx=bins, name="Frequency", opacity=0.7))
    hist.add_trace(go.Scatter(
        x=series.sort_values(),
        y=(series.value_counts().rank(pct=True)).reindex(series.sort_values()).values,
        mode="lines", yaxis="y2", name="ECDF", line=dict(width=2),
    ))
    hist.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Cumulative"))
    return _base_layout(hist, "Histogram", "Value", "Frequency")


def ecdf_chart(series: pd.Series) -> go.Figure:
    s = pd.to_numeric(series, errors="coerce").dropna().sort_values()
    if s.empty:
        return go.Figure()
    y = np.arange(1, len(s) + 1) / len(s)
    fig = go.Figure(go.Scatter(x=s, y=y, mode="lines", line=dict(width=2)))
    return _base_layout(fig, "Empirical CDF", "Value", "Cumulative probability")


# ---------------------------------------------------------------- Scatter / regression
def scatter_chart(df: pd.DataFrame, x: str, y: str, color: pd.Series = None,
                  size: str = None) -> go.Figure:
    sub = df[[x, y]].copy()
    sub[x] = pd.to_numeric(sub[x], errors="coerce")
    sub[y] = pd.to_numeric(sub[y], errors="coerce")
    fig = px.scatter(sub, x=x, y=y, color=color if color is not None else None,
                     size=size, opacity=0.7, trendline="ols")
    return _base_layout(fig, f"{x} vs {y}", x, y)


# ---------------------------------------------------------------- Bar / value-by-group
def bar_chart(series: pd.Series, title: str = "Bar Chart", horizontal: bool = True) -> go.Figure:
    data = series.sort_values(ascending=False)
    if horizontal:
        fig = go.Figure(go.Bar(x=data.values, y=[str(i) for i in data.index], orientation="h",
                               marker_color=_PALETTE[0]))
        return _base_layout(fig, title, "Value", "")
    fig = go.Figure(go.Bar(x=[str(i) for i in data.index], y=data.values, marker_color=_PALETTE[0]))
    return _base_layout(fig, title, "", "Value")


def grouped_bar_chart(df: pd.DataFrame, x: str, y: str, color: str, agg: str = "mean") -> go.Figure:
    fig = px.bar(df, x=x, y=y, color=color, barmode="group", color_discrete_sequence=_PALETTE,
                 text_auto=True)
    return _base_layout(fig, f"{y} by {x} grouped by {color}", x, y)


def comparison_bar(df: pd.DataFrame, group_col: str, value_col: str, agg: str = "mean") -> go.Figure:
    from analysis.comparison import compare_groups
    table = compare_groups(df, group_col, value_col, agg=agg, top_n=15)
    fig = px.bar(table, x="group", y="value", color="group", color_discrete_sequence=_PALETTE,
                 title=f"{value_col} by {group_col} ({agg})")
    return _base_layout(fig, f"{value_col} by {group_col}", group_col, value_col)


# ---------------------------------------------------------------- Composition
def composition_chart(table: pd.DataFrame, kind: str = "donut", value_col: str = "share_pct") -> go.Figure:
    if kind == "treemap":
        fig = px.treemap(table, path=["category"], values=value_col, title="Composition Treemap")
        return _base_layout(fig, "Composition Treemap", "", "")
    if kind == "stacked_bar":
        fig = px.bar(table, x="category", y=value_col, color="category",
                     color_discrete_sequence=_PALETTE)
        return _base_layout(fig, "Composition (Stacked)", "", value_col)
    # donut / pie
    fig = go.Figure(go.Pie(
        labels=table["category"], values=table["value"] if "value" in table else table[value_col],
        hole=0.4 if kind == "donut" else 0.0, textinfo="percent+label",
    ))
    fig.update_layout(title="Composition (Donut)", template="plotly_white")
    return fig


# ---------------------------------------------------------------- Correlation heatmap
def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale="RdBu", zmid=0, text=np.round(corr.values, 2), texttemplate="%{text}",
        colorbar=dict(title="r"),
    ))
    fig.update_layout(title="Correlation Heatmap", template="plotly_white", height=550)
    return fig


# ---------------------------------------------------------------- Time series
def time_series_chart(index, series, title="Time Series", ma_series=None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=index, y=series, mode="lines+markers", name="Value",
                             line=dict(width=2)))
    if ma_series is not None:
        fig.add_trace(go.Scatter(x=index, y=ma_series, mode="lines", name="Moving Average",
                                 line=dict(width=2, dash="dot")))
    return _base_layout(fig, title, "Date", "Value")


def forecast_chart(index, history, forecast_values, title="Forecast") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=index, y=history, mode="lines", name="History", line=dict(width=2)))
    future_index = pd.date_range(start=index[-1], periods=len(forecast_values) + 1, freq="D")[1:]
    fig.add_trace(go.Scatter(x=list(future_index), y=forecast_values, mode="lines+markers",
                             name="Forecast", line=dict(width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=list(future_index), y=forecast_values,
                             mode="lines", line=dict(color="rgba(0,0,0,0)"),
                             showlegend=False, fill="tonexty", fillcolor="rgba(31,119,180,0.15)"))
    return _base_layout(fig, title, "Date", "Value")


# ---------------------------------------------------------------- Missingness
def missingness_bar(missing_counts: pd.Series) -> go.Figure:
    data = missing_counts.sort_values(ascending=False)
    fig = go.Figure(go.Bar(x=[str(i) for i in data.index], y=data.values, marker_color=_PALETTE[1]))
    fig.update_layout(xaxis_tickangle=-45)
    return _base_layout(fig, "Missing Values by Column", "", "Missing count")


# ---------------------------------------------------------------- Sentiment
def sentiment_donut(counts: Dict[str, int]) -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=list(counts.keys()), values=list(counts.values()), hole=0.4,
        textinfo="percent+label", marker=dict(colors=["#2ca02c", "#d62728", "#aaa"]),
    ))
    fig.update_layout(title="Sentiment Distribution", template="plotly_white")
    return fig
