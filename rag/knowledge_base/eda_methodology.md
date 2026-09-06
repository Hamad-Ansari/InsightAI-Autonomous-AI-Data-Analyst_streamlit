# Exploratory Data Analysis (EDA) Methodology

Exploratory Data Analysis (EDA) is the process of examining a dataset to
summarise its main characteristics, often with visual methods. The goal is to
understand the structure, discover patterns, spot anomalies, test hypotheses and
formulate questions for deeper analysis before modelling.

## Core dimensions of EDA
1. **Composition** — "What is this data made of?" Examine the share of records
   across categories, segments, regions or product types using donut, treemap
   and stacked bar charts.
2. **Distribution** — "How are values spread?" Understand central tendency,
   dispersion, skewness, kurtosis and the presence of outliers using histograms,
   KDE, box and violin plots.
3. **Comparison** — "How do groups differ?" Compare a metric across categories
   using bar charts, grouped bars and ranking charts.
4. **Relationship** — "How are variables connected?" Explore correlation and
   association using heatmaps, scatter plots and regression lines.

## Key principles
- Always profile the dataset first: rows, columns, data types, cardinality,
  missingness and duplicates.
- Never let a model or LLM silently compute statistics — compute them with
  Python and interpret them with reasoning.
- Separate the original data from the cleaned data; never destroy the source.
