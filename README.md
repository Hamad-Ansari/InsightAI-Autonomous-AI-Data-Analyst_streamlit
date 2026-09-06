# InsightAI — Autonomous AI Data Analyst

InsightAI is a **local, free, production-style Python application** that lets a business user upload almost any tabular dataset and automatically perform data understanding, cleaning, EDA (composition / distribution / comparison / relationship), statistical analysis, sentiment analysis, time-series analysis, anomaly detection, visualization, a **RAG**-augmented **AI reasoning** layer, an interactive dashboard, and a final exportable report (HTML / PDF / JSON / CSV).

It is **not wired to one industry** — it works for e-commerce, retail, banking, finance, marketing, sales, HR, healthcare, manufacturing, education, logistics, customer support, social media, real estate, supply chain, SaaS and operations.

The architecture keeps data local end-to-end:

```
Dataset → Python/Pandas/sklearn/statsmodels → Local Ollama (reasoning) → Local RAG (context)
```

**The LLM never computes statistics.** Python computes every number; Ollama *explains* it.

---

## Features

- **Ingestion** — CSV, XLSX, XLS, TSV, Parquet; automatic encoding detection (UTF-8 / UTF-8-SIG / Latin-1); upload validation.
- **Schema detection** — classifies columns into NUMERIC, CATEGORICAL, DATETIME, TEXT, BOOLEAN, ID, CURRENCY, PERCENTAGE, UNKNOWN via heuristics.
- **Automatic cleaning** — trailing-whitespace trim, category-case normalisation, date parsing, numeric-string coercion, exact-duplicate removal, missing-value imputation, and dropping empty/constant columns. **The original dataset is never modified**; every change is logged.
- **Outliers** — IQR / Z-score / Isolation Forest classification (Normal / Potential / Extreme) with Keep / Remove / Winsorise options (never silently deleted).
- **15-step analysis pipeline** — business understanding → profiling → data-quality score (0–100) → cleaning → univariate → composition → distribution → comparison → relationship → missing-data analysis → time-series → sentiment → anomaly → AI insights → executive report.
- **Four analysis dimensions** — Composition, Distribution, Comparison, Relationship.
- **Automatic chart selection** — rule-based recommendation by data type.
- **Interactive Plotly visualizations** — hover, zoom, legend, downloadable images.
- **Sentiment analysis** — local lexicon scoring (VADER-compatible) with positive/negative words and themes.
- **Time-series analysis** — daily/weekly/monthly aggregation, rolling means, growth rates, seasonality, decomposition, and forecasting (naive / moving average / Exponential Smoothing / ARIMA). Historical analysis is kept separate from forecasting.
- **Anomaly & risk detection** — per-column reports and an anomaly table.
- **RAG knowledge base** — Sentence-Transformers embeddings + ChromaDB (with an in-memory TF-IDF fallback) over curated analytical-methodology documents.
- **AI business insights** — structured Executive Summary / Top Findings / Data Quality / Trends / Risks / Opportunities / Recommendations / Further Questions. Falls back to a **deterministic engine** when Ollama is unavailable.
- **Dashboard & reports** — KPI cards, executive summary, exportable HTML / PDF / JSON / CSV, cleaned-data download.

---

## Tech stack

| Layer | Tools |
|---|---|
| Language | Python 3.9+ |
| Web app | Streamlit |
| Data | pandas, NumPy |
| Statistics | SciPy, statsmodels, scikit-learn |
| Visualization | Plotly (Matplotlib/Seaborn optional) |
| AIModel | Ollama (any model, e.g. `llama3.1:8b`) |
| RAG | Sentence-Transformers, ChromaDB (TF-IDF fallback) |
| Schemas | Pydantic |
| Reports | ReportLab (PDF), Jinja-free HTML, JSON, CSV |
| Storage | Optional SQLite for history (extensible) |

---

## Project structure

```
InsightAI/
├── app.py                     # Streamlit UI
├── requirements.txt
├── .env.example
├── README.md
├── Dockerfile
├── docker-compose.yml
├── config/    settings.py
├── ingestion/ loader.py, schema_detector.py
├── cleaning/  cleaner.py, missing_values.py, duplicates.py, outliers.py
├── analysis/  profiler, quality, univariate, composition, distribution,
│              comparison, relationships, sentiment, timeseries, anomaly
├── visualization/ chart_selector.py, charts.py, dashboard.py
├── rag/       embeddings.py, ingest.py, retriever.py, knowledge_base/
├── llm/       ollama_client.py, prompts.py, insight_generator.py
├── reporting/ report_generator.py, html_report.py, pdf_report.py
├── agent/     analyst_agent.py, planner.py, tools.py
├── utils/     logger.py, validators.py, helpers.py
├── tests/     pytest suite
├── docs/      user_guide.html
└── sample_data/  generate_sample_data.py, sales.csv, ecommerce.csv, customer_feedback.csv
```

---

## Installation

### 1. Install Python (3.9–3.12 recommended)

Download from [python.org](https://www.python.org/downloads/). On Windows, tick **“Add Python to PATH”** during installation.

### 2. Install Ollama

Download from [ollama.com](https://ollama.com) and install. Then pull a model:

```bash
ollama pull llama3.1:8b
```

Verify it is running (default URL `http://localhost:11434`):

```bash
ollama serve
```

### 3. Clone / download the project

```bash
git clone <your-repo-url> InsightAI
cd InsightAI
```

### 4. Create and activate a virtual environment

**Windows**
```bat
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 5. Install requirements

```bash
pip install -r requirements.txt
```

> Heavy optional packages (`sentence-transformers`, `chromadb`) are used when
> present and fall back gracefully when absent. For the smallest install you can
> drop them from `requirements.txt`; the app still works.

### 6. Configure environment

```bash
cp .env.example .env
# edit .env if you wish
```

### 7. Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## First run

1. Click **“Load a sample dataset (sales)”** on the Home page, **or** upload your own CSV/XLSX/XLS/TSV/Parquet from the sidebar.
2. InsightAI automatically runs the pipeline and shows KPI cards.
3. Explore the sidebar pages: Dataset Info, Cleaning, EDA, Composition, Distribution, Comparison, Relationship, Sentiment, Time Series, AI Insights, RAG Knowledge, Dashboard, Final Report, Export, Settings.
4. If Ollama is running, AI Insights come from the local LLM; otherwise the deterministic engine produces the report.

---

## Example dataset workflow

| Dataset | What InsightAI reports |
|---|---|
| `sample_data/sales.csv` | Sales Analytics: composition by region/product, revenue distribution, time-series trend, correlation, anomalies |
| `sample_data/ecommerce.csv` | E-commerce: category mix, price distribution, sentiment from reviews |
| `sample_data/customer_feedback.csv` | Support: sentiment across topics/regions, satisfaction score |

---

## Testing

```bash
pip install pytest
python -m pytest -q
```

33 tests cover loading, schema detection, cleaning, outliers, composition, distribution, comparison, relationships, sentiment, time-series, chart selection, the agent, and report generation.

---

## Docker

Build and run the Streamlit app:

```bash
docker compose up --build
# or
docker build -t insightai .
docker run -p 8501:8501 insightai
```

> Ollama is expected to run **separately** on the host (or as its own service).
> Point the app at it via the `OLLAMA_BASE_URL` env var. GPU acceleration of the
> Ollama container is host-dependent (see Ollama docs).

---

## Troubleshooting

- **“Ollama is not running.”** — Start it: `ollama serve`, then reload the Settings page.
- **Performance** — use the Sampling option in Settings, and increase the cache.
- **Excel `.xls`** — needs `xlrd` (already in requirements).
- **Parquet** — needs `pyarrow` (already in requirements).
- **Chart/report image export** — needs `kaleido`; if absent the report omits images but still works.

---

## Privacy & security

- Data is processed **locally**. Nothing is uploaded to external APIs by default.
- Uploaded files are **never executed as code**; extensions and sizes are validated.
- API keys are kept in `.env` (never committed).
- The default pipeline sends dataset statistics only to a **local** Ollama server.

---

## Future improvements

- Persistent SQLite analysis history.
- Multi-dataset comparison.
- Configurable per-dataset PII redaction.
- Exportable JSON "analysis contracts" for reproducibility.

---

## License

MIT — free to use and modify. See the `LICENSE` file (add one for your project).
