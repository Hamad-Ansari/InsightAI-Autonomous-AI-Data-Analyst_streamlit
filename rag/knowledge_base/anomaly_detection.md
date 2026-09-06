# Anomaly & Risk Detection

Anomalies are unusual records or values that differ significantly from the norm.

## Methods
- **IQR (Tukey fences)** — values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR are
  potential outliers; make the multiplier larger (e.g. 3.0) for extreme outliers.
- **Z-score** — values with |z| > 3 are unusual; |z| > 4.5 are extreme. Requires
  roughly normal data.
- **Isolation Forest** — an ensemble that isolates anomalies; works on
  multi-dimensional numeric data without distribution assumptions.

## Guidelines
- Detect and **classify**, do not silently delete.
- Distinguish genuine anomalies (fraud, errors, spikes) from legitimate
  business events (seasonal peaks, one-off promotions).
- Report count and percentage per column and flag driving columns.
