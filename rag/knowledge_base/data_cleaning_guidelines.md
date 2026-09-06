# Data Cleaning Guidelines

Cleaning should be **conservative and transparent**. Every transformation must
be logged and the original data preserved.

## Safe automatic operations
- Trim surrounding whitespace from string values.
- Normalise obvious inconsistent capitalisation in categorical values.
- Convert recognisable date strings to datetime.
- Coerce numeric strings (e.g. "1,234" or "$50") to numeric.
- Remove exact duplicate rows.
- Drop fully-empty or constant columns.

## Missing values
- **Numeric** — impute with median (robust) or mean (configurable).
- **Categorical** — impute with the mode.
- **Text** — treat empty/null as missing; do not invent content.
- **Datetime** — do not blindly impute. Forward/back-fill or leave as is.

## Outliers
Classify values as Normal / Potential / Extreme using IQR or Z-score, but do
**not** automatically delete them. Offer options to Keep, Remove or Winsorise.
