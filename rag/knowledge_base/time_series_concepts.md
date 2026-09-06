# Time-Series Analysis Concepts

Time-series data is ordered by a date/time field. Key elements:

- **Trend** — long-term direction (increasing / decreasing / flat).
- **Seasonality** — repeating patterns (daily, weekly, monthly, yearly).
- **Cycles** — longer, non-fixed oscillations.
- **Noise** — random variation.

## Useful transformations
- **Rolling mean (moving average)** — smooths short-term noise to reveal trend.
- **Growth rate** — percentage change between periods.
- **Period-over-period change** — comparison with the prior period.

## Forecasting methods
- **Naive** — carry the last value forward.
- **Moving average** — average of recent values.
- **Exponential Smoothing (Holt-Winters)** — weighted average with damped trend.
- **ARIMA / SARIMA** — autoregressive, differenced models; SARIMA adds seasonality.

## Critical rules
- Clearly separate **historical analysis** from **forecasting**.
- Never present a forecast as certain. Forecasts are estimates based on past data.
- Require sufficient history: a few months or more before forecasting.
