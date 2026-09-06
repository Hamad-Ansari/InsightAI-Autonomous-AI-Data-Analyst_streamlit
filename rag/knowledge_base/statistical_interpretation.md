# Statistical Interpretation Guide

## Measures of central tendency
- **Mean** — arithmetic average; sensitive to outliers.
- **Median** — midpoint; robust to outliers. Prefer median when data is skewed.
- **Mode** — most frequent value; useful for categorical data.

## Measures of spread
- **Standard deviation** — average distance from the mean; higher = more spread.
- **Variance** — squared standard deviation.
- **Interquartile range (IQR)** — difference between Q3 and Q1; robust spread.
- **Range** — max minus min.

## Shape
- **Skewness** — symmetry. Zero = symmetric, positive = long right tail
  (right/positive skew), negative = long left tail (left/negative skew).
  Values beyond roughly +/-0.5 indicate noticeable skew.
- **Kurtosis** — tailedness. High positive = heavy tails (outlier-prone);
  near zero = normal-like; negative = light tails.

## Correlation
- **Pearson r** — linear association between -1 and +1. r close to 0 = weak.
- **Spearman rho** — monotonic (rank-based) association; robust to outliers and
  non-linear monotone relationships.
- Correlation does **not** imply causation.
- For categorical association use **Cramér's V** (0 = none, 1 = perfect).
