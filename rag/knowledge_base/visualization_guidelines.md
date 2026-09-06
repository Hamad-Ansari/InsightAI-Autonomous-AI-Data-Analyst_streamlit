# Visualization Guidelines

Choose the chart that matches the data types. Avoid inappropriate charts.

- **Numeric + Numeric** → scatter plot (optionally with regression line).
- **Categorical + Numeric** → bar chart (mean/sum of metric per category).
- **Single Numeric** → histogram + KDE, box plot, violin plot, ECDF.
- **Datetime + Numeric** → line chart with optional moving average.
- **Categorical + Categorical** → grouped or stacked bar chart.
- **Correlation matrix** → heatmap (RdBu, zero-centered).
- **Composition** → donut, treemap or stacked bar.

## Best practices
- Always label axes and add a descriptive title.
- Keep charts interactive (hover, zoom, legend) where possible.
- Order categorical bars by value for quick ranking.
- Highlight the strongest patterns, but never overstate.
