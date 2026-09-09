# Time Series Backfill

Generate a missing **prefix** of a time series while preserving seasonal structure and the historical residual distribution. Includes a Streamlit web UI.

## Method

1. Infer calendar frequency and align the series to a regular index from the desired start date.
   **Weekdays-only mode (default):** use a business-day calendar (`B`); weekends are dropped from the input and never generated.
2. Decompose the observed part with **STL** (trend + seasonal + residual). Multiplicative mode works in log-space.
3. Extrapolate the trend backward with a local linear fit.
4. Replay the average seasonal profile.
5. Fill residuals with a **seasonal block bootstrap** from historical residuals.
6. Optionally blend the seam with the first observed value.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`).

## Usage (UI)

1. Use the built-in demo series, or upload a CSV with a date column and a numeric value column.
2. Set **Desired series start date** — must be before the first observed date.
3. Optionally tune model (`additive` / `multiplicative`), season period, bootstrap block size, and random seed.
4. Click **Generate missing prefix**.
5. Inspect the chart / distribution plots and download `series_backfilled.csv`.

## Usage (Python)

```python
import pandas as pd
from backfill import backfill_series, load_series_from_frame

df = pd.read_csv("sample_data.csv")
series = load_series_from_frame(df, "date", "value")

result = backfill_series(
    series,
    desired_start="2016-01-01",
    model="additive",
    weekdays_only=True,  # default: Mon–Fri only
    random_state=42,
)

print(result.series.head())
print(result.n_generated, result.freq, result.season_period)
```

## Sample data

```bash
python generate_sample.py   # writes sample_data.csv (weekdays, 2017–2025)
```

## Notes

- By default only **Monday–Friday** values are kept/generated (`weekdays_only=True`).
- If `desired_start` falls on a weekend, it is snapped to the next weekday.
- The series should be roughly regularly spaced (business-daily / daily / weekly / monthly, …).
- `desired_start` must be earlier than the first observed timestamp.
- Multiplicative mode requires strictly positive values.
