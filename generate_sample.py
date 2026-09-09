"""Generate a sample weekday CSV for manual testing."""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
idx = pd.bdate_range("2017-01-01", "2025-12-31")
t = np.arange(len(idx))
trend = 100 + 0.012 * t
seasonal = 8 * np.sin(2 * np.pi * t / 252) + 3 * np.sin(2 * np.pi * t / 5)
noise = rng.normal(0, 2.5, size=len(idx))
for i in range(1, len(noise)):
    noise[i] = 0.4 * noise[i - 1] + noise[i]

df = pd.DataFrame({"date": idx, "value": trend + seasonal + noise})
assert (df["date"].dt.dayofweek < 5).all()
out = Path(__file__).resolve().parent / "sample_data.csv"
df.to_csv(out, index=False)
print(f"Wrote {out} ({len(df)} weekday rows)")
