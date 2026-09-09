"""Generate a sample CSV for manual testing."""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
idx = pd.date_range("2017-01-01", "2025-12-31", freq="D")
t = np.arange(len(idx))
trend = 100 + 0.01 * t
seasonal = 8 * np.sin(2 * np.pi * t / 365.25) + 3 * np.sin(2 * np.pi * t / 7)
noise = rng.normal(0, 2.5, size=len(idx))
for i in range(1, len(noise)):
    noise[i] = 0.4 * noise[i - 1] + noise[i]

df = pd.DataFrame({"date": idx, "value": trend + seasonal + noise})
out = Path(__file__).resolve().parent / "sample_data.csv"
df.to_csv(out, index=False)
print(f"Wrote {out} ({len(df)} rows)")
