"""
profile.py
----------
Data Profiling (Part 1, Data Quality Requirement #2).

Rather than pulling in a heavyweight framework (ydata-profiling / Great
Expectations) whose HTML output doesn't fit well into a reviewable
notebook diff, we implement a lightweight, transparent profiler. This
also means every downstream validation rule (validate.py) is derived
directly and traceably from these same statistics -- satisfying
requirement #3 ("validation rules ... based on the statistical
properties of this master dataset").

For each column we compute:
  - dtype (as-loaded), non-null count, null count, null %
  - distinct count, distinct %
  - for categorical-looking columns: value_counts + frequency-based
    "long tail" cut-off (values below the threshold are candidates for
    invalid/misspelt values)
  - for numeric-looking columns: min / max / mean / std / quartiles
"""
from __future__ import annotations

import pandas as pd
import numpy as np


CATEGORICAL_COLS = ["town", "flat_type", "flat_model", "storey_range"]
NUMERIC_COLS = ["floor_area_sqm", "resale_price", "lease_commence_date"]
RARE_VALUE_FREQ_THRESHOLD = 0.0005  # values occurring in <0.05% of rows are flagged as "rare" candidates


def profile_dataset(df: pd.DataFrame) -> dict:
    """Return a dict-of-dicts profile report, one entry per column."""
    n = len(df)
    report = {}

    for col in df.columns:
        if col.startswith("_"):
            continue
        s = df[col]
        col_report = {
            "dtype": str(s.dtype),
            "non_null": int(s.notna().sum()),
            "nulls": int(s.isna().sum()),
            "null_pct": round(100 * s.isna().mean(), 3),
            "distinct": int(s.nunique(dropna=True)),
            "distinct_pct": round(100 * s.nunique(dropna=True) / n, 3) if n else 0,
        }

        if col in CATEGORICAL_COLS:
            vc = s.value_counts(dropna=True)
            freq = vc / n
            col_report["top_values"] = vc.head(10).to_dict()
            col_report["rare_values"] = freq[freq < RARE_VALUE_FREQ_THRESHOLD].index.tolist()

        if col in NUMERIC_COLS:
            numeric = pd.to_numeric(s, errors="coerce")
            col_report["non_numeric_count"] = int(numeric.isna().sum() - s.isna().sum())
            if numeric.notna().any():
                col_report.update({
                    "min": float(numeric.min()),
                    "max": float(numeric.max()),
                    "mean": round(float(numeric.mean()), 2),
                    "std": round(float(numeric.std()), 2),
                    "p25": float(numeric.quantile(0.25)),
                    "p50": float(numeric.quantile(0.50)),
                    "p75": float(numeric.quantile(0.75)),
                })

        report[col] = col_report

    return report


def print_profile(report: dict) -> None:
    for col, stats in report.items():
        print(f"\n=== {col} ===")
        for k, v in stats.items():
            if k in ("top_values",):
                print(f"  {k}:")
                for val, cnt in v.items():
                    print(f"      {val!r}: {cnt}")
            elif k == "rare_values" and v:
                print(f"  {k} ({len(v)}): {v[:20]}{' ...' if len(v) > 20 else ''}")
            elif k == "rare_values":
                print(f"  {k}: none")
            else:
                print(f"  {k}: {v}")
