"""
validate.py
-----------
Data Quality Requirement #3: validation rules for Date, Town, Flat Type,
Flat Model, storey_range -- derived from the statistical properties of the
master dataset (see profile.py), not hardcoded external lookup lists.

Design principle: "statistically derived" means each field's set of
acceptable values / formats is learned FROM the data (frequency
distribution) rather than an externally maintained enum. This makes the
pipeline robust to legitimate new categories appearing in future data
(e.g. a new flat_model) while still catching one-off typos / garbage
values, which show up as extreme long-tail rarities.

Also implements Data Quality Requirement #7 (additional cleaning rules):
  - floor_area_sqm must be a positive number within a sane range
  - resale_price must be a positive number
  - block / street_name must be non-empty

Every row that fails ANY rule is routed to the "failed" dataset with a
`_fail_reason` column explaining why (for auditability), per Data Output
Requirement (failed dataset).
"""
from __future__ import annotations

import re
import pandas as pd

from profile import RARE_VALUE_FREQ_THRESHOLD

DATE_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
STOREY_RANGE_RE = re.compile(r"^\d{2} TO \d{2}$")

# Sane physical bounds for HDB flats, used as a backstop sanity check
# (Data Quality Requirement #7). These are wide on purpose -- they exist to
# catch data-entry errors (e.g. 0 sqm, negative price), not to encode
# business judgement about "normal" flat sizes (that's the anomaly
# heuristic in transform.py, which is softer/statistical).
MIN_FLOOR_AREA_SQM = 20
MAX_FLOOR_AREA_SQM = 500
MIN_RESALE_PRICE = 1000


def derive_valid_categories(report: dict, col: str) -> set:
    """A category value is considered 'valid' if it is NOT in the
    statistically-derived rare/long-tail set for that column."""
    rare = set(report.get(col, {}).get("rare_values", []))
    all_values = set(report.get(col, {}).get("top_values", {}).keys())
    # top_values only holds the head(10); rebuild full valid set = seen values - rare values
    # (caller passes the full value_counts index via report; see validate_dataset)
    return all_values, rare


def validate_dataset(df: pd.DataFrame, report: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply validation rules to `df`. Returns (passed_df, failed_df).
    failed_df carries an extra `_fail_reason` column (one row can have
    multiple reasons joined by '; ').
    """
    reasons = pd.Series([""] * len(df), index=df.index)

    def flag(mask: pd.Series, reason: str):
        nonlocal reasons
        reasons.loc[mask] = reasons.loc[mask].where(reasons.loc[mask] == "", reasons.loc[mask] + "; ")
        reasons.loc[mask] = reasons.loc[mask] + reason

    # --- Date (month) ---
    bad_date_format = ~df["month"].astype(str).str.match(DATE_RE)
    flag(bad_date_format, "invalid month format (expected YYYY-MM)")

    # --- Town: statistically rare values flagged ---
    town_vc = df["town"].value_counts(dropna=True)
    town_freq = town_vc / len(df)
    rare_towns = set(town_freq[town_freq < RARE_VALUE_FREQ_THRESHOLD].index)
    bad_town = df["town"].isna() | df["town"].isin(rare_towns)
    flag(bad_town, "town is null or a statistical rarity/outlier (<0.05% frequency)")

    # --- Flat Type ---
    ft_vc = df["flat_type"].value_counts(dropna=True)
    ft_freq = ft_vc / len(df)
    rare_ft = set(ft_freq[ft_freq < RARE_VALUE_FREQ_THRESHOLD].index)
    bad_ft = df["flat_type"].isna() | df["flat_type"].isin(rare_ft)
    flag(bad_ft, "flat_type is null or a statistical rarity/outlier (<0.05% frequency)")

    # --- Flat Model ---
    fm_vc = df["flat_model"].value_counts(dropna=True)
    fm_freq = fm_vc / len(df)
    rare_fm = set(fm_freq[fm_freq < RARE_VALUE_FREQ_THRESHOLD].index)
    bad_fm = df["flat_model"].isna() | df["flat_model"].isin(rare_fm)
    flag(bad_fm, "flat_model is null or a statistical rarity/outlier (<0.05% frequency)")

    # --- storey_range: format + internal consistency (lower <= upper, 5-storey bands) ---
    bad_sr_format = ~df["storey_range"].astype(str).str.match(STOREY_RANGE_RE)
    flag(bad_sr_format, "invalid storey_range format (expected 'NN TO NN')")

    sr_valid_fmt = df["storey_range"].astype(str).str.match(STOREY_RANGE_RE)
    lower = pd.to_numeric(df["storey_range"].astype(str).str.slice(0, 2), errors="coerce")
    upper = pd.to_numeric(df["storey_range"].astype(str).str.slice(6, 8), errors="coerce")
    bad_sr_logic = sr_valid_fmt & (lower > upper)
    flag(bad_sr_logic, "storey_range lower bound exceeds upper bound")

    # --- Additional cleaning rules (Data Quality Requirement #7) ---
    floor_area = pd.to_numeric(df["floor_area_sqm"], errors="coerce")
    bad_floor_area = floor_area.isna() | (floor_area < MIN_FLOOR_AREA_SQM) | (floor_area > MAX_FLOOR_AREA_SQM)
    flag(bad_floor_area, f"floor_area_sqm missing/non-numeric or outside [{MIN_FLOOR_AREA_SQM},{MAX_FLOOR_AREA_SQM}] sqm")

    price = pd.to_numeric(df["resale_price"], errors="coerce")
    bad_price = price.isna() | (price < MIN_RESALE_PRICE)
    flag(bad_price, f"resale_price missing/non-numeric or below ${MIN_RESALE_PRICE}")

    lease_commence = pd.to_numeric(df["lease_commence_date"], errors="coerce")
    bad_lease = lease_commence.isna() | (lease_commence < 1960) | (lease_commence > pd.Timestamp.today().year)
    flag(bad_lease, "lease_commence_date missing/non-numeric or implausible")

    bad_block = df["block"].isna() | (df["block"].astype(str).str.strip() == "")
    flag(bad_block, "block is missing")

    bad_street = df["street_name"].isna() | (df["street_name"].astype(str).str.strip() == "")
    flag(bad_street, "street_name is missing")

    failed_mask = reasons != ""
    failed_df = df.loc[failed_mask].copy()
    failed_df["_fail_reason"] = reasons.loc[failed_mask]
    passed_df = df.loc[~failed_mask].copy()

    return passed_df, failed_df
