"""
transform.py
------------
Implements:
  Data Quality Requirement #4: remaining lease recomputation
  Data Quality Requirement #5: composite-key dedup (keep higher price)
  Data Quality Requirement #6: anomalous resale price heuristic
  Data Transformation Requirement #1: "Resale Identifier" construction
  Data Transformation Requirement #2: post-identifier dedup (keep higher price)
"""
from __future__ import annotations

import re
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

LEASE_YEARS = 99


# ---------------------------------------------------------------------------
# Data Quality Requirement #4: remaining lease
# ---------------------------------------------------------------------------
def compute_remaining_lease(df: pd.DataFrame, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    Assumption: lease commences on 1 Jan of `lease_commence_date` (only the
    year is given in the source data). Lease expires 99 years later.
    Remaining lease = (lease_start + 99y) - as_of, floored to whole months
    (i.e. partial months are dropped, matching HDB's own convention of
    reporting remaining lease as "X years Y months").

    Adds columns: remaining_lease_years, remaining_lease_months,
    remaining_lease_display (e.g. "74 years 03 months").
    """
    as_of = as_of or pd.Timestamp.today().normalize()
    df = df.copy()

    lease_commence_year = pd.to_numeric(df["lease_commence_date"], errors="coerce")

    years_list, months_list, display_list = [], [], []
    for y in lease_commence_year:
        if pd.isna(y):
            years_list.append(np.nan)
            months_list.append(np.nan)
            display_list.append(None)
            continue
        lease_start = pd.Timestamp(year=int(y), month=1, day=1)
        lease_expiry = lease_start + pd.DateOffset(years=LEASE_YEARS)
        delta = relativedelta(lease_expiry, as_of)
        # Floor down: if expiry already passed, clip at 0
        total_months = max(delta.years * 12 + delta.months, 0)
        rem_years, rem_months = divmod(total_months, 12)
        years_list.append(rem_years)
        months_list.append(rem_months)
        display_list.append(f"{rem_years} years {rem_months:02d} months")

    df["remaining_lease_years"] = years_list
    df["remaining_lease_months"] = months_list
    df["remaining_lease_display"] = display_list
    return df


# ---------------------------------------------------------------------------
# Data Quality Requirement #5: composite-key dedup
# ---------------------------------------------------------------------------
def dedup_composite_key(df: pd.DataFrame, price_col: str = "resale_price") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Composite key = every column except resale_price (and internal
    lineage columns prefixed with `_`). Where two rows share an identical
    composite key but differ in price, keep the HIGHER price and route the
    rest to the "failed" (discarded-duplicate) set.
    """
    key_cols = [c for c in df.columns if c != price_col and not c.startswith("_")]
    price_numeric = pd.to_numeric(df[price_col], errors="coerce")
    work = df.copy()
    work["_price_numeric"] = price_numeric

    # Rank rows within each composite-key group by price, descending.
    work["_rank"] = work.groupby(key_cols, dropna=False)["_price_numeric"].rank(method="first", ascending=False)

    kept = work[work["_rank"] == 1].drop(columns=["_price_numeric", "_rank"])
    discarded = work[work["_rank"] != 1].drop(columns=["_price_numeric", "_rank"]).copy()
    if len(discarded):
        discarded["_fail_reason"] = "duplicate composite key (all columns except resale_price identical); lower price discarded"

    return kept, discarded


# ---------------------------------------------------------------------------
# Data Quality Requirement #6: anomalous price heuristic
# ---------------------------------------------------------------------------
def flag_price_anomalies(df: pd.DataFrame, iqr_multiplier: float = 3.0) -> pd.DataFrame:
    """
    Heuristic (documented per requirement #6):
    We compute price-per-square-metre (a fairer normalising metric than
    raw price, since flat size varies a lot) grouped by (town, flat_type).
    Within each group we flag a transaction as a potential anomaly if its
    price-per-sqm falls outside [Q1 - k*IQR, Q3 + k*IQR], using k=3.0
    (a wider-than-usual multiplier of 3x IQR, vs. the conventional 1.5x,
    since resale prices are legitimately right-skewed and we want to flag
    only GENUINE outliers for manual review, not just the normal tail of
    the distribution -- this is a soft "flag for review" step, not an
    automatic rejection, so it does not remove rows from the cleaned set).

    Adds columns: price_per_sqm, is_price_anomaly, price_anomaly_reason.
    This is deliberately separate from the hard validation rules in
    validate.py: anomalies are suspicious-but-plausible, not invalid data.
    """
    df = df.copy()
    price = pd.to_numeric(df["resale_price"], errors="coerce")
    area = pd.to_numeric(df["floor_area_sqm"], errors="coerce")
    df["price_per_sqm"] = price / area

    def _flag_group(g: pd.Series) -> pd.Series:
        q1, q3 = g.quantile(0.25), g.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr
        return (g < lower) | (g > upper)

    df["is_price_anomaly"] = (
        df.groupby(["town", "flat_type"])["price_per_sqm"]
        .transform(_flag_group)
        .fillna(False)
    )
    df["price_anomaly_reason"] = np.where(
        df["is_price_anomaly"],
        f"price_per_sqm outside {iqr_multiplier}x IQR of its (town, flat_type) group",
        "",
    )
    return df


# ---------------------------------------------------------------------------
# Data Transformation Requirement #1: Resale Identifier
# ---------------------------------------------------------------------------
def _block_digits(block: str) -> str:
    """First 3 digits of block (non-digit chars stripped), zero-padded left."""
    digits = re.sub(r"\D", "", str(block))
    digits3 = digits[:3]
    return digits3.zfill(3)


def build_resale_identifier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resale Identifier = S + BBB + PP + MM + T, where:
      S   : literal "S"
      BBB : first 3 digits of `block` (non-digits stripped), zero-padded
      PP  : first 2 digits of the AVERAGE resale price for this row's
            (year-month, town, flat_type) group
      MM  : month portion of this row's `month` (YYYY-MM -> MM)
      T   : first character of `town`
    """
    df = df.copy()
    price_numeric = pd.to_numeric(df["resale_price"], errors="coerce")

    group_avg = (
        df.assign(_price=price_numeric)
        .groupby(["month", "town", "flat_type"])["_price"]
        .transform("mean")
    )

    def _avg_price_prefix(avg: float) -> str:
        # First 2 digits of the (rounded-down, integer) average price.
        int_avg = str(int(avg))
        return int_avg[:2].zfill(2)

    block_part = df["block"].apply(_block_digits)
    price_part = group_avg.apply(_avg_price_prefix)
    month_part = df["month"].astype(str).str.slice(5, 7)  # 'YYYY-MM' -> 'MM'
    town_part = df["town"].astype(str).str.strip().str[0].str.upper()

    df["resale_identifier"] = "S" + block_part + price_part + month_part + town_part
    return df


def dedup_by_identifier(df: pd.DataFrame, id_col: str = "resale_identifier", price_col: str = "resale_price") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Data Transformation Requirement #2: after constructing the identifier,
    two genuinely different transactions can collide onto the same
    identifier (it is a derived, lossy code -- not a hash of the full
    row). Where that happens, keep the higher price and discard the rest.
    """
    price_numeric = pd.to_numeric(df[price_col], errors="coerce")
    work = df.copy()
    work["_price_numeric"] = price_numeric
    work["_rank"] = work.groupby(id_col)["_price_numeric"].rank(method="first", ascending=False)

    kept = work[work["_rank"] == 1].drop(columns=["_price_numeric", "_rank"])
    discarded = work[work["_rank"] != 1].drop(columns=["_price_numeric", "_rank"]).copy()
    if len(discarded):
        discarded["_fail_reason"] = "duplicate resale_identifier; lower price discarded"

    return kept, discarded
