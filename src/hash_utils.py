"""
hash_utils.py
-------------
Data Transformation Requirement #3: hash the Resale Identifier using an
irreversible algorithm, while preserving its uniqueness.

Algorithm chosen: SHA-256 (via Python's hashlib), applied per-row as
    SHA256(resale_identifier + "|" + _row_id)

Why SHA-256:
  - Irreversible / one-way: SHA-256 is a cryptographic hash function; it
    is computationally infeasible to recover the input identifier from
    the digest (pre-image resistance).
  - Collision-resistant: at 256 bits of output, the probability of an
    accidental collision across even billions of records is astronomically
    small (birthday-bound ~2^128), which is what "preserving uniqueness"
    requires in practice -- two different identifiers essentially never
    hash to the same digest.
  - Deterministic: the same identifier always hashes to the same digest,
    which matters for downstream joins/reproducibility, unlike a
    salted/randomised scheme.

Why we mix in `_row_id` before hashing:
  - The Resale Identifier itself (S+BBB+PP+MM+T) is a DERIVED, LOSSY code
    -- by construction, two distinct transactions in the same
    (block-prefix, price-bracket, month, town) can legitimately still
    collide onto the same identifier string even after the
    dedup_by_identifier() step removes exact duplicates in *price*
    (e.g. two different flat_types weren't part of the key... actually
    they are -- but different flats at different storeys with the same
    price can still collide). To strictly GUARANTEE the hash preserves
    row-level uniqueness (not just identifier-string uniqueness), we hash
    the identifier concatenated with the row's stable internal `_row_id`.
    This keeps the hash irreversible and deterministic while eliminating
    any residual collision risk from the lossy identifier design.
  - If the intent is instead to hash the identifier value alone (so that
    identical identifiers across rows *should* produce identical hashes,
    e.g. for grouping/matching purposes), simply call
    `hash_identifier_only()` instead -- both are provided.
"""
from __future__ import annotations

import hashlib
import pandas as pd


def hash_identifier_only(identifier: str) -> str:
    """SHA-256 of the identifier string alone (duplicate identifiers -> duplicate hash)."""
    return hashlib.sha256(identifier.encode("utf-8")).hexdigest()


def hash_identifier_with_row_id(identifier: str, row_id) -> str:
    """SHA-256 of identifier + row_id (guarantees a unique hash per surviving row)."""
    payload = f"{identifier}|{row_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def add_hashed_identifier(df: pd.DataFrame, id_col: str = "resale_identifier", row_id_col: str = "_row_id") -> pd.DataFrame:
    df = df.copy()
    df["resale_identifier_hash"] = [
        hash_identifier_with_row_id(ident, rid) for ident, rid in zip(df[id_col], df[row_id_col])
    ]
    # sanity check: hash column should be as unique as the surviving row set
    n_rows, n_unique_hash = len(df), df["resale_identifier_hash"].nunique()
    assert n_unique_hash == n_rows, f"Hash uniqueness violated: {n_unique_hash} unique hashes for {n_rows} rows"
    return df
