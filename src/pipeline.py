"""
pipeline.py
-----------
End-to-end orchestration, mirroring the Data Output Requirements:
  Raw -> Cleaned -> Transformed -> Failed -> Hashed
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from extract import download_datasets, load_raw_files
from profile import profile_dataset, print_profile
from validate import validate_dataset
from transform import (
    compute_remaining_lease,
    dedup_composite_key,
    flag_price_anomalies,
    build_resale_identifier,
    dedup_by_identifier,
)
from hash_utils import add_hashed_identifier

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("hdb_etl.pipeline")


def run(raw_dir: Path, output_dir: Path, attempt_download: bool = True) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("raw", "cleaned", "transformed", "failed", "hashed"):
        (output_dir / sub).mkdir(exist_ok=True)

    all_failed_frames = []

    # ---------------- RAW ----------------
    if attempt_download:
        download_datasets(raw_dir)
    raw_df = load_raw_files(raw_dir)
    raw_df.to_csv(output_dir / "raw" / "master_raw.csv", index=False)
    logger.info("RAW: %d rows written", len(raw_df))

    # ---------------- PROFILE ----------------
    report = profile_dataset(raw_df)

    # ---------------- CLEANED (validation) ----------------
    passed_df, failed_validation_df = validate_dataset(raw_df, report)
    if len(failed_validation_df):
        all_failed_frames.append(failed_validation_df)
    logger.info("VALIDATION: %d passed, %d failed", len(passed_df), len(failed_validation_df))

    # ---------------- CLEANED (composite-key dedup) ----------------
    deduped_df, dup_discarded_df = dedup_composite_key(passed_df)
    if len(dup_discarded_df):
        all_failed_frames.append(dup_discarded_df)
    logger.info("DEDUP (composite key): %d kept, %d discarded", len(deduped_df), len(dup_discarded_df))

    # ---------------- CLEANED (remaining lease + anomaly flag) ----------------
    cleaned_df = compute_remaining_lease(deduped_df)
    cleaned_df = flag_price_anomalies(cleaned_df)
    cleaned_df.to_csv(output_dir / "cleaned" / "master_cleaned.csv", index=False)
    logger.info("CLEANED: %d rows written (%d flagged as price anomalies, not removed)",
                len(cleaned_df), cleaned_df["is_price_anomaly"].sum())

    # ---------------- TRANSFORMED (Resale Identifier) ----------------
    with_id_df = build_resale_identifier(cleaned_df)
    transformed_df, id_dup_discarded_df = dedup_by_identifier(with_id_df)
    if len(id_dup_discarded_df):
        all_failed_frames.append(id_dup_discarded_df)
    transformed_df.to_csv(output_dir / "transformed" / "master_transformed.csv", index=False)
    logger.info("TRANSFORMED: %d rows written, %d discarded for duplicate identifier",
                len(transformed_df), len(id_dup_discarded_df))

    # ---------------- HASHED ----------------
    hashed_df = add_hashed_identifier(transformed_df)
    hashed_df.to_csv(output_dir / "hashed" / "master_hashed.csv", index=False)
    logger.info("HASHED: %d rows written", len(hashed_df))

    # ---------------- FAILED ----------------
    if all_failed_frames:
        failed_df = pd.concat(all_failed_frames, axis=0, ignore_index=True, sort=False)
    else:
        failed_df = pd.DataFrame(columns=list(raw_df.columns) + ["_fail_reason"])
    failed_df.to_csv(output_dir / "failed" / "master_failed.csv", index=False)
    logger.info("FAILED: %d rows written", len(failed_df))

    return {
        "report": report,
        "raw_df": raw_df,
        "cleaned_df": cleaned_df,
        "transformed_df": transformed_df,
        "hashed_df": hashed_df,
        "failed_df": failed_df,
    }


if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent
    result = run(base / "data" / "raw", base / "output", attempt_download=True)
    print_profile(result["report"])
