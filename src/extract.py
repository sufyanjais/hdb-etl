"""
extract.py
----------
Handles Dataset Extraction (Part 1, "Dataset Extraction" requirement).

data.gov.sg publishes HDB resale flat price transactions as SEVERAL separate
CSV files, split by registration-date vintage, e.g.:
    - Resale Flat Prices Based on Approval Date (1990-1999)
    - Resale Flat Prices Based on Approval Date (2000-Feb 2012)
    - Resale Flat Prices Based on Registration Date (Mar 2012-Dec 2014)
    - Resale Flat Prices Based on Registration Date (Jan 2015-Dec 2016)
    - Resale Flat Prices Based on Registration Date (from Jan 2017 onwards)

Each vintage can have a *different schema* (e.g. older files don't have
`floor_area_sqm` split out, or use `remaining_lease` instead of
`lease_commence_date`). To satisfy "combine into a single master dataset
that contains all attributes found in all files", we:

  1. Programmatically discover & download every file relevant to the
     requested date window (Jan 2012 - Dec 2016) from the data.gov.sg
     Collections API (no manual clicking).
  2. Read each CSV as-is (no manual edits).
  3. Union the schemas (outer join on columns) rather than intersecting,
     so no attribute is silently dropped.
  4. Tag every row with its `_source_file` for lineage/auditability.

NOTE ON SANDBOX EXECUTION: this environment has no outbound network access,
so `download_datasets()` below is provided in full for use in a networked
environment (e.g. your own machine / CI), but will gracefully no-op and
fall back to whatever CSV files already exist in RAW_DIR -- which is how
this notebook was actually executed, using the single sample file you
supplied (Mar 2012-Dec 2014). The pipeline logic downstream is written to
generalise to any number of additional vintage files dropped into RAW_DIR.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from urllib.request import urlopen, Request

import pandas as pd

logger = logging.getLogger("hdb_etl.extract")

# data.gov.sg dataset collection covering resale flat prices.
COLLECTION_ID = 189
COLLECTION_API = f"https://api-open.data.gov.sg/v1/public/api/collections/{COLLECTION_ID}/metadata"
POLL_DOWNLOAD_API = "https://api-open.data.gov.sg/v1/public/api/datasets/{dataset_id}/poll-download"


def download_datasets(raw_dir: Path, date_start: str = "2012-01", date_end: str = "2016-12") -> list[Path]:
    """
    Programmatically discover every dataset in the data.gov.sg "Resale Flat
    Prices" collection and download the CSVs that overlap [date_start, date_end].

    This uses data.gov.sg's official public API (Collections -> Datasets ->
    poll-download) rather than scraping the HTML page, so it is robust to
    UI changes and requires no manual interface interaction.

    Returns the list of file paths written. If network access is
    unavailable, logs a warning and returns an empty list (caller falls
    back to pre-existing files in raw_dir).
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    try:
        with urlopen(Request(COLLECTION_API, headers={"User-Agent": "hdb-etl/1.0"}), timeout=15) as resp:
            meta = json.loads(resp.read())
        dataset_ids = meta["data"]["collectionMetadata"]["childDatasets"]

        for ds_id in dataset_ids:
            poll_url = POLL_DOWNLOAD_API.format(dataset_id=ds_id)
            with urlopen(Request(poll_url, headers={"User-Agent": "hdb-etl/1.0"}), timeout=15) as resp:
                poll_meta = json.loads(resp.read())
            download_url = poll_meta["data"]["url"]

            out_path = raw_dir / f"{ds_id}.csv"
            with urlopen(download_url, timeout=60) as resp, open(out_path, "wb") as f:
                f.write(resp.read())

            # Cheap relevance filter: only keep files whose `month` column
            # range overlaps the requested window. Read just enough to check.
            try:
                head = pd.read_csv(out_path, usecols=["month"])
                if head["month"].max() < date_start or head["month"].min() > date_end:
                    out_path.unlink()
                    continue
            except Exception:
                pass  # if schema differs (no `month` col), keep it -- union step will handle it

            written.append(out_path)
            logger.info("Downloaded %s", out_path.name)

    except Exception as exc:  # network disabled, API shape changed, etc.
        logger.warning("download_datasets() skipped (%s). Using local files in %s instead.", exc, raw_dir)

    return written


def load_raw_files(raw_dir: Path) -> pd.DataFrame:
    """
    Load every *.csv under raw_dir as-is (no manual modification), and
    UNION their schemas into one master DataFrame. Columns present in some
    files but not others are filled with NaN for the rows that lack them,
    per the "contain all the attributes found in all files" requirement.

    Adds lineage columns:
      - _source_file : originating CSV filename (audit trail)
      - _row_id      : stable synthetic row identifier for traceability
                        through the pipeline (cleaned/failed/hashed outputs)
    """
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}. Run download_datasets() or place files there.")

    frames = []
    for fp in csv_files:
        df = pd.read_csv(fp, dtype=str)  # read as str first: no premature type coercion / silent data loss
        df["_source_file"] = fp.name
        frames.append(df)
        logger.info("Loaded %s (%d rows, %d cols)", fp.name, len(df), df.shape[1])

    # outer-join union of columns (concat with sort=False preserves first-seen col order,
    # pandas automatically unions columns and fills missing with NaN)
    master = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    master["_row_id"] = range(1, len(master) + 1)

    logger.info("Master raw dataset: %d rows, %d columns from %d file(s)", len(master), master.shape[1], len(csv_files))
    return master
