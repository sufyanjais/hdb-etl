# HDB Resale Flat Prices — ETL Pipeline (Part 1)

## How to run
```bash
pip install pandas numpy python-dateutil
jupyter notebook HDB_Resale_ETL_Pipeline.ipynb
# Run All. Cell 1 will attempt to auto-download the full Jan 2012–Dec 2016
# dataset from data.gov.sg; if you're offline it falls back to whatever
# CSVs already exist in data/raw/ (the provided sample is included there).
```

## Structure
```
HDB_Resale_ETL_Pipeline.ipynb   <- main deliverable: documented, executed notebook
src/
  extract.py       Dataset Extraction (programmatic download + schema union)
  profile.py       Data Profiling
  validate.py      Validation rules (Date, Town, Flat Type, Flat Model, storey_range)
  transform.py     Remaining lease, composite-key dedup, anomaly heuristic, Resale Identifier
  hash_utils.py     Irreversible hashing of the Resale Identifier
  pipeline.py       CLI orchestrator (python3 src/pipeline.py) — same pipeline, script form
data/raw/           Input CSV(s) — drop additional data.gov.sg vintage files here to generalise
                    to the full Jan 2012–Dec 2016 window
output/
  raw/              Combined raw dataset, unmodified
  cleaned/           Passed validation + dedup + remaining lease + anomaly flags
  transformed/        + Resale Identifier, identifier-level dedup
  hashed/            + irreversible hash of Resale Identifier
  failed/            All rejected/discarded rows, with _fail_reason
```

## Results on the supplied sample (Mar 2012–Dec 2014, 52,203 rows)
| Stage | Rows |
|---|---|
| Raw | 52,203 |
| Failed validation | 32 |
| Discarded (composite-key duplicate) | 1,085 |
| Cleaned | 51,086 (161 flagged as price anomalies, kept) |
| Discarded (identifier duplicate) | 6,684 |
| Transformed / Hashed | 44,402 |
| **Total in Failed dataset** | **7,801** |

See the notebook's Appendix for full documentation of assumptions and design decisions.
