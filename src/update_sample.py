"""Helper for disclosure collection: merge hand-checked rows into data/disclosure_sample.csv.

Usage: python src/update_sample.py rows.json
rows.json is a list of objects keyed by "name" (exactly as in the sample) with any of:
scope12_ghg, water, waste (Y/N), report_year, employees, revenue_eur_m,
omnibus_in_scope (Y/N), source_url, notes.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "data" / "disclosure_sample.csv"


def main(src: str) -> None:
    df = pd.read_csv(PATH, dtype=str, keep_default_na=False)
    if "omnibus_in_scope" not in df.columns:
        df.insert(df.columns.get_loc("source_url"), "omnibus_in_scope", "")
    rows = json.loads(Path(src).read_text())
    for r in rows:
        hit = df["name"] == r["name"]
        if hit.sum() != 1:
            raise SystemExit(f"name not found uniquely: {r['name']}")
        for k, v in r.items():
            if k != "name":
                df.loc[hit, k] = str(v)
    df.to_csv(PATH, index=False)
    done = (df["source_url"] != "").sum()
    print(f"updated {len(rows)} rows; {done}/{len(df)} collected")


if __name__ == "__main__":
    main(sys.argv[1])
