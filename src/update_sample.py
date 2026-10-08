"""Helper for disclosure collection: merge hand-checked rows into data/disclosure_sample.csv.

Usage: python src/update_sample.py data/raw/batchNN.json
The batch file is a list of objects keyed by "name" (exactly as in the sample) with any of:
scope12_ghg, water, waste (Y / N / U), report_year, employees, revenue_eur_m,
omnibus_in_scope (Y / N / n/a), source_url, notes, verification (tracenable / primary / partial).

After merging, the derived columns are recomputed for every row:
  n_metrics = number of Y among scope12_ghg, water, waste
  scored    = 1 if n_metrics >= 2 else 0   (the two-of-three rule)
  verification defaults to "tracenable" for a collected row that has none.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "data" / "disclosure_sample.csv"
FLAGS = ["scope12_ghg", "water", "waste"]


def recompute(df: pd.DataFrame) -> pd.DataFrame:
    df["n_metrics"] = sum((df[c] == "Y").astype(int) for c in FLAGS).astype(str)
    df["scored"] = (df["n_metrics"].astype(int) >= 2).astype(int).astype(str)
    if "verification" not in df.columns:
        df["verification"] = ""
    collected = df["source_url"] != ""
    df.loc[collected & (df["verification"] == ""), "verification"] = "tracenable"
    return df


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
    df = recompute(df)
    df.to_csv(PATH, index=False)
    done = (df["source_url"] != "").sum()
    print(f"updated {len(rows)} rows; {done}/{len(df)} collected; {df['scored'].astype(int).sum()} scored")


if __name__ == "__main__":
    main(sys.argv[1])
