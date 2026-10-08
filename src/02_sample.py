"""Step 2: draw the stratified disclosure sample and write a blank collection sheet.

10 companies per size decile, financials excluded (Osmosis does not score them),
fixed seed so the draw reproduces. Output: data/disclosure_sample.csv with empty
columns to fill by hand from each company's latest annual / sustainability report.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261009
PER_DECILE = 10

COLLECT_COLS = [
    "scope12_ghg",      # Y/N  Scope 1+2 GHG emissions disclosed
    "water",            # Y/N  water withdrawal disclosed
    "waste",            # Y/N  total waste disclosed
    "report_year",      # fiscal year of the report checked
    "employees",        # headcount, for the Omnibus I flag
    "revenue_eur_m",    # revenue in EUR millions, for the Omnibus I flag
    "source_url",       # where the flags were checked
    "notes",
]


def main() -> None:
    uni = pd.read_csv(ROOT / "data" / "universe.csv")
    pool = uni[~uni["is_financial"]]
    parts = [
        g.sample(n=min(PER_DECILE, len(g)), random_state=SEED + int(d))
        for d, g in pool.groupby("size_decile")
    ]
    sample = (
        pd.concat(parts)
        .sort_values(["size_decile", "weight"], ascending=[False, False])
        .reset_index(drop=True)
    )
    keep = ["ticker", "name", "sector", "country", "eu_domicile", "weight", "size_decile"]
    sample = sample[keep]
    for c in COLLECT_COLS:
        sample[c] = ""
    dest = ROOT / "data" / "disclosure_sample.csv"
    if dest.exists():
        raise SystemExit(f"{dest} already exists; refusing to overwrite collected data")
    sample.to_csv(dest, index=False)
    print(f"{len(sample)} names -> {dest}")
    print(sample.groupby("size_decile").size().to_string())


if __name__ == "__main__":
    main()
