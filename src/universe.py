"""Step 1: build data/universe.csv from the iShares STOXX Europe 600 holdings export.

Usage: python src/01_universe.py data/raw/<holdings>.csv

The iShares export has a few metadata lines before the real header row, so we
locate the header by searching for the line that starts with "Ticker".
Weight is proportional to free-float market cap, so it doubles as the size measure.
"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# iShares "Location" (country of risk) used as a proxy for domicile; treated as in
# CSRD scope for the Omnibus I scenario. Proxy limitation is stated in the report.
EU = {
    "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic", "Czechia",
    "Denmark", "Estonia", "Finland", "France", "Germany", "Greece", "Hungary", "Ireland",
    "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Netherlands", "Poland",
    "Portugal", "Romania", "Slovakia", "Slovenia", "Spain", "Sweden",
    # EEA states that have transposed CSRD
    "Norway", "Iceland", "Liechtenstein",
}


def read_ishares(path: Path) -> pd.DataFrame:
    raw = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    start = next(i for i, line in enumerate(raw) if line.strip('"').startswith("Ticker"))
    # Footer lines (disclaimers) have few commas; keep only lines that look like data.
    body = [raw[start]] + [l for l in raw[start + 1:] if l.count(",") >= 5]
    df = pd.read_csv(io.StringIO("\n".join(body)), dtype=str)
    # Some regional exports use an apostrophe (’ or ') as the thousands separator.
    for c in ("Market Value", "Weight (%)", "Notional Value", "Shares", "Price"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].str.replace("[’',]", "", regex=True), errors="coerce")
    return df


def main(src: str) -> None:
    df = read_ishares(Path(src))
    df.columns = [c.strip() for c in df.columns]
    wcol = next(c for c in df.columns if c.lower().startswith("weight"))
    loc = next((c for c in df.columns if c.lower() in ("location", "location of risk")), None)

    df = df[df["Asset Class"].str.strip().eq("Equity")].copy()
    # The published weight is rounded to 2dp, which ties most small names; market value
    # is exact and proportional to the fund weight, so weight is rebuilt from it (in %).
    mv = df["Market Value"] if "Market Value" in df.columns else df[wcol]
    out = pd.DataFrame({
        "ticker": df["Ticker"].astype(str).str.strip(),
        "name": df["Name"].str.strip(),
        "sector": df["Sector"].str.strip(),
        "country": df[loc].str.strip() if loc else np.nan,
        "market_value": mv,
    }).dropna(subset=["market_value"])
    out = out[out["market_value"] > 0]
    out["weight"] = 100 * out["market_value"] / out["market_value"].sum()
    out["is_financial"] = out["sector"].eq("Financials")
    out["eu_domicile"] = out["country"].isin(EU)
    out["log_weight"] = np.log(out["weight"])

    # Size deciles on the scoreable (non-financial) universe: 1 = smallest, 10 = largest.
    nf = ~out["is_financial"]
    out.loc[nf, "size_decile"] = pd.qcut(out.loc[nf, "weight"].rank(method="first"), 10, labels=False) + 1
    out = out.sort_values("weight", ascending=False).reset_index(drop=True)

    dest = ROOT / "data" / "universe.csv"
    out.to_csv(dest, index=False)
    print(f"{len(out)} equities, {nf.sum()} non-financial, {out['eu_domicile'].sum()} EU-domiciled -> {dest}")
    print(out.groupby("sector").size().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
