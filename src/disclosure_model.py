"""Step 3 (H1): does environmental disclosure rise with company size?

Model: logit(scored) ~ log(size). Sector fixed effects are not identifiable with 7
unscored firms (most sectors are 100% scored, so the dummies separate perfectly);
sector is shown descriptively instead. Robustness: Fisher exact test on a median
size split, and a refit treating the 5 "partial"-verification rows as scored
(the most conservative reading of unverified non-disclosure).

Writes outputs/h1.json, consumed by 04_simulation.py and 05_build_report.py.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fit_logit(y: pd.Series, x: pd.Series):
    X = sm.add_constant(x.rename("log_size"))
    return sm.Logit(y, X).fit(disp=0)


def main() -> None:
    s = pd.read_csv(ROOT / "data" / "disclosure_sample.csv")
    uni = pd.read_csv(ROOT / "data" / "universe.csv")
    s["size_decile"] = s["size_decile"].astype(int)
    s["log_size"] = np.log(s["weight"])

    # Centre log size on the scoreable universe so the intercept is interpretable.
    nf = uni[~uni["is_financial"]]
    centre = float(np.log(nf["weight"]).mean())
    x = s["log_size"] - centre

    m = fit_logit(s["scored"], x)
    b, se = float(m.params["log_size"]), float(m.bse["log_size"])
    ci = [float(v) for v in m.conf_int().loc["log_size"]]

    # Robustness: partial-verification rows counted as scored.
    y_cons = s["scored"].where(s["verification"] != "partial", 1)
    m_cons = fit_logit(y_cons, x) if y_cons.sum() < len(y_cons) else None

    # Pre-specified comparison: Fisher exact test on a median split (deciles 6-10 vs 1-5).
    # The "top 4 deciles fully scored" pattern is reported descriptively only, because
    # that cut was noticed after looking at the data.
    big = s["size_decile"] >= 6
    table = [[int((s.loc[big, "scored"] == 0).sum()), int((s.loc[big, "scored"] == 1).sum())],
             [int((s.loc[~big, "scored"] == 0).sum()), int((s.loc[~big, "scored"] == 1).sum())]]
    fisher_p = float(stats.fisher_exact(table, alternative="less")[1])
    top = s["size_decile"] >= 7

    by_dec = []
    for d, g in s.groupby("size_decile"):
        k, n = int(g["scored"].sum()), len(g)
        lo, hi = wilson(k, n)
        xd = float(np.log(nf.loc[nf["size_decile"] == d, "weight"]).median() - centre)
        p_fit = float(m.predict(np.array([[1.0, xd]]))[0])
        by_dec.append({"decile": int(d), "n": n, "scored": k, "rate": k / n,
                       "ci_lo": lo, "ci_hi": hi, "p_fitted": p_fit})

    unscored = s[s["scored"] == 0][["name", "sector", "size_decile", "n_metrics", "verification", "country"]]
    sector_tab = (s.groupby("sector")["scored"].agg(["size", "mean"])
                  .rename(columns={"size": "n", "mean": "rate"}).reset_index())

    res = {
        "n_sample": int(len(s)),
        "n_scored": int(s["scored"].sum()),
        "scored_rate": float(s["scored"].mean()),
        "log_size_centre": centre,
        "logit_intercept": float(m.params["const"]),
        "logit_slope": b,
        "logit_slope_se": se,
        "logit_slope_ci95": ci,
        "logit_slope_p": float(m.pvalues["log_size"]),
        "logit_slope_conservative": float(m_cons.params["log_size"]) if m_cons is not None else None,
        "logit_slope_conservative_p": float(m_cons.pvalues["log_size"]) if m_cons is not None else None,
        "n_unscored_conservative": int((y_cons == 0).sum()),
        "fisher_table_median_split": table,
        "fisher_p_median_split_one_sided": fisher_p,
        "rate_top4_deciles": float(s.loc[top, "scored"].mean()),
        "rate_bottom6_deciles": float(s.loc[~top, "scored"].mean()),
        "rate_top_decile": float(s.loc[s["size_decile"] == 10, "scored"].mean()),
        "rate_bottom_decile": float(s.loc[s["size_decile"] == 1, "scored"].mean()),
        "by_decile": by_dec,
        "unscored": unscored.to_dict(orient="records"),
        "by_sector": sector_tab.to_dict(orient="records"),
        "n_partial": int((s["verification"] == "partial").sum()),
        "n_primary": int((s["verification"] == "primary").sum()),
    }
    (OUT / "h1.json").write_text(json.dumps(res, indent=2))
    print(m.summary2().tables[1].round(3).to_string())
    print(f"\nslope {b:.3f} (95% CI {ci[0]:.3f} to {ci[1]:.3f}), p={res['logit_slope_p']:.3f}")
    print(f"conservative slope: {res['logit_slope_conservative']}, p={res['logit_slope_conservative_p']}")
    print(f"Fisher one-sided p={fisher_p:.3f}; table {table}")
    print(pd.DataFrame(by_dec).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
