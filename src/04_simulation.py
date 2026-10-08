"""Step 4 (H2, H3, H4 + sensitivity): what does the missing-data rule do to a portfolio?

Stylised set-up on the full non-financial STOXX Europe 600 universe:
  * true Resource Efficiency score z ~ N(0,1), independent of size BY CONSTRUCTION
  * each firm is scored with probability p(size) from the fitted H1 logit
  * sector-neutral tilt: w = b * max(0, 1 + lambda * score), rescaled so each sector
    keeps its benchmark weight (long-only, lambda = 0.5)

Treatments for unscored firms:
  T0  oracle: every firm scored (reference, not achievable)
  T1  unscored = sector bottom score (penalise non-disclosure)
  T2  unscored = neutral, held at benchmark weight
  T3  T2, then scores size-neutralised within sector (cap-weighted)
  T4  T1, then scores size-neutralised within sector (keeps the penalty, removes the size bet)

Recorded per run: active size exposure (active weights x z-scored log size), RE capture
(active weights x true score), active share, and a stylised tracking error.
Writes outputs/sim.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
SEED = 20261009
N_SIM = 1000
LAM = 0.5
SIGMA_IDIO = 0.25   # stylised annual idiosyncratic vol per stock
SIGMA_SIZE = 0.04   # stylised annual vol of a 1-sd log-size exposure
TREATMENTS = ["T0", "T1", "T2", "T3", "T4"]


# ----------------------------------------------------------------------------- setup
def load_universe():
    uni = pd.read_csv(ROOT / "data" / "universe.csv")
    u = uni[~uni["is_financial"]].reset_index(drop=True)
    b = (u["weight"] / u["weight"].sum()).to_numpy()
    logw = np.log(u["weight"].to_numpy())
    x = (logw - logw.mean()) / logw.std()
    codes, sectors = pd.factorize(u["sector"])
    G = np.eye(len(sectors))[codes]            # n x K one-hot
    return u, b, logw, x, codes, G


def p_scored(logw, intercept, slope, centre):
    return 1.0 / (1.0 + np.exp(-(intercept + slope * (logw - centre))))


def solve_intercept(logw, slope, centre, target_mean):
    """Intercept that makes the universe-average P(scored) equal target_mean."""
    lo, hi = -20.0, 20.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if p_scored(logw, mid, slope, centre).mean() < target_mean:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ------------------------------------------------------------------------- mechanics
def tilt(score, b, G, fixed=None):
    """Sector-neutral long-only linear tilt: w = b * (1 + lambda * (s - s_bar_sector)).

    s_bar_sector is the cap-weighted sector mean over tilted names, so under full
    information each name's expected active weight is exactly zero (no mechanical
    size bias). Negative weights are clipped to zero and the sector rescaled.
    `fixed` (S x n bool) names stay at benchmark weight.
    """
    S = score.shape[0]
    wts = np.broadcast_to(b, score.shape) if fixed is None else np.where(fixed, 0.0, b)
    Bg = wts @ G
    sbar = np.divide((wts * score) @ G, Bg, out=np.zeros_like(Bg), where=Bg > 0)
    raw = wts * np.clip(1.0 + LAM * (score - sbar @ G.T), 0.0, None)
    budget = Bg if fixed is not None else np.broadcast_to(b @ G, (S, G.shape[1]))
    sec_raw = raw @ G
    scale = np.divide(budget, sec_raw, out=np.zeros_like(sec_raw), where=sec_raw > 0)
    w = raw * (scale @ G.T)
    if fixed is not None:
        w = np.where(fixed, b, w)
    return w - b


def neutralise(score, x, b, G, mask=None):
    """Remove the cap-weighted within-sector linear relation between score and size."""
    wts = b if mask is None else np.where(mask, b, 0.0)
    Bg = wts @ G
    Bg = np.where(Bg > 0, Bg, 1.0)
    xbar = ((wts * x) @ G) / Bg
    xc = x - xbar @ G.T
    sbar = ((wts * score) @ G) / Bg
    sc = score - sbar @ G.T
    cov = (wts * sc * xc) @ G
    var = (wts * xc**2) @ G
    beta = np.divide(cov, var, out=np.zeros_like(cov), where=var > 0)
    return score - (beta @ G.T) * xc


def sector_bottom_fill(z, scored, codes, K):
    s = np.where(scored, z, np.nan)
    out = s.copy()
    for k in range(K):
        idx = codes == k
        sub = s[:, idx]
        mins = np.nanmin(np.where(np.isnan(sub), np.inf, sub), axis=1)
        mins = np.where(np.isinf(mins), 0.0, mins)          # sector with nobody scored
        out[:, idx] = np.where(np.isnan(sub), mins[:, None], sub)
    return out


def run_treatments(z, scored, b, x, codes, G):
    K = G.shape[1]
    unscored = ~scored
    s1 = sector_bottom_fill(z, scored, codes, K)
    s2 = np.where(scored, z, 0.0)
    act = {
        "T0": tilt(z, b, G),                                   # oracle: everyone scored
        "T1": tilt(s1, b, G),
        "T2": tilt(s2, b, G, fixed=unscored),
        "T3": tilt(neutralise(s2, x, b, G, mask=scored), b, G, fixed=unscored),
        "T4": tilt(neutralise(s1, x, b, G), b, G),
    }
    return act


def metrics(a, z, x, sd_logw, small, b):
    size = a @ x
    re = (a * z).sum(axis=1)
    ash = 0.5 * np.abs(a).sum(axis=1)
    te_size = np.abs(size) * SIGMA_SIZE
    te = np.sqrt(te_size**2 + (a**2).sum(axis=1) * SIGMA_IDIO**2)
    return {
        "size": size,
        "size_pct": np.exp(size * sd_logw) - 1.0,   # % larger weighted geometric-average size
        # active weight of the smaller half of names, as % of their benchmark weight
        "small_half_rel": (a[:, small].sum(axis=1)) / b[small].sum(),
        "re": re,
        "active_share": ash,
        "te": te,
        "te_size_share": te_size**2 / te**2,
    }


def summarise(v):
    v = np.asarray(v)
    return {"mean": float(v.mean()), "p05": float(np.percentile(v, 5)),
            "p95": float(np.percentile(v, 95)), "sd": float(v.std())}


def simulate(p, b, x, codes, G, sd_logw, rng, n_sim=N_SIM, extra_unscore=None):
    n = len(b)
    z = rng.standard_normal((n_sim, n))
    scored = rng.random((n_sim, n)) < p
    if extra_unscore is not None:
        scored &= ~extra_unscore(rng, n_sim)
    acts = run_treatments(z, scored, b, x, codes, G)
    small = x < np.median(x)
    out = {t: {k: summarise(v) for k, v in metrics(acts[t], z, x, sd_logw, small, b).items()} for t in acts}
    re0 = (acts["T0"] * z).sum(axis=1)
    size0 = acts["T0"] @ x
    small0 = acts["T0"][:, small].sum(axis=1) / b[small].sum()
    for t in acts:
        out[t]["re_vs_oracle"] = summarise((acts[t] * z).sum(axis=1) / re0)
        # paired: what the missing-data treatment adds on top of the oracle portfolio
        out[t]["size_vs_oracle"] = summarise(acts[t] @ x - size0)
        out[t]["small_half_vs_oracle"] = summarise(acts[t][:, small].sum(axis=1) / b[small].sum() - small0)
    out["unscored_count"] = summarise((~scored).sum(axis=1))
    out["unscored_weight"] = summarise(((~scored) * b).sum(axis=1))
    return out


# ------------------------------------------------------------------------------ main
def main() -> None:
    global LAM
    h1 = json.loads((OUT / "h1.json").read_text())
    u, b, logw, x, codes, G = load_universe()
    sd_logw = float(logw.std())
    centre = h1["log_size_centre"]
    a0, b0 = h1["logit_intercept"], h1["logit_slope"]
    p_base = p_scored(logw, a0, b0, centre)
    base_unscored = float(1 - p_base.mean())
    res = {"n_universe": int(len(u)), "n_sim": N_SIM, "lambda": LAM, "seed": SEED,
           "sigma_idio": SIGMA_IDIO, "sigma_size": SIGMA_SIZE,
           "sd_log_size": sd_logw, "base_unscored_rate_universe": base_unscored}

    # --- H2 / H4: base case ------------------------------------------------------
    rng = np.random.default_rng(SEED)
    res["base"] = simulate(p_base, b, x, codes, G, sd_logw, rng)
    # Placebo: flat disclosure (slope 0, same average rate) -> T1 tilt should vanish.
    p_flat = np.full_like(p_base, p_base.mean())
    res["flat"] = simulate(p_flat, b, x, codes, G, sd_logw, np.random.default_rng(SEED + 1))

    # --- H3: Omnibus I ------------------------------------------------------------
    s = pd.read_csv(ROOT / "data" / "disclosure_sample.csv")
    eu_s = s[s["eu_domicile"]]
    q_dec = (eu_s.assign(out=eu_s["omnibus_in_scope"].eq("N"))
             .groupby("size_decile")["out"].agg(["sum", "size"]))
    q_dec["share"] = q_dec["sum"] / q_dec["size"]
    q_map = q_dec["share"].to_dict()
    eu = u["eu_domicile"].to_numpy()
    q = np.where(eu, u["size_decile"].map(q_map).fillna(0.0).to_numpy(), 0.0)
    res["omnibus_at_risk_by_decile"] = [
        {"decile": int(d), "eu_in_sample": int(r["size"]), "at_risk_in_sample": int(r["sum"]),
         "share_of_eu": float(r["share"])} for d, r in q_dec.iterrows()]
    res["omnibus_expected_at_risk_names"] = float(q.sum())
    res["omnibus_expected_at_risk_share_names"] = float(q.mean())
    res["omnibus_expected_at_risk_weight"] = float((q * b).sum())
    res["omnibus_sample_at_risk"] = int(eu_s["omnibus_in_scope"].eq("N").sum())
    res["omnibus_sample_eu"] = int(len(eu_s))
    res["omnibus"] = {}
    for k, prob in enumerate([0.25, 0.5, 1.0]):
        def extra(rng_, n_sim, prob=prob):
            at_risk = rng_.random((n_sim, len(q))) < q
            return at_risk & (rng_.random((n_sim, len(q))) < prob)
        res["omnibus"][str(prob)] = simulate(p_base, b, x, codes, G, sd_logw,
                                             np.random.default_rng(SEED + 10 + k), extra_unscore=extra)

    # --- Sensitivity sweep: slope multiplier x unscored base rate ------------------
    sweep = []
    mults = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    rates = [round(base_unscored, 4), 0.15, 0.30]
    for j, rate in enumerate(rates):
        for i, m in enumerate(mults):
            slope = b0 * m
            a = solve_intercept(logw, slope, centre, 1 - rate)
            p = p_scored(logw, a, slope, centre)
            r = simulate(p, b, x, codes, G, sd_logw, np.random.default_rng(SEED + 100 + 10 * j + i), n_sim=500)
            sweep.append({"unscored_rate": rate, "slope_mult": m, "slope": slope,
                          **{f"{t}_size_{k}": r[t]["size"][k] for t in TREATMENTS for k in ("mean", "p05", "p95")},
                          **{f"{t}_size_pct_mean": r[t]["size_pct"]["mean"] for t in TREATMENTS},
                          **{f"{t}_small_half_rel_{k}": r[t]["small_half_rel"][k] for t in TREATMENTS for k in ("mean", "p05", "p95")},
                          **{f"{t}_re_vs_oracle_mean": r[t]["re_vs_oracle"]["mean"] for t in TREATMENTS},
                          **{f"{t}_size_vs_oracle_{k}": r[t]["size_vs_oracle"][k] for t in TREATMENTS for k in ("mean", "p05", "p95")},
                          **{f"{t}_small_half_vs_oracle_{k}": r[t]["small_half_vs_oracle"][k] for t in TREATMENTS for k in ("mean", "p05", "p95")},
                          **{f"{t}_re_mean": r[t]["re"]["mean"] for t in TREATMENTS}})
    res["sweep"] = sweep

    # Lambda robustness (tilt strength) at base calibration.
    res["lambda_check"] = {}
    for lam in (0.25, 1.0):
        LAM = lam
        r = simulate(p_base, b, x, codes, G, sd_logw, np.random.default_rng(SEED + 500), n_sim=500)
        res["lambda_check"][str(lam)] = {t: {"size_mean": r[t]["size"]["mean"], "re_mean": r[t]["re"]["mean"]}
                                         for t in TREATMENTS}
    LAM = 0.5

    (OUT / "sim.json").write_text(json.dumps(res, indent=2))

    # console summary
    print(f"universe {len(u)} names; base unscored rate {base_unscored:.3f}")
    for name in ("base", "flat"):
        print(f"\n[{name}] unscored names {res[name]['unscored_count']['mean']:.1f}, "
              f"weight {res[name]['unscored_weight']['mean']:.3%}")
        for t in TREATMENTS:
            r = res[name][t]
            print(f"  {t}: size {r['size']['mean']:+.4f} [{r['size']['p05']:+.4f},{r['size']['p95']:+.4f}]"
                  f"  size% {r['size_pct']['mean']:+.2%}  smallhalf {r['small_half_rel']['mean']:+.2%}"
                  f"  RE/oracle {r['re_vs_oracle']['mean']:.3f}"
                  f"  AS {r['active_share']['mean']:.3f}  TE {r['te']['mean']:.3%}"
                  f"  TE-size-share {r['te_size_share']['mean']:.1%}")
    print(f"\nOmnibus expected at-risk names {res['omnibus_expected_at_risk_names']:.1f} "
          f"({res['omnibus_expected_at_risk_share_names']:.1%} of names, "
          f"{res['omnibus_expected_at_risk_weight']:.2%} of weight)")
    for prob, r in res["omnibus"].items():
        print(f"  pi={prob}: unscored {r['unscored_count']['mean']:.1f}; "
              + "  ".join(f"{t} size {r[t]['size']['mean']:+.4f}" for t in TREATMENTS))
    sw = pd.DataFrame(sweep)
    print("\n" + sw.pivot(index="slope_mult", columns="unscored_rate", values="T1_size_mean").round(4).to_string())
    print(sw.pivot(index="slope_mult", columns="unscored_rate", values="T4_size_mean").round(4).to_string())
    print("lambda check", json.dumps(res["lambda_check"], indent=0)[:400])


if __name__ == "__main__":
    main()
