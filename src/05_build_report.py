"""Step 5: export outputs/results.json, the four charts, and site/index.html.

Every number in the report text is read from results.json (via the `r` dict in the
template), so nothing is typed by hand.
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs_version
from jinja2 import Environment, FileSystemLoader, StrictUndefined

ROOT = Path(__file__).resolve().parents[1]
OUT, SITE = ROOT / "outputs", ROOT / "site"
SITE.mkdir(exist_ok=True)

# Validated palette (dataviz reference, light mode): T1 blue, T2 orange, ordinal blue ramp.
C = {
    "T1": "#2a78d6", "T2": "#eb6834",
    "ramp": ["#86b6ef", "#2a78d6", "#104281"],
    "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#ffffff",
}
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
PLOT_CONFIG = {"displayModeBar": False, "responsive": True}


def pct(v, d=1, sign=False):
    s = f"{abs(v) * 100:.{d}f}%" if not sign else f"{v * 100:+.{d}f}%"
    return s


# ------------------------------------------------------------------- results.json
def build_results():
    h1 = json.loads((OUT / "h1.json").read_text())
    sim = json.loads((OUT / "sim.json").read_text())
    base, flat = sim["base"], sim["flat"]
    uni = pd.read_csv(ROOT / "data" / "universe.csv")
    u = uni[~uni["is_financial"]]
    x = np.log(u["weight"])
    small = x < np.median(x)
    small_w = float(u.loc[small, "weight"].sum() / u["weight"].sum())
    sd = sim["sd_log_size"]
    sw = pd.DataFrame(sim["sweep"])
    rates = sorted(sw["unscored_rate"].unique())

    def sweep_at(rate, mult, col):
        return float(sw[(sw.unscored_rate == rate) & (sw.slope_mult == mult)][col].iloc[0])

    dec = {d["decile"]: d for d in h1["by_decile"]}
    om = sim["omnibus"]
    r = {
        # H1
        "sample_n": h1["n_sample"], "sample_scored": h1["n_scored"],
        "sample_unscored": h1["n_sample"] - h1["n_scored"],
        "sample_scored_pct": h1["scored_rate"],
        "rate_top4": h1["rate_top4_deciles"], "rate_bottom6": h1["rate_bottom6_deciles"],
        "logit_slope": h1["logit_slope"], "logit_ci_lo": h1["logit_slope_ci95"][0],
        "logit_ci_hi": h1["logit_slope_ci95"][1], "logit_p": h1["logit_slope_p"],
        "fitted_d1": dec[1]["p_fitted"], "fitted_d10": dec[10]["p_fitted"],
        "fisher_median_p": h1["fisher_p_median_split_one_sided"],
        "n_partial": h1["n_partial"], "n_primary": h1["n_primary"],
        # simulation set-up
        "uni_n": sim["n_universe"], "n_sim": sim["n_sim"], "lambda": sim["lambda"],
        "uni_unscored_rate": sim["base_unscored_rate_universe"],
        "uni_unscored_names": base["unscored_count"]["mean"],
        "uni_unscored_weight": base["unscored_weight"]["mean"],
        "small_half_bench_weight": small_w,
        # H2 / H4
        "T1_small": base["T1"]["small_half_vs_oracle"]["mean"],
        "T1_small_p05": base["T1"]["small_half_vs_oracle"]["p05"],
        "T1_small_p95": base["T1"]["small_half_vs_oracle"]["p95"],
        "T1_small_pp": base["T1"]["small_half_vs_oracle"]["mean"] * small_w,
        "T1_geo_size": math.exp(base["T1"]["size_vs_oracle"]["mean"] * sd) - 1,
        "T2_small": base["T2"]["small_half_vs_oracle"]["mean"],
        "T2_small_p05": base["T2"]["small_half_vs_oracle"]["p05"],
        "T2_small_p95": base["T2"]["small_half_vs_oracle"]["p95"],
        "T1_re": base["T1"]["re_vs_oracle"]["mean"], "T2_re": base["T2"]["re_vs_oracle"]["mean"],
        "T3_re": base["T3"]["re_vs_oracle"]["mean"], "T4_re": base["T4"]["re_vs_oracle"]["mean"],
        "T3_size_abs": base["T3"]["size"]["mean"], "T4_size_abs": base["T4"]["size"]["mean"],
        "T1_te": base["T1"]["te"]["mean"], "T2_te": base["T2"]["te"]["mean"],
        "flat_T1_small": flat["T1"]["small_half_vs_oracle"]["mean"],
        # H3
        "om_sample_at_risk": sim["omnibus_sample_at_risk"], "om_sample_eu": sim["omnibus_sample_eu"],
        "om_names": sim["omnibus_expected_at_risk_names"],
        "om_share_names": sim["omnibus_expected_at_risk_share_names"],
        "om_weight": sim["omnibus_expected_at_risk_weight"],
        "om_T1_small_25": om["0.25"]["T1"]["small_half_vs_oracle"]["mean"],
        "om_T1_small_50": om["0.5"]["T1"]["small_half_vs_oracle"]["mean"],
        "om_T1_small_100": om["1.0"]["T1"]["small_half_vs_oracle"]["mean"],
        "om_T1_small_100_p05": om["1.0"]["T1"]["small_half_vs_oracle"]["p05"],
        "om_T1_small_100_p95": om["1.0"]["T1"]["small_half_vs_oracle"]["p95"],
        "om_T2_small_100": om["1.0"]["T2"]["small_half_vs_oracle"]["mean"],
        # sensitivity sweep
        "sweep_rates": rates,
        "sweep_flat_T1": sweep_at(rates[0], 0.0, "T1_small_half_vs_oracle_mean"),
        "sweep_15_T1": sweep_at(0.15, 1.0, "T1_small_half_vs_oracle_mean"),
        "sweep_30_T1": sweep_at(0.30, 1.0, "T1_small_half_vs_oracle_mean"),
        "sweep_30_T1_2x": sweep_at(0.30, 2.0, "T1_small_half_vs_oracle_mean"),
        "sweep_T2_maxabs": float(sw["T2_small_half_vs_oracle_mean"].abs().max()),
    }
    # display strings (the template uses these, so the text never carries hand-typed numbers)
    f = {
        "sample_scored_pct": pct(r["sample_scored_pct"], 0),
        "rate_top4": pct(r["rate_top4"], 0),
        "logit_slope": f"{r['logit_slope']:.2f}", "logit_ci_lo": f"{r['logit_ci_lo']:.2f}",
        "logit_ci_hi": f"{r['logit_ci_hi']:.2f}", "logit_p": f"{r['logit_p']:.2f}",
        "fitted_d1": pct(r["fitted_d1"], 0), "fitted_d10": pct(r["fitted_d10"], 0),
        "uni_unscored_rate": pct(r["uni_unscored_rate"], 0),
        "uni_unscored_names": f"{r['uni_unscored_names']:.0f}",
        "uni_unscored_weight": pct(r["uni_unscored_weight"], 1),
        "small_half_bench_weight": pct(r["small_half_bench_weight"], 0),
        "T1_small": pct(r["T1_small"]), "T1_small_p05": pct(r["T1_small_p95"]),
        "T1_small_p95": pct(r["T1_small_p05"]), "T1_small_pp": f"{abs(r['T1_small_pp']) * 100:.1f}",
        "T1_geo_size": pct(r["T1_geo_size"]),
        "T2_small": pct(r["T2_small"]), "T2_band": pct(max(abs(r["T2_small_p05"]), abs(r["T2_small_p95"]))),
        "T1_re": pct(r["T1_re"], 0), "T2_re": pct(r["T2_re"], 0),
        "T1_re1": pct(r["T1_re"], 1), "T2_re1": pct(r["T2_re"], 1),
        "T34_re": pct((r["T3_re"] + r["T4_re"]) / 2, 0),
        "T34_cost": pct(1 - (r["T3_re"] + r["T4_re"]) / 2, 0),
        "T1_te": pct(r["T1_te"], 2), "T2_te": pct(r["T2_te"], 2),
        "flat_T1_small": pct(r["flat_T1_small"], 1, sign=True),
        "om_names": f"{r['om_names']:.0f}", "om_share_names": pct(r["om_share_names"]),
        "om_weight": pct(r["om_weight"]),
        "om_T1_small_25": pct(r["om_T1_small_25"]), "om_T1_small_50": pct(r["om_T1_small_50"]),
        "om_T1_small_100": pct(r["om_T1_small_100"]),
        "om_T1_small_100_lo": pct(r["om_T1_small_100_p95"]), "om_T1_small_100_hi": pct(r["om_T1_small_100_p05"]),
        "om_T2_small_100": pct(r["om_T2_small_100"]),
        "sweep_base_rate": pct(rates[0], 0),
        "sweep_flat_T1": pct(r["sweep_flat_T1"], 1, sign=True),
        "sweep_15_T1": pct(r["sweep_15_T1"], 0), "sweep_30_T1": pct(r["sweep_30_T1"], 0),
        "sweep_30_T1_2x": pct(r["sweep_30_T1_2x"], 0),
        "sweep_T2_maxabs": pct(r["sweep_T2_maxabs"], 1),
    }
    r["display"] = f
    (OUT / "results.json").write_text(json.dumps(r, indent=2))
    return r, h1, sim


# -------------------------------------------------------------------------- charts
def base_layout(fig, height=360, **kw):
    fig.update_layout(
        height=height, margin=dict(l=8, r=12, t=8, b=8),
        paper_bgcolor=C["surface"], plot_bgcolor=C["surface"],
        font=dict(family=FONT, size=13, color=C["ink2"]),
        hoverlabel=dict(font=dict(family=FONT, size=12), bgcolor="#ffffff", bordercolor=C["axis"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(color=C["ink2"])),
        **kw,
    )
    fig.update_xaxes(gridcolor=C["grid"], linecolor=C["axis"], zerolinecolor=C["axis"],
                     tickfont=dict(color=C["muted"]), title_font=dict(color=C["ink2"], size=12),
                     automargin=True)
    fig.update_yaxes(gridcolor=C["grid"], linecolor=C["axis"], zerolinecolor=C["axis"],
                     tickfont=dict(color=C["muted"]), title_font=dict(color=C["ink2"], size=12),
                     automargin=True)
    return fig


def chart_c1(h1):
    d = pd.DataFrame(h1["by_decile"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d["decile"], y=d["p_fitted"], mode="lines", name="Fitted logit",
        line=dict(color=C["muted"], width=2, dash="dot"),
        hovertemplate="Decile %{x}<br>Fitted P(scored) %{y:.0%}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=d["decile"], y=d["rate"], mode="markers", name="Observed share scored (95% CI)",
        marker=dict(color=C["ink"], size=10, line=dict(color="#ffffff", width=2)),
        error_y=dict(type="data", symmetric=False, array=d["ci_hi"] - d["rate"],
                     arrayminus=d["rate"] - d["ci_lo"], color=C["ink2"], thickness=1.5, width=0),
        customdata=np.stack([d["scored"], d["n"], d["ci_lo"], d["ci_hi"]], axis=1),
        hovertemplate=("Decile %{x}: %{customdata[0]:.0f} of %{customdata[1]:.0f} scored"
                       "<br>95% CI %{customdata[2]:.0%}–%{customdata[3]:.0%}<extra></extra>")))
    base_layout(fig)
    fig.update_xaxes(title_text="Size decile (1 = smallest, 10 = largest)", dtick=1)
    fig.update_yaxes(title_text="Share of companies scored", tickformat=".0%", range=[0.35, 1.04])
    return fig


def chart_c2(sim):
    rows = [("Today", sim["base"]),
            ("Omnibus, ¼ stop", sim["omnibus"]["0.25"]),
            ("Omnibus, ½ stop", sim["omnibus"]["0.5"]),
            ("Omnibus, all stop", sim["omnibus"]["1.0"])]
    labels = [r[0] for r in rows][::-1]
    fig = go.Figure()
    for t, name, off in (("T1", "T1 · unscored = sector bottom (current rule)", 0.14),
                         ("T2", "T2 · unscored = neutral", -0.14)):
        m = [r[1][t]["small_half_vs_oracle"]["mean"] for r in rows][::-1]
        lo = [r[1][t]["small_half_vs_oracle"]["p05"] for r in rows][::-1]
        hi = [r[1][t]["small_half_vs_oracle"]["p95"] for r in rows][::-1]
        y = [i + off for i in range(len(rows))]
        fig.add_trace(go.Scatter(
            x=m, y=y, mode="markers", name=name,
            marker=dict(color=C[t], size=11, line=dict(color="#ffffff", width=2)),
            error_x=dict(type="data", symmetric=False, array=np.array(hi) - np.array(m),
                         arrayminus=np.array(m) - np.array(lo), color=C[t], thickness=2, width=0),
            customdata=np.stack([lo, hi], axis=1), text=labels,
            hovertemplate=("%{text}<br>Mean %{x:.1%}<br>5th–95th pct %{customdata[0]:.1%} to "
                           "%{customdata[1]:.1%}<extra>" + t + "</extra>")))
    fig.add_vline(x=0, line=dict(color=C["axis"], width=1))
    base_layout(fig, height=340)
    fig.update_yaxes(tickvals=list(range(len(rows))), ticktext=labels, showgrid=False,
                     tickfont=dict(color=C["ink2"]), range=[-0.6, len(rows) - 0.4])
    fig.update_xaxes(title_text="Smaller half vs full information", tickformat=".0%",
                     dtick=0.05, tickangle=0)
    return fig


def chart_c3(sim):
    sw = pd.DataFrame(sim["sweep"])
    rates = sorted(sw["unscored_rate"].unique())
    fig = go.Figure()
    base_rate = rates[0]
    b0 = sw[sw.unscored_rate == base_rate].sort_values("slope_mult")
    fig.add_trace(go.Scatter(
        x=list(b0["slope_mult"]) + list(b0["slope_mult"])[::-1],
        y=list(b0["T1_small_half_vs_oracle_p95"]) + list(b0["T1_small_half_vs_oracle_p05"])[::-1],
        mode="lines", fill="toself", fillcolor="rgba(42,120,214,0.12)", line=dict(width=0),
        hoverinfo="skip", showlegend=False))
    for rate, col in zip(rates, C["ramp"]):
        g = sw[sw.unscored_rate == rate].sort_values("slope_mult")
        lab = f"T1 · {rate:.0%} unscored" + (" (STOXX 600 today)" if rate == base_rate else " (illustrative)")
        fig.add_trace(go.Scatter(
            x=g["slope_mult"], y=g["T1_small_half_vs_oracle_mean"], mode="lines+markers", name=lab,
            line=dict(color=col, width=2), marker=dict(size=8, color=col, line=dict(color="#fff", width=2)),
            hovertemplate="Slope ×%{x}<br>Smaller half %{y:.1%}<extra>" + lab + "</extra>"))
    fig.add_trace(go.Scatter(
        x=b0["slope_mult"], y=b0["T2_small_half_vs_oracle_mean"], mode="lines+markers",
        name=f"T2 · {base_rate:.0%} unscored", line=dict(color=C["T2"], width=2),
        marker=dict(size=8, color=C["T2"], line=dict(color="#fff", width=2)),
        hovertemplate="Slope ×%{x}<br>Smaller half %{y:.1%}<extra>T2</extra>"))
    fig.add_vline(x=1.0, line=dict(color=C["muted"], width=1, dash="dot"))
    fig.add_annotation(x=1.0, y=0.02, yref="y", text="fitted", showarrow=False, xanchor="left",
                       xshift=4, font=dict(color=C["muted"], size=11))
    fig.add_hline(y=0, line=dict(color=C["axis"], width=1))
    base_layout(fig, height=380)
    fig.update_xaxes(title_text="Size gradient in disclosure (× fitted slope)", dtick=0.25)
    fig.update_yaxes(title_text="Smaller half, active vs full information", tickformat=".0%")
    return fig


def chart_c4(sim):
    d = pd.DataFrame(sim["omnibus_at_risk_by_decile"])
    fig = go.Figure(go.Bar(
        x=d["decile"], y=d["share_of_eu"], marker=dict(color=C["muted"], cornerradius=4),
        customdata=np.stack([d["at_risk_in_sample"], d["eu_in_sample"]], axis=1),
        hovertemplate=("Decile %{x}: %{customdata[0]:.0f} of %{customdata[1]:.0f} EU/EEA companies"
                       "<br>below Omnibus I thresholds<extra></extra>"),
        name="Share below thresholds"))
    base_layout(fig, height=300, bargap=0.35, showlegend=False)
    fig.update_xaxes(title_text="Size decile (1 = smallest, 10 = largest)", dtick=1)
    fig.update_yaxes(title_text="EU/EEA sample below thresholds", tickformat=".0%", rangemode="tozero")
    return fig


def to_div(fig, div_id):
    return fig.to_html(include_plotlyjs=False, full_html=False, config=PLOT_CONFIG, div_id=div_id)


# ------------------------------------------------------------------------------ main
def main():
    r, h1, sim = build_results()
    figs = {"c1": chart_c1(h1), "c2": chart_c2(sim), "c3": chart_c3(sim), "c4": chart_c4(sim)}
    for k, fig in figs.items():
        fig.write_html(OUT / f"chart_{k}.html", include_plotlyjs="cdn", config=PLOT_CONFIG)
    cfg = json.loads((ROOT / "site_config.json").read_text())
    env = Environment(loader=FileSystemLoader(ROOT / "src"), undefined=StrictUndefined, autoescape=False)
    html = env.get_template("template.html").render(
        r=r, f=r["display"], cfg=cfg, charts={k: to_div(v, k) for k, v in figs.items()},
        plotly_version=get_plotlyjs_version())
    (SITE / "index.html").write_text(html)
    redirects = "\n".join(f"/{p}   /index.html   200" for p in cfg["recipients"]) + "\n"
    (SITE / "_redirects").write_text(redirects)
    print("wrote outputs/results.json, outputs/chart_c*.html, site/index.html, site/_redirects")
    print(json.dumps(r["display"], indent=1))


if __name__ == "__main__":
    main()
