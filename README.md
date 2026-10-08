# Separating efficiency from disclosure

How missing environmental data shapes the size exposure of a Resource Efficiency tilt.

An independent analysis built on public data and a stylised model. The published report is `site/index.html`.

## Question

Resource-efficiency signals often score a company only if it discloses enough on at least 2 of 3 metrics (carbon, water, waste), and treat unscored companies as inefficient.

If disclosure rises with company size, does that rule turn an efficiency tilt partly into a size tilt? And would a different treatment of *absence*, still using no estimated or vendor data, avoid it?

## Pipeline

Each step writes to `data/` or `outputs/`. All random draws use fixed seeds.

| Step | Script | Output |
|---|---|---|
| 1. Universe | `src/01_universe.py data/raw/EXSA_holdings.csv` | `data/universe.csv`: 600 equities, size deciles on the 470 non-financials |
| 2. Sample | `src/02_sample.py` | `data/disclosure_sample.csv`: 10 random names per decile. Run once; it refuses to overwrite collected data |
| 2b. Collection | by hand; each batch logged in `data/raw/batch*.json` and merged in order with `src/update_sample.py` | flags, source URLs and verification level. See `data/COLLECTION_NOTES.md` |
| 3. Disclosure model (H1) | `src/03_disclosure_model.py` | `outputs/h1.json` |
| 4. Simulation (H2–H4, sweep) | `src/04_simulation.py` | `outputs/sim.json` |
| 5. Report | `src/05_build_report.py` | `outputs/results.json`, `outputs/chart_c*.html`, `site/index.html`, `site/_redirects` |

To reproduce the results from the committed data:

```bash
pip install -r requirements.txt
./run_all.sh
```

Every number in the report text is read from `outputs/results.json`. Nothing is typed by hand.

Author name, date, contact links and the personal link paths live in `site_config.json`. `site/_redirects` serves the same page on each personal path (e.g. `/a7k2`), so the host's analytics can show which link was opened.

## Repository layout

```
data/raw/        iShares holdings export + collection batch logs
data/            universe.csv, disclosure_sample.csv, COLLECTION_NOTES.md
src/             pipeline scripts and the page template
outputs/         h1.json, sim.json, results.json, standalone charts
site/            index.html + _redirects (the deployed folder)
```

## Model in one paragraph

Each run gives every non-financial STOXX Europe 600 constituent a true efficiency score drawn from N(0,1), independent of size by construction. Each company is then scored with the probability fitted in step 3. A sector-neutral, long-only linear tilt follows, `w = b·(1 + λ(s − s̄_sector))` with λ = 0.5, run 1,000 times. Treatments of unscored companies:

- **T1:** sector bottom score (penalise non-disclosure)
- **T2:** neutral, held at benchmark weight
- **T3:** T2 plus size-neutralised scores
- **T4:** T1 plus size-neutralised scores

Each is compared, run by run, with an oracle that scores every company. The Omnibus I scenario marks EU/EEA companies below 1,000 average employees or €450m turnover as at risk of no longer disclosing. The sensitivity sweep varies the disclosure–size slope from 0 to 2× its fitted value, at 7%, 15% and 30% unscored.

## Limitations

- **Sample size.** 100 companies, 7 unscored, so the fitted slope is imprecise.
- **Disclosure flags.** First pass via Tracenable, with primary-source checks wherever scored status depended on it.
- **Simulated scores.** Resource Efficiency scores are simulated, not real.
- **Size measure.** Size is proxied by ETF weight.
- **Omnibus flags.** These are approximate.

The report covers each in full.

## Data sources

- iShares STOXX Europe 600 UCITS ETF (DE) holdings export, 8 October 2026 (`data/raw/`)
- Company annual and sustainability reports, linked per row in `data/disclosure_sample.csv`
- Tracenable company pages (tracenable.com), which cite the underlying filings
