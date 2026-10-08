# How `disclosure_sample.csv` was collected

**Sample.** 100 non-financial STOXX Europe 600 constituents, 10 per size decile, drawn with a fixed seed by `src/02_sample.py` from the iShares EXSA holdings export (8 Oct 2026). Size = fund market value, which is proportional to free-float market cap.

**What counts as disclosed.** A metric is `Y` when the company publishes a quantified figure for its own operations in its latest report (FY2024 or FY2025):

- `scope12_ghg`: absolute Scope 1 and Scope 2 emissions (tCO2e)
- `water`: water withdrawal **or** consumption volume
- `waste`: total waste generated (tonnes)

A company is **scored** when at least 2 of the 3 are `Y`, mirroring the rule Osmosis describes publicly.

**Sources and protocol.**

1. **First pass: Tracenable.** Each company's Tracenable page (`tracenable.com/company/<slug>/ghg-emissions`) was read. Tracenable extracts reported figures from company filings and names the source document, which is recorded in `notes`. A metric tab that exists is treated as positive evidence of disclosure.
2. **Missing water tab = `U` (unverified), not `N`.** Tracenable's water coverage has gaps: IHG, STMicroelectronics and Boliden all publish water data but have no water tab. `U` never changes a company's scored status, because every `U` row already has GHG and waste.
3. **Primary-source check where it matters.** Any company with fewer than 2 metrics on Tracenable was checked against its own report before being recorded as unscored. This caught one Tracenable false negative: Sodexo's waste (234,609 t, URD FY2025, ESRS E5-5), which flipped Sodexo to scored.
4. **Mismatched entries were not used.** The `bam` page on Tracenable did not look like Royal BAM Group, so BAM was read directly from its own report.

**Verification levels** (`verification` column):

- `tracenable`: 89 rows
- `primary`: 6 rows (read in the company's own document)
- `partial`: 5 rows (unscored on Tracenable; the company's summary or overview showed no water or waste figures, but the full environmental section could not be fetched)

**Spot-checks.** 15 rows were checked by hand:

- 6 full primary reads
- 5 partial source checks
- 4 waste tabs (Rational, Covivio, Sacyr, UMG), to confirm a tab always holds real values. No false positives were found.

**Omnibus I flag** (`omnibus_in_scope`, EU/EEA domicile only):

- `N` when the firm is below 1,000 average employees **or** below €450m net turnover.
- `Y` when the firm is clearly above both thresholds (large listed groups; exact figures recorded where looked up).
- `n/a` outside the EU/EEA. Domicile is proxied by iShares "Location".

**Limitations.**

- Tracenable data lags: some rows reflect FY2024.
- `partial` rows may hide disclosures deeper in long PDFs. The modelling step reports results with and without them.
