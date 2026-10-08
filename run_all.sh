#!/usr/bin/env bash
# Rebuild every result from the raw holdings file and the collected disclosure sample.
set -euo pipefail
cd "$(dirname "$0")"
python3 src/01_universe.py data/raw/EXSA_holdings.csv
python3 src/03_disclosure_model.py
python3 src/04_simulation.py
python3 src/05_build_report.py
