#!/usr/bin/env bash
# Rebuild every result from the raw holdings file and the collected disclosure sample.
set -euo pipefail
cd "$(dirname "$0")"
python3 src/universe.py data/raw/EXSA_holdings.csv
python3 src/disclosure_model.py
python3 src/simulation.py
python3 src/build_report.py
