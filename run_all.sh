#!/usr/bin/env bash
set -e

python3 -m scripts.run_clean
python3 -m scripts.run_ebm
python3 -m scripts.run_noisy

python3 plots/plot_curves.py
