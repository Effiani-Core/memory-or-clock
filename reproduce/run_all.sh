#!/bin/bash
# Re-runs every experiment and analysis in the paper, in the order they were run.
# Each chain stops at its first failed step and shows it; repeat that step by hand with
# `ucl study-chain --plan <plan> --retry`. Nothing is retried automatically.
# Total: about 30 hours on one laptop CPU (Intel Core i7-12700H, 2 PyTorch threads).
set -euo pipefail
cd "$(dirname "$0")/.."
UCL=".venv/bin/python -m ucl.cli"

# Part I: 30 paired seeds x {LSTM, MLP} on task B, each re-evaluated 3 times.
$UCL study-chain --plan plans/c08_parte1_taller.json
$UCL study-variance --plan plans/c08_parte1_taller.json

# Part II: learning-rate search, then the five main arms (rates chosen by the plan's rule).
$UCL study-chain --plan plans/c01_lr_busqueda.json
$UCL study-chain --plan plans/c01_parte2_fase1.json
# Exploratory clock control (MLP-wide without the clock).
$UCL study-chain --plan plans/c01_parte2_sinreloj.json
$UCL study-contrasts --spec analyses/c01_parte2_desarrollo.json

# Sealed panel: the committed configs/sealed/c01_parte2_B.json was built with
#   $UCL study-sealed-build --name c01_parte2_B --task B --base 3150000 --size 64
# Opening it plays every run once on it; the opening is recorded and refused a second time.
$UCL study-sealed-open --spec analyses/c01_parte2_confirmatorio.json
$UCL study-contrasts --spec analyses/c01_parte2_confirmatorio.json

# Mechanism: time probe on the LSTM state, then the clock arms on the sealed panel.
$UCL study-probe-time --plan plans/c01_parte2_fase1.json --labels LSTM,LSTM-reset --node memory
$UCL study-chain --plan plans/c01_mecanismo.json
$UCL study-sealed-open --spec analyses/c01_mecanismo_sellado.json
$UCL study-contrasts --spec analyses/c01_mecanismo.json

# Numbers and figures of the paper.
.venv/bin/python paper/data/part1_numbers.py
.venv/bin/python paper/data/part2_numbers.py
.venv/bin/python paper/data/mechanism_numbers.py
.venv/bin/python paper/figures/make_figures.py
.venv/bin/python paper/figures/make_figures_part2.py
