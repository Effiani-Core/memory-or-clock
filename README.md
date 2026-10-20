# Memory or a Clock?

Code, plans and paper source for *Memory or a Clock? Auditing a Memory Advantage in
XLand-MiniGrid with Seeds, Maps and Evaluation Noise Apart* (Rodrigo Reveron-Fonseca, 2026).

On a partially observable XLand-MiniGrid task, an LSTM beats a memory-free MLP by 25 points.
Given the time step, memory-free agents come within 3 to 5 points of it, so most of the
apparent memory advantage is an advantage of knowing the time. The paper also splits the
variance of the comparison into training seed, map and evaluation noise, and gives the number of
seeds each metric needs.

## What is here

| Path | Contents |
|---|---|
| `src/ucl/` | The laboratory: environment adapter, architectures, A2C training, evaluation, re-evaluation, event-driven study chain, sealed panels, variance and contrast analyses, time probe |
| `configs/study_001/` | Study protocol, map panels, tasks and every architecture in the paper |
| `configs/sealed/` | The sealed test panel (64 task-B maps) |
| `plans/` | Preregistered run plans, one per chain (Part I, learning-rate search, Part II, clock control, mechanism) |
| `analyses/` | Frozen analysis specifications (contrasts, thresholds, Holm families) |
| `docs/` | Environment notes and the study plan (`PLAN_STUDY_001.md`, in Spanish) |
| `paper/` | LaTeX source, figure scripts, and the scripts that export every reported number |
| `reproduce/` | `run_all.sh` (the whole study, in order) and `determinism.py` |
| `tests/` | Unit tests, including the statistics on synthetic data with known answers |

Runs, re-evaluations, journals, sealed-panel results and all checkpoints are archived
separately on Zenodo (924 MB compressed), see **Data** below.

## Setup

Linux x86_64 (we used WSL2), Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run pytest
```

Everything runs on CPU. Figures also need `matplotlib`.

## Reproduce

```bash
bash reproduce/run_all.sh
```

This re-trains all 188 runs one at a time (about 30 hours on one laptop CPU), re-evaluates
them, opens the sealed panel and writes every number and figure of the paper. Each step is
journaled under `runs/chains/`; a failed step stops the chain and stays visible until it is
repeated by hand with `--retry`. Re-running a seed reproduces its final weights bit for bit
(`python reproduce/determinism.py`).

To check the paper's numbers without re-training, extract the data archive at the repository
root (it creates `runs/`) and run only the last block of `reproduce/run_all.sh`.

## Data

Per-run summaries, learning curves, per-episode outcomes, re-evaluations, chain journals,
sealed-panel results and every checkpoint (924 MB):
[doi:10.5281/zenodo.23512311](https://doi.org/10.5281/zenodo.23512311).

## Laboratory interface

The package also contains a Streamlit dashboard and other experiments of the laboratory that
the paper does not use (`ucl dashboard`; see `docs/LAB_README.md`, in Spanish).

## License

MIT. See `LICENSE` and `CITATION.cff`.
