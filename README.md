# SC4020-DataMining

Community detection on graphs: Louvain, Leiden and variants on Cora,
email-Eu-core and LFR benchmark graphs. What changed and why is in
[CHANGELOG.md](CHANGELOG.md).

## Setup

Python 3.12. Use the pinned versions: other networkx versions generate
different LFR graphs or Louvain partitions and will not reproduce
`results/raw.csv`.

```
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run order

All scripts run from `Datasets/`.

```
cd Datasets
python download_data.py      # 1. fetch Cora and email-Eu-core into data/raw/
python graph_datasets.py     # 2. sanity-check the loaders
python run_experiments.py    # 3. all experiments -> results/raw.csv (about 1 hour)
python dataset_stats.py      # 4. dataset characterisation -> results/dataset_stats.csv
python make_plots.py         # 5. all figures -> results/figures/
python stats.py              # 6. significance tests for each hypothesis
```

`run_experiments.py` and `make_plots.py` take names to run a subset, e.g.
`python run_experiments.py ablation` or `python make_plots.py consensus`.
Re-running an experiment replaces its rows in `raw.csv`; other rows are kept.

## Experiments (`run_experiments.py`)

| Name | Question |
|---|---|
| `baseline` | How good and how stable is each method at default settings? (20 seeds) |
| `resolution` | How does the resolution parameter gamma change the result? |
| `lfr_mu` | Where does recovery collapse as communities get noisier (LFR mu)? |
| `refinement` | Leiden with 1, 2 or unlimited iterations |
| `ablation` | What does Leiden's refinement step add? Same code with it on and off |
| `consensus_tune` | Picks consensus-clustering settings on separate development graphs |
| `consensus` | Does consensus clustering improve accuracy or stability? |
| `scaling` | Runtime and disconnected communities from 1k to 50k nodes |

## Figures (`make_plots.py`)

| File | Shows |
|---|---|
| `fig1_lfr_mu.png` | NMI and modularity vs LFR mixing |
| `fig2_seed_variance.png` | Spread over 20 seeds |
| `fig3_resolution.png` | Communities found and NMI vs gamma |
| `fig4_clusterings.png` | Detected communities vs ground truth |
| `fig5_contingency.png` | Which true communities get split or merged |
| `fig6_ablation.png` | Leiden with vs without refinement |
| `fig7_degree_dist.png` | Degree distributions |
| `fig8_real_on_lfr.png` | LFR's real mixing, and real datasets placed on the LFR curve |
| `fig9_scaling.png` | Runtime and disconnection vs graph size |
| `fig10_consensus.png` | Consensus clustering: accuracy, gain and seed agreement |
| `fig11_consensus_params.png` | Consensus settings: accuracy and cost |
