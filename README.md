# Controlled Comparison of Causal Scoring Rules for Root Cause Analysis

Reproduction package for

> N. Suvarna, K. Pappu, S. Sharma and A. Parashar, "A Controlled Comparison of
> Causal Scoring Rules for Root Cause Analysis in Spine-Leaf Datacenter
> Networks," in *Proc. 18th IEEE International Conference on Advanced Infocomm
> Technology (ICAIT)*, Auckland, New Zealand, Nov. 2026.

The paper holds the dependency graph, the training data, the candidate set and
the protocol fixed, and varies only the scoring rule: counterfactual restoration
(with and without the candidate-deviation correction), residual scoring in the
style of CIRCA, distributional divergence in the style of MicroCause, and
anomaly magnitude. Every number in the paper is produced by a script in this
repository.

## Layout

```
run_all.sh           one-shot reproduction of every table and figure
requirements.txt     pinned dependencies (Python 3.12)
src/                 generator, DAG, scoring rules, evaluation and figure scripts
data/                frozen inputs: the 26-edge DAG, layer map, and the AIOps subgraph
results/             the JSON files the paper's tables were typeset from
paper/               LaTeX source, bibliography and figures of the camera-ready
```

## Setup

```bash
python3.12 -m venv venv
venv/bin/pip install -r requirements.txt
```

The version pins matter. The synthetic generator seeds NumPy explicitly and is
deterministic for a given seed *and* a given NumPy version; unpinned NumPy
produces a different dataset for the same seed. The pinned set is the one the
paper reports (scikit-learn 1.8, NumPy 2.4, SciPy 1.17, pandas 3.0,
statsmodels 0.14).

## Reproduce

```bash
bash run_all.sh            # uses ./venv/bin/python
PYTHON=python3 bash run_all.sh   # or any interpreter with requirements.txt installed
```

Steps 6–13 produce the paper. Steps 1–5 are an earlier single-seed diagnostic
pipeline kept for completeness; they read the frozen `data/datacenter_dataset.csv`
rather than regenerating it. Everything in the paper regenerates its dataset
per seed inside the script.

The full run takes roughly an hour on a laptop; the 40-seed power test
(step 9) is the slow part.

## Where each result comes from

| Paper item | Script | Output |
|---|---|---|
| Table II, primary comparison, 20 seeds | `src/full_candidate_eval.py 20` | `results/full_candidate_eval.json` |
| Table II p-values, all pairwise tests | `src/significance_all.py` | `results/significance_all.json` |
| Table III, accuracy by anomaly rank of the root | `src/cluster_intervals.py` | `results/cluster_intervals.json` |
| Candidate-deviation correction, held-out check | `src/cluster_intervals.py` | `results/cluster_intervals.json` |
| Estimator-variance test (Sec. VI-A) | `src/estimator_variance.py` | `results/estimator_variance.json` |
| Concurrent faults, full candidate set | `src/multifault_fullcand.py 20` | `results/multifault_fullcand.json` |
| Table IV, pre-registered partitions | `src/regime_analysis.py 20`, `src/paired_intervals.py 20` | `results/regime_analysis.json`, `results/paired_intervals.json` |
| Table V, restricted candidate set | `src/final_results.py` | `results/final_results.json` |
| Concurrent faults, restricted set | `src/multifault_faultonly.py` | `results/results_multifault_coverage.json` |
| Five-seed vs. forty-seed power test | `src/ambiguous_power.py 40` | `results/ambiguous_power.json` |
| Table VI, robustness and realism | `src/secondary_fullcand.py`, `src/sensitivity_realism.py` | `results/secondary_fullcand.json` |
| Inference latency vs. graph size | `src/scaling_experiment.py` | `results/scaling_results.json` |
| Structure recovery on AIOps 2020 (RQ3) | `src/aiops_subgraph_discovery.py`, `src/aiops_eval.py` | `data/aiops_*`, `results/structural_similarity_report.json` |
| Fig. 2, architecture-derived DAG | `src/generate_figure2.py` | `outputs/figures/` |
| Fig. 3, results comparison | `src/generate_figure_results.py` | `outputs/figures/` |
| Fig. 4, DAG comparison | `src/generate_figure_dag_comparison.py` | `outputs/figures/` |

## The graph

`data/datacenter_adjacency_binary.csv` is the 28-node, 26-edge
architecture-derived DAG; `src/dag.py` is the same graph as code with the
provenance class of every edge. `data/datacenter_node_layers.csv` assigns each
metric to its layer (physical, control plane, overlay, server).

## AIOps Challenge 2020 telemetry

The structure-recovery experiment uses the public AIOps Challenge 2020 corpus
(https://github.com/NetManAIOps/AIOps-Challenge-2020-Data). The 3 GB archive
is not included. `src/download_data.sh` fetches it given `DATA_URL`, and
`src/aiops_subgraph_discovery.py` reduces it to the 40-metric subgraph in
`data/aiops_*`, which is included so that the discovery result can be checked
without the download.

## License

Apache License 2.0. See `LICENSE`.
