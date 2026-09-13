#!/usr/bin/env bash
# Reproduces every paper artifact end-to-end against frozen canonical CSVs.
# Single-source-of-truth runner — no version forks, no stale variants.
#
# Usage: bash run_all.sh
# Output: outputs/{eval,robustness,aiops,struct_sim,figures,final}/
#
# Steps 1-5 are the original diagnostic pipeline. Steps 6-9 produce the numbers
# reported in the paper, all under one protocol (fault incidents only, no
# healthy gating, five seeds with full dataset regeneration per seed).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-${ROOT}/venv/bin/python}"
SRC="${ROOT}/src"
DATA="${ROOT}/data"
OUT="${ROOT}/outputs"

if [[ ! -x "${PY}" ]] && ! command -v "${PY}" >/dev/null 2>&1; then
    echo "No Python found. Either create the venv:"
    echo "  python3.12 -m venv ${ROOT}/venv"
    echo "  ${ROOT}/venv/bin/pip install -r ${ROOT}/requirements.txt"
    echo "or point PYTHON at an interpreter with requirements.txt installed."
    exit 1
fi

# Each step gets isolated cwd because scripts use relative paths.
# Symlink scripts (subprocess needs them resolvable in cwd) + copy data.
prep_dir() {
    local stepdir="$1"
    shift
    rm -rf "${stepdir}" && mkdir -p "${stepdir}"
    for f in "$@"; do
        if [[ "$f" == *.py ]]; then
            ln -sf "${SRC}/${f}" "${stepdir}/${f}"
        else
            cp "${DATA}/${f}" "${stepdir}/${f}"
        fi
    done
}

DC_DATA=(
    datacenter_dataset.csv
    datacenter_adjacency_binary.csv
    datacenter_adjacency_weighted.csv
    datacenter_node_layers.csv
    datacenter_topological_order.txt
    datacenter_structural_summary.json
)

DC_SCRIPTS=(
    dag.py
    datacenter_dataset_generator.py
    datacenter_eval.py
)

AIOPS_DATA=(
    aiops_adjacency_binary.csv
    aiops_adjacency_weighted.csv
    aiops_node_layers.csv
    aiops_reduced_dataset.csv
    aiops_reduced_subgraph.json
    aiops_structural_summary.json
    aiops_topological_order.txt
)

# 1. Datacenter eval (single-seed + cross-seed consolidation)
echo "=== [1/13] Datacenter eval ==="
prep_dir "${OUT}/eval" "${DC_DATA[@]}" "${DC_SCRIPTS[@]}"
( cd "${OUT}/eval" && "${PY}" datacenter_eval.py > stdout.log 2>&1 )
echo "  $(grep -E 'top1_linear|top1_rf|top1_pearson|top1_pagerank' "${OUT}/eval/results_datacenter_rca.json" | tr -d ',')"

# 2. Robustness (importance-weighted edge ablation)
echo "=== [2/13] Robustness ==="
prep_dir "${OUT}/robustness" "${DC_DATA[@]}" "${DC_SCRIPTS[@]}"
ln -sf "${SRC}/robustness.py" "${OUT}/robustness/robustness.py"
( cd "${OUT}/robustness" && "${PY}" robustness.py > stdout.log 2>&1 )
echo "  $(grep -E 'Linear: ' "${OUT}/robustness/stdout.log" | tail -4)"

# 3. AIOps real-world
echo "=== [3/13] AIOps real-world ==="
prep_dir "${OUT}/aiops" "${AIOPS_DATA[@]}"
ln -sf "${SRC}/aiops_eval.py" "${OUT}/aiops/aiops_eval.py"
( cd "${OUT}/aiops" && "${PY}" aiops_eval.py > stdout.log 2>&1 )
echo "  $(grep -E 'Causal Top-1|Causal Top-3|Causal MRR|Invocation' "${OUT}/aiops/stdout.log")"

# 4. Structural similarity
echo "=== [4/13] Structural similarity ==="
prep_dir "${OUT}/struct_sim" \
    aiops_adjacency_binary.csv \
    datacenter_adjacency_binary.csv \
    aiops_structural_summary.json \
    datacenter_structural_summary.json
ln -sf "${SRC}/structural_similarity.py" "${OUT}/struct_sim/structural_similarity.py"
( cd "${OUT}/struct_sim" && "${PY}" structural_similarity.py > stdout.log 2>&1 )

# 5. Figures
echo "=== [5/13] Figures ==="
prep_dir "${OUT}/figures" \
    dag_structure_for_fig2.json \
    aiops_structural_summary.json \
    datacenter_structural_summary.json
ln -sf "${SRC}/generate_figure2.py" "${OUT}/figures/generate_figure2.py"
ln -sf "${SRC}/generate_figure3.py" "${OUT}/figures/generate_figure3.py"
( cd "${OUT}/figures" && "${PY}" generate_figure2.py > fig2.log 2>&1 )
( cd "${OUT}/figures" && "${PY}" generate_figure3.py > fig3.log 2>&1 )
ls "${OUT}/figures"/*.{png,pdf} 2>/dev/null

# Steps 6-11 produce every number reported in the paper, all under one
# protocol: the full candidate set of Eq. (2), fault incidents only, no healthy
# gating, and full dataset regeneration per seed.
PAPER_SCRIPTS=(
    dag.py
    datacenter_dataset_generator.py
    datacenter_eval.py
    baselines_causal.py
    final_results.py
    full_candidate_eval.py
)

prep_dir "${OUT}/final" "${DC_DATA[@]}" "${PAPER_SCRIPTS[@]}"

# 6. Restricted-candidate comparison (Table III) -- the benchmarking shortcut.
echo "=== [6/13] Restricted candidate set (Table III) ==="
( cd "${OUT}/final" && "${PY}" final_results.py > restricted.log 2>&1 )
grep -A9 '^| Method' "${OUT}/final/restricted.log" || true

# 7. Primary comparison on the full candidate set (Table II).
echo "=== [7/13] Full candidate set, 20 seeds (Table II) ==="
( cd "${OUT}/final" && "${PY}" full_candidate_eval.py 20 > fullcand.log 2>&1 )
grep -E '^  (counterfactual|anomaly|residual)' "${OUT}/final/fullcand.log" || true

# 8. Pairwise significance for every comparison the paper makes.
echo "=== [8/13] Significance battery ==="
ln -sf "${SRC}/significance_all.py" "${OUT}/final/significance_all.py"
( cd "${OUT}/final" && "${PY}" significance_all.py > significance.log 2>&1 )
grep -E '^  (ours|clean|residual)' "${OUT}/final/significance.log" | head -8 || true

# 9. Statistical power: the five-seed effect against forty seeds.
echo "=== [9/13] Power test, 40 seeds ==="
ln -sf "${SRC}/ambiguous_power.py" "${OUT}/final/ambiguous_power.py"
( cd "${OUT}/final" && "${PY}" ambiguous_power.py 40 > power.log 2>&1 )
grep -E 'refinement|discordant' "${OUT}/final/power.log" || true

# 10. Regime analysis and concurrent faults on the full candidate set.
echo "=== [10/13] Regimes and concurrent faults ==="
for f in regime_analysis.py multifault_fullcand.py; do
    ln -sf "${SRC}/${f}" "${OUT}/final/${f}"
done
( cd "${OUT}/final" && "${PY}" regime_analysis.py 20 > regime.log 2>&1 )
( cd "${OUT}/final" && "${PY}" multifault_fullcand.py 20 > multifault.log 2>&1 )
grep -E '^  (ALL|H1|H2|H3)' "${OUT}/final/regime.log" || true

# 11. Robustness, realism, and the comparative figure.
echo "=== [11/13] Robustness, realism, figure ==="
for f in robustness_faultonly.py secondary_fullcand.py generate_figure_results.py; do
    ln -sf "${SRC}/${f}" "${OUT}/final/${f}"
done
( cd "${OUT}/final" && "${PY}" secondary_fullcand.py > secondary.log 2>&1 )
( cd "${OUT}/final" && "${PY}" generate_figure_results.py >> secondary.log 2>&1 )
grep -E '^  (clean|20%|Gaussian|balanced|measured|prevalence)' "${OUT}/final/secondary.log" || true

# 12. Seed-clustered intervals, paired intervals, estimator variance, and the
#     concurrent-fault run (Tables III and V, Section VI-A).
echo "=== [12/13] Intervals, estimator variance, concurrent faults ==="
for f in cluster_intervals.py paired_intervals.py estimator_variance.py multifault_faultonly.py sensitivity_realism.py; do
    ln -sf "${SRC}/${f}" "${OUT}/final/${f}"
done
( cd "${OUT}/final" && "${PY}" cluster_intervals.py > cluster.log 2>&1 )
( cd "${OUT}/final" && "${PY}" paired_intervals.py 20 > paired.log 2>&1 )
( cd "${OUT}/final" && "${PY}" estimator_variance.py > evar.log 2>&1 )
( cd "${OUT}/final" && "${PY}" multifault_faultonly.py > mf.log 2>&1 )
( cd "${OUT}/final" && "${PY}" sensitivity_realism.py > realism.log 2>&1 )
ls "${OUT}/final"/*.json

# 13. Inference latency against graph size (Section VI-E) and the DAG
#     comparison figure (Fig. 4).
echo "=== [13/13] Scaling and DAG comparison figure ==="
( cd "${ROOT}" && "${PY}" "${SRC}/scaling_experiment.py" > "${OUT}/scaling.log" 2>&1 )
( cd "${ROOT}" && "${PY}" "${SRC}/generate_figure_dag_comparison.py" >> "${OUT}/figures/fig4.log" 2>&1 )
ls "${OUT}/scaling" "${OUT}/figures"

echo ""
echo "=== DONE — outputs in ${OUT}/ ==="
