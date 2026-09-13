#!/usr/bin/env python3
"""
Robustness under telemetry imperfection, scored with the same fault-only
protocol as final_results.py so the numbers are directly comparable to the
main table (the earlier robustness harness used all-incident accuracy).

Stressors: missing telemetry (masked, imputed with the healthy baseline mean),
additive Gaussian measurement noise, and DAG misspecification (removing the
weakest edges by healthy-data R^2).

Writes robustness_faultonly.json.
"""

import contextlib
import copy
import io
import json
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

with contextlib.redirect_stdout(io.StringIO()):
    from datacenter_eval import CausalInterventionRCA, DatacenterCAG, METRICS  # noqa: E402

from final_results import fit_sem, load, rank_sem, regenerate, score  # noqa: E402

SEEDS = [20260227, 42, 7, 1234, 999]
MASK_RATE = 0.20
NOISE_STD = 0.06
EDGE_REMOVAL = 0.20
OUT_PATH = "robustness_faultonly.json"


def perturb_dag(cag, model, fraction):
    """Remove the weakest `fraction` of edges, ranked by healthy-data R^2."""
    out = copy.deepcopy(cag)
    edges = sorted(
        ((p, c) for p, kids in out.dag.items() for c in kids),
        key=lambda e: model.edge_r2.get(e, 0.0),
    )
    for parent, child in edges[: int(round(fraction * len(edges)))]:
        out.dag[parent] = [c for c in out.dag[parent] if c != child]
    out.parents = {m: [] for m in METRICS}
    for parent, kids in out.dag.items():
        for child in kids:
            out.parents[child].append(parent)
    out.descendants = {m: out._get_descendants(m) for m in METRICS}
    return out


def main():
    runs = {k: [] for k in ("clean", "missing", "noise", "dag_perturb")}

    for seed in SEEDS:
        regenerate(seed)
        _, X_test, _, y_test, X_healthy = load(seed)
        cag = DatacenterCAG()
        sem = fit_sem(cag, X_healthy)
        r2f = cag.root_to_fault
        rng = np.random.default_rng(seed)

        runs["clean"].append(
            score(y_test.values, rank_sem(sem, cag, X_test, True), r2f))

        masked = X_test.copy(deep=True)
        drop = rng.random(masked[METRICS].shape) < MASK_RATE
        vals = np.array(masked[METRICS].to_numpy(dtype=float), copy=True)
        means = np.array([sem.baseline[m]["mean"] for m in METRICS])
        vals[drop] = np.broadcast_to(means, vals.shape)[drop]
        masked[METRICS] = vals
        runs["missing"].append(
            score(y_test.values, rank_sem(sem, cag, masked, True), r2f))

        noisy = X_test.copy(deep=True)
        noisy[METRICS] = noisy[METRICS] + rng.normal(0, NOISE_STD, noisy[METRICS].shape)
        runs["noise"].append(
            score(y_test.values, rank_sem(sem, cag, noisy, True), r2f))

        pert = perturb_dag(cag, sem, EDGE_REMOVAL)
        sem_pert = CausalInterventionRCA(pert, use_rf=False)
        sem_pert.fit(X_healthy)
        sem_pert.edge_r2 = {e: v for e, v in sem.edge_r2.items()
                            if e[1] in pert.dag.get(e[0], [])}
        runs["dag_perturb"].append(
            score(y_test.values, rank_sem(sem_pert, pert, X_test, True), r2f))

    agg = {k: {m: [float(np.mean([r[m] for r in v])),
                   float(np.std([r[m] for r in v]))]
               for m in ("top1", "top3", "mrr")}
           for k, v in runs.items()}

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"seeds": SEEDS, "per_seed": runs, "aggregate": agg}, f, indent=2)

    base = agg["clean"]["top1"][0]
    names = {"clean": "Clean", "missing": f"{int(MASK_RATE*100)}% missing telemetry",
             "noise": f"Gaussian noise (sigma={NOISE_STD})",
             "dag_perturb": f"{int(EDGE_REMOVAL*100)}% weakest DAG edges removed"}
    print("\nFault-only Top-1, 5 seeds (same protocol as main table)\n")
    print(f"| {'Condition':38s} | Top-1 | delta |")
    print(f"|{'-'*40}|-------|-------|")
    for k, label in names.items():
        m, s = agg[k]["top1"]
        print(f"| {label:38s} | {m:.3f} ± {s:.3f} | {m-base:+.3f} |")
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
