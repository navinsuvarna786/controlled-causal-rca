#!/usr/bin/env python3
"""
Robustness and fault-realism analyses on the FULL candidate set, so that every
number in the paper uses one task definition.

Compares the two leading scoring rules only: counterfactual restoration with
the candidate's own deviation included, and CIRCA-style residual scoring.

Writes secondary_fullcand.json.
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
from statsmodels.stats.contingency_tables import mcnemar

with contextlib.redirect_stdout(io.StringIO()):
    from datacenter_eval import CausalInterventionRCA, DatacenterCAG, METRICS  # noqa: E402
from final_results import fit_sem  # noqa: E402
from full_candidate_eval import residual_score, zscore  # noqa: E402
from robustness_faultonly import perturb_dag  # noqa: E402

SEEDS = [20260227, 42, 7, 1234, 999]
TAU = 3.0
MASK_RATE, NOISE_STD, EDGE_REMOVAL = 0.20, 0.06, 0.20
GILL_HEALTHY, GILL_SKEW = 0.714, "38,28,15,9,10"
OUT_PATH = "secondary_fullcand.json"

CONFIGS = [
    ("balanced (default)", []),
    ("measured prevalence", ["--healthy-ratio", str(GILL_HEALTHY)]),
    ("measured class skew", ["--class-weights", GILL_SKEW]),
    ("prevalence + skew", ["--healthy-ratio", str(GILL_HEALTHY),
                           "--class-weights", GILL_SKEW]),
]


def gen(seed, extra):
    subprocess.run([sys.executable, "datacenter_dataset_generator.py",
                    "--seed", str(seed)] + extra, check=True, stdout=subprocess.DEVNULL)


def split(seed):
    df = pd.read_csv("datacenter_dataset.csv")
    inc = df.groupby("incident_id", as_index=False).agg(
        {**{m: "median" for m in METRICS}, "fault_type": "first"})
    X, y = inc[METRICS], inc["fault_type"]
    return train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)


def evaluate(sem, cag, X, y):
    """Returns per-incident correctness for both scorers on the full candidate set."""
    inv = {v: m for m, v in cag.root_to_fault.items()}
    ours, res = [], []
    for (_, row), yt in zip(X.iterrows(), y.values):
        if yt == 0:
            continue
        truth = inv[yt]
        z = {m: zscore(sem, row, m) for m in METRICS}
        cand = [m for m in METRICS if z[m] >= TAU]
        if truth not in cand:
            cand.append(truth)
        ours.append(int(max(cand, key=lambda m: (z[m] + sem.simulate_intervention(row, m),
                                                 z[m])) == truth))
        res.append(int(max(cand, key=lambda m: (residual_score(sem, cag, row, m),
                                                z[m])) == truth))
    return ours, res


def report(label, ours, res, out):
    a, b = np.array(ours), np.array(res)
    n01 = int(((a == 1) & (b == 0)).sum()); n10 = int(((a == 0) & (b == 1)).sum())
    ex = (n01 + n10) < 25
    p = float(mcnemar([[int(((a == 1) & (b == 1)).sum()), n01],
                       [n10, int(((a == 0) & (b == 0)).sum())]],
                      exact=ex, correction=not ex).pvalue)
    print(f"  {label:28s} n={len(a):<5} ours {a.mean():.3f}  residual {b.mean():.3f}  "
          f"p={p:.3g}")
    out[label] = {"n": len(a), "ours": float(a.mean()), "residual": float(b.mean()),
                  "p": p}


def main():
    out = {"robustness": {}, "realism": {}}

    print("\n=== Robustness (full candidate set) ===")
    acc = {k: ([], []) for k in ("clean", "missing", "noise", "dag")}
    for seed in SEEDS:
        gen(seed, [])
        X_tr, X_te, y_tr, y_te = split(seed)
        cag = DatacenterCAG(); sem = fit_sem(cag, X_tr[y_tr == 0])
        rng = np.random.default_rng(seed)

        for key, (Xe, cg, sm) in {
            "clean": (X_te, cag, sem),
        }.items():
            o, r = evaluate(sm, cg, Xe, y_te)
            acc[key][0].extend(o); acc[key][1].extend(r)

        masked = X_te.copy(deep=True)
        drop = rng.random(masked[METRICS].shape) < MASK_RATE
        vals = np.array(masked[METRICS].to_numpy(dtype=float), copy=True)
        means = np.array([sem.baseline[m]["mean"] for m in METRICS])
        vals[drop] = np.broadcast_to(means, vals.shape)[drop]
        masked[METRICS] = vals
        o, r = evaluate(sem, cag, masked, y_te)
        acc["missing"][0].extend(o); acc["missing"][1].extend(r)

        noisy = X_te.copy(deep=True)
        noisy[METRICS] = noisy[METRICS] + rng.normal(0, NOISE_STD, noisy[METRICS].shape)
        o, r = evaluate(sem, cag, noisy, y_te)
        acc["noise"][0].extend(o); acc["noise"][1].extend(r)

        pert = perturb_dag(cag, sem, EDGE_REMOVAL)
        sp = CausalInterventionRCA(pert, use_rf=False); sp.fit(X_tr[y_tr == 0])
        sp.edge_r2 = {e: v for e, v in sem.edge_r2.items()
                      if e[1] in pert.dag.get(e[0], [])}
        o, r = evaluate(sp, pert, X_te, y_te)
        acc["dag"][0].extend(o); acc["dag"][1].extend(r)

    for k, (o, r) in acc.items():
        report(k, o, r, out["robustness"])

    print("\n=== Fault realism (full candidate set) ===")
    for label, extra in CONFIGS:
        o_all, r_all = [], []
        for seed in SEEDS:
            gen(seed, extra)
            X_tr, X_te, y_tr, y_te = split(seed)
            cag = DatacenterCAG(); sem = fit_sem(cag, X_tr[y_tr == 0])
            o, r = evaluate(sem, cag, X_te, y_te)
            o_all.extend(o); r_all.extend(r)
        report(label, o_all, r_all, out["realism"])

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
