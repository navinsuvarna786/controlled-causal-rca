#!/usr/bin/env python3
"""
Definitive main-results table under a single, uniform evaluation protocol.

Every method is scored the same way:
  * fault incidents only (healthy incidents are a detection problem, not a
    ranking problem, and are reported separately);
  * no healthy-threshold gating, so no method is credited or penalised for a
    detection threshold that differs between implementations;
  * the same architecture-informed DAG, the same healthy-only training split,
    and the same candidate root set;
  * Top-1 / Top-3 / MRR, mean +/- std over 5 seeds.

This replaces the earlier mixed protocol in which the SEM rankers were scored
with all-incident accuracy while the correlation and centrality baselines were
scored fault-only, which made the two families non-comparable.

Run from a directory holding datacenter_dataset.csv and the DAG artefacts.
Writes final_results.json.
"""

import contextlib
import io
import json
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# datacenter_eval is a script, not a library: importing it runs a full
# evaluation and prints several hundred lines. Swallow that here.
with contextlib.redirect_stdout(io.StringIO()):
    from datacenter_eval import (  # noqa: E402
        CausalInterventionRCA,
        DatacenterCAG,
        METRICS,
        PageRankBaseline,
        PearsonBaseline,
    )
import baselines_causal as bc  # noqa: E402

SEEDS = [20260227, 42, 7, 1234, 999]
MARGIN = 0.25
OUT_PATH = "final_results.json"


def regenerate(seed):
    subprocess.run(
        [sys.executable, "datacenter_dataset_generator.py", "--seed", str(seed)],
        check=True, stdout=subprocess.DEVNULL,
    )


def load(seed):
    df = pd.read_csv("datacenter_dataset.csv")
    inc = (df.groupby("incident_id", as_index=False)
             .agg({**{m: "median" for m in METRICS}, "fault_type": "first"}))
    X, y = inc[METRICS], inc["fault_type"]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    return Xtr, Xte, ytr, yte, Xtr[ytr == 0]


def fit_sem(cag, X_healthy, use_rf=False):
    model = CausalInterventionRCA(cag, use_rf=use_rf)
    model.fit(X_healthy)
    r2 = {}
    for parent, kids in cag.dag.items():
        for child in kids:
            d = X_healthy[[parent, child]].dropna()
            if len(d) < 2:
                r2[(parent, child)] = 0.0
                continue
            lr = LinearRegression().fit(d[[parent]].values, d[child].values)
            v = float(lr.score(d[[parent]].values, d[child].values))
            r2[(parent, child)] = v if np.isfinite(v) else 0.0
    model.edge_r2 = r2
    return model


def rank_sem(model, cag, X, refine):
    """Anomaly ranking, optionally refined between the top two candidates by
    counterfactual restoration when their anomaly scores are within MARGIN."""
    out = []
    for _, row in X.iterrows():
        s = model.root_anomaly_scores(row)
        order = sorted(cag.root_metrics, key=lambda r: s[r], reverse=True)
        if refine and (s[order[0]] - s[order[1]]) <= MARGIN:
            a, b = order[0], order[1]
            best = a if model.simulate_intervention(row, a) >= \
                model.simulate_intervention(row, b) else b
            order = [best] + [r for r in order if r != best]
        out.append(order)
    return out


def rank_causal_baseline(scoring, row_iter, artifacts):
    """RR-RCA (CIRCA-style residual) or KS-RCA (MicroCause-style divergence)."""
    (X_test, X_healthy, baseline, edge_models, edge_r2, metrics) = artifacts
    parents_map = bc.parents_of()
    ks_table = bc.precompute_ks(X_healthy, X_test, metrics) if scoring == "ks_rca" else None
    out = []
    for i, (_, row) in enumerate(row_iter):
        if scoring == "rr_rca":
            scores = bc.score_rr_rca(row, baseline, edge_models, edge_r2, parents_map)
        else:
            scores = bc.score_ks_rca(row, ks_table[i], edge_r2, parents_map)
        out.append(sorted(bc.ROOT_METRICS, key=lambda r: scores[r], reverse=True))
    return out


def score(y_true, ranked, root_to_fault, k=3):
    """Fault-only Top-1 / Top-k / MRR. Healthy incidents are excluded."""
    t1, tk, rr = [], [], []
    for yt, order in zip(y_true, ranked):
        if yt == 0:
            continue
        faults = [root_to_fault[r] for r in order]
        t1.append(1.0 if faults[0] == yt else 0.0)
        tk.append(1.0 if yt in faults[:k] else 0.0)
        rr.append(next((1.0 / i for i, f in enumerate(faults, 1) if f == yt), 0.0))
    return {"top1": float(np.mean(t1)), "top3": float(np.mean(tk)),
            "mrr": float(np.mean(rr)), "n": len(t1)}


def main():
    per_seed = {m: [] for m in
                ("ours", "ours_no_refine", "rr_rca", "ks_rca", "pearson", "pagerank")}

    for seed in SEEDS:
        regenerate(seed)
        Xtr, Xte, ytr, yte, Xh = load(seed)
        cag = DatacenterCAG()
        r2f = cag.root_to_fault
        sem = fit_sem(cag, Xh)

        per_seed["ours"].append(
            score(yte.values, rank_sem(sem, cag, Xte, True), r2f))
        per_seed["ours_no_refine"].append(
            score(yte.values, rank_sem(sem, cag, Xte, False), r2f))

        # CIRCA-/MicroCause-style baselines over the identical DAG and split.
        (_, bX_test, _, b_y_test, _, _, bX_healthy, b_baseline,
         b_edge_models, b_edge_r2, b_metrics) = bc.fit_artifacts(seed)
        art = (bX_test, bX_healthy, b_baseline, b_edge_models, b_edge_r2, b_metrics)
        for name in ("rr_rca", "ks_rca"):
            ranked = rank_causal_baseline(name, bX_test.iterrows(), art)
            per_seed[name].append(score(b_y_test.values, ranked, bc.ROOT_TO_FAULT))

        pear = PearsonBaseline(r2f); pear.fit(Xtr, ytr)
        per_seed["pearson"].append(
            score(yte.values, pear.predict_ranked(Xte), r2f))

        pr = PageRankBaseline(r2f); pr.fit(Xtr)
        per_seed["pagerank"].append(
            score(yte.values, pr.predict_ranked(Xte), r2f))

    agg = {}
    for name, runs in per_seed.items():
        agg[name] = {
            f"{metric}_{stat}": float(fn([r[metric] for r in runs]))
            for metric in ("top1", "top3", "mrr")
            for stat, fn in (("mean", np.mean), ("std", np.std))
        }
        agg[name]["n_fault_mean"] = float(np.mean([r["n"] for r in runs]))

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"seeds": SEEDS, "protocol":
                   "fault-only, no healthy gating, shared DAG and split",
                   "per_seed": per_seed, "aggregate": agg}, f, indent=2)

    labels = {
        "ours": "Counterfactual SEM (ours)",
        "ours_no_refine": "  ablation: anomaly-only (no restore)",
        "rr_rca": "RR-RCA (CIRCA-style residual)",
        "ks_rca": "KS-RCA (MicroCause-style divergence)",
        "pearson": "Pearson correlation",
        "pagerank": "PageRank centrality",
    }
    print(f"\nFault-only, 5 seeds, shared DAG, no healthy gating "
          f"(n={agg['ours']['n_fault_mean']:.0f} fault incidents/seed)\n")
    print(f"| {'Method':38s} | Top-1 | Top-3 | MRR |")
    print(f"|{'-'*40}|-------|-------|-----|")
    for name, label in labels.items():
        a = agg[name]
        print(f"| {label:38s} | {a['top1_mean']:.3f} ± {a['top1_std']:.3f} "
              f"| {a['top3_mean']:.3f} ± {a['top3_std']:.3f} "
              f"| {a['mrr_mean']:.3f} ± {a['mrr_std']:.3f} |")
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
