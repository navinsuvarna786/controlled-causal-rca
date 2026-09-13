#!/usr/bin/env python3
"""
Concurrent-fault evaluation under the same protocol as final_results.py:
five seeds with FULL dataset regeneration (the earlier harness re-split a
single multi-fault dataset five times), fault incidents only, no healthy
gating, and the divergence baseline evaluated alongside our ranker so the
comparison is like-for-like.

Generator runs with --multi-fault-prob 0.4, matched to the 41% multi-failure
rate for correlated failure groups reported by Gill et al. (SIGCOMM 2011).

Writes multifault_faultonly.json.
"""

import contextlib
import io
import json
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

with contextlib.redirect_stdout(io.StringIO()):
    from datacenter_eval import CausalInterventionRCA, DatacenterCAG, METRICS  # noqa: E402
import baselines_causal as bc  # noqa: E402
from final_results import fit_sem, rank_sem  # noqa: E402

SEEDS = [20260227, 42, 7, 1234, 999]
MULTI_FAULT_PROB = 0.4
OUT_PATH = "multifault_faultonly.json"


def regenerate_multifault(seed):
    subprocess.run(
        [sys.executable, "datacenter_dataset_generator.py", "--seed", str(seed),
         "--multi-fault-prob", str(MULTI_FAULT_PROB)],
        check=True, stdout=subprocess.DEVNULL,
    )


def load(seed):
    df = pd.read_csv("datacenter_dataset.csv")
    agg = {**{m: "median" for m in METRICS}, "fault_type": "first",
           "fault_type_secondary": "first"}
    inc = df.groupby("incident_id", as_index=False).agg(agg)
    X, y, y2 = inc[METRICS], inc["fault_type"], inc["fault_type_secondary"]
    return train_test_split(X, y, y2, test_size=0.3, random_state=seed, stratify=y)


def coverage_scores(y, y2, ranked, root_to_fault):
    """Top-1@primary, Top-1@coverage, Top-2@coverage over all fault incidents,
    and the same restricted to incidents with two initiating roots."""
    acc = {k: [] for k in ("t1p", "t1c", "t2c")}
    multi = {k: [] for k in ("t1p", "t1c", "t2c")}
    n_multi = 0
    for yt, y2t, order in zip(y, y2, ranked):
        if yt == 0:
            continue
        faults = [root_to_fault[r] for r in order]
        truth = {yt} | ({y2t} if y2t != 0 else set())
        is_multi = y2t != 0
        n_multi += is_multi
        vals = {
            "t1p": 1.0 if faults[0] == yt else 0.0,
            "t1c": 1.0 if faults[0] in truth else 0.0,
            "t2c": 1.0 if truth & set(faults[:2]) else 0.0,
        }
        for k, v in vals.items():
            acc[k].append(v)
            if is_multi:
                multi[k].append(v)
    out = {f"all_{k}": float(np.mean(v)) for k, v in acc.items()}
    out.update({f"multi_{k}": (float(np.mean(v)) if v else float("nan"))
                for k, v in multi.items()})
    out["n_multi"] = int(n_multi)
    out["n_fault"] = len(acc["t1p"])
    return out


def main():
    runs = {"ours": [], "ks_rca": []}

    for seed in SEEDS:
        regenerate_multifault(seed)
        X_tr, X_te, y_tr, y_te, y2_tr, y2_te = load(seed)
        cag = DatacenterCAG()
        sem = fit_sem(cag, X_tr[y_tr == 0])
        runs["ours"].append(coverage_scores(
            y_te.values, y2_te.values,
            rank_sem(sem, cag, X_te, True), cag.root_to_fault))

        # MicroCause-style divergence over the identical DAG and split.
        (_, bX_te, _, b_y_te, _, b_y2_te, bX_h, b_base,
         b_models, b_r2, b_metrics) = bc.fit_artifacts(seed)
        parents_map = bc.parents_of()
        ks_table = bc.precompute_ks(bX_h, bX_te, b_metrics)
        ranked = [sorted(bc.ROOT_METRICS,
                         key=lambda r: bc.score_ks_rca(
                             row, ks_table[i], b_r2, parents_map)[r],
                         reverse=True)
                  for i, (_, row) in enumerate(bX_te.iterrows())]
        runs["ks_rca"].append(coverage_scores(
            b_y_te.values, b_y2_te.values, ranked, bc.ROOT_TO_FAULT))

    agg = {}
    for name, v in runs.items():
        agg[name] = {}
        for k in ("all_t1p", "all_t1c", "all_t2c",
                  "multi_t1p", "multi_t1c", "multi_t2c"):
            vals = [r[k] for r in v if not np.isnan(r[k])]
            agg[name][k] = [float(np.mean(vals)), float(np.std(vals))]
        agg[name]["n_multi_total"] = int(sum(r["n_multi"] for r in v))

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"seeds": SEEDS, "multi_fault_prob": MULTI_FAULT_PROB,
                   "per_seed": runs, "aggregate": agg}, f, indent=2)

    print(f"\nConcurrent faults, p={MULTI_FAULT_PROB}, 5 seeds with full regeneration")
    print(f"pooled multi-fault incidents: ours={agg['ours']['n_multi_total']}\n")
    hdr = ("Method", "T1@prim", "T1@cov", "T2@cov",
           "mf T1@prim", "mf T1@cov", "mf T2@cov")
    print("| {:22s} | {} |".format(hdr[0], " | ".join(f"{h:13s}" for h in hdr[1:])))
    for name, label in (("ours", "Counterfactual (ours)"),
                        ("ks_rca", "KS-RCA (MicroCause)")):
        a = agg[name]
        cells = " | ".join(
            f"{a[k][0]:.3f} ± {a[k][1]:.3f}" for k in
            ("all_t1p", "all_t1c", "all_t2c", "multi_t1p", "multi_t1c", "multi_t2c"))
        print(f"| {label:22s} | {cells} |")
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
