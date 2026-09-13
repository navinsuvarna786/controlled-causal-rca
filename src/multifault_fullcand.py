#!/usr/bin/env python3
"""
Concurrent faults on the FULL candidate set.

This is the one regime where interventional semantics have a principled edge.
With two simultaneous roots, a downstream metric carries contributions from
both cascades. A regression residual is a joint observational quantity: it
cannot separate which root produced the deviation, and the second root inflates
it. Counterfactual restoration simulates do(A) alone, which isolates A's
contribution while leaving B's cascade in place.

Reports Top-1@coverage and Top-2@coverage (the prediction matches either true
initiating root), restricted to genuinely multi-fault incidents, over the full
anomalous-metric candidate set of Eq. (2).

Writes multifault_fullcand.json.
"""

import contextlib
import io
import json
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.stats.contingency_tables import mcnemar

with contextlib.redirect_stdout(io.StringIO()):
    from datacenter_eval import DatacenterCAG, METRICS  # noqa: E402
from final_results import fit_sem  # noqa: E402
from full_candidate_eval import residual_score, zscore  # noqa: E402

N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
BASE = [20260227, 42, 7, 1234, 999]
SEEDS = BASE + [20_000 + i for i in range(max(0, N_SEEDS - len(BASE)))]
TAU = 3.0
MULTI_FAULT_PROB = 0.4
OUT_PATH = "multifault_fullcand.json"


def regen(seed):
    subprocess.run([sys.executable, "datacenter_dataset_generator.py",
                    "--seed", str(seed),
                    "--multi-fault-prob", str(MULTI_FAULT_PROB)],
                   check=True, stdout=subprocess.DEVNULL)


def load(seed):
    df = pd.read_csv("datacenter_dataset.csv")
    inc = df.groupby("incident_id", as_index=False).agg(
        {**{m: "median" for m in METRICS}, "fault_type": "first",
         "fault_type_secondary": "first"})
    return train_test_split(inc[METRICS], inc["fault_type"],
                            inc["fault_type_secondary"],
                            test_size=0.3, random_state=seed,
                            stratify=inc["fault_type"])


def main():
    cag = DatacenterCAG()
    single = {k: [] for k in ("ours", "residual", "anomaly")}
    multi1 = {k: [] for k in ("ours", "residual", "anomaly")}
    multi2 = {k: [] for k in ("ours", "residual", "anomaly")}

    for k, seed in enumerate(SEEDS, 1):
        regen(seed)
        X_tr, X_te, y_tr, y_te, y2_tr, y2_te = load(seed)
        sem = fit_sem(cag, X_tr[y_tr == 0])
        inv = {v: m for m, v in cag.root_to_fault.items()}

        for (_, row), yt, y2t in zip(X_te.iterrows(), y_te.values, y2_te.values):
            if yt == 0:
                continue
            truths = {inv[yt]} | ({inv[y2t]} if y2t != 0 else set())
            z = {m: zscore(sem, row, m) for m in METRICS}
            cand = [m for m in METRICS if z[m] >= TAU]
            for t in truths:
                if t not in cand:
                    cand.append(t)

            ranked = {
                "ours": sorted(cand, key=lambda m: -(z[m] + sem.simulate_intervention(row, m))),
                "residual": sorted(cand, key=lambda m: -residual_score(sem, cag, row, m)),
                "anomaly": sorted(cand, key=lambda m: -z[m]),
            }
            tgt = multi1 if y2t != 0 else single
            for name, order in ranked.items():
                tgt[name].append(int(order[0] in truths))
                if y2t != 0:
                    multi2[name].append(int(bool(truths & set(order[:2]))))
        if k % 5 == 0:
            print(f"  {k}/{len(SEEDS)} seeds, {len(multi1['ours'])} multi-fault", flush=True)

    def block(title, d, out_key, out):
        print(f"\n{title}  (n={len(d['ours'])})")
        for name in ("residual", "ours", "anomaly"):
            print(f"    {name:10s} {np.mean(d[name]):.4f}")
        a, b = np.array(d["ours"]), np.array(d["residual"])
        n01 = int(((a == 1) & (b == 0)).sum()); n10 = int(((a == 0) & (b == 1)).sum())
        ex = (n01 + n10) < 25
        p = float(mcnemar([[int(((a == 1) & (b == 1)).sum()), n01],
                           [n10, int(((a == 0) & (b == 0)).sum())]],
                          exact=ex, correction=not ex).pvalue)
        verdict = ("OURS WINS" if a.mean() > b.mean() and p < 0.05 else
                   "residual wins" if a.mean() < b.mean() and p < 0.05 else "tie")
        print(f"    ours vs residual: {a.mean():.4f} vs {b.mean():.4f}  "
              f"{n01}/{n10}  p={p:.4g}  -> {verdict}")
        out[out_key] = {"n": len(a), **{k: float(np.mean(v)) for k, v in d.items()},
                        "n01": n01, "n10": n10, "p": p, "verdict": verdict}

    out = {}
    block("SINGLE-fault incidents, Top-1", single, "single_top1", out)
    block("MULTI-fault incidents, Top-1@coverage", multi1, "multi_top1cov", out)
    block("MULTI-fault incidents, Top-2@coverage", multi2, "multi_top2cov", out)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"seeds": SEEDS, "results": out}, f, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
