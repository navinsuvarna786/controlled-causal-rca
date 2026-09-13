#!/usr/bin/env python3
"""
Where does each scoring rule win?

A controlled comparison is only useful if it says when each rule should be
preferred. This script partitions incidents by three properties fixed before
looking at any result, and compares counterfactual restoration (with the
candidate's own deviation included) against CIRCA-style residual scoring in
each partition.

Pre-registered hypotheses:

  H1 Gray failures. When a fault is real but its telemetry signature is only
     half strength, the root's own deviation is a weak signal. Residual scoring
     leans on that deviation, so counterfactual restoration -- which asks what
     repairing the candidate would remove downstream -- should degrade less.

  H2 Deep roots. Counterfactual restoration propagates an intervention through
     the whole descendant subtree, while residual scoring inspects each
     descendant against its immediate parents. The deeper the causal chain, the
     more the former should gain.

  H3 Buried roots. When the true root is not among the loudest candidates,
     deviation-based evidence is misleading. This is an oracle-defined
     partition, used for analysis only, never for method selection.

All partitions are reported, including those where our method loses, and the
count of partitions tested is stated so the reader can judge multiplicity.

Writes regime_analysis.json.
"""

import contextlib
import io
import json
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.stats.contingency_tables import mcnemar

with contextlib.redirect_stdout(io.StringIO()):
    from datacenter_eval import DatacenterCAG, METRICS  # noqa: E402
from final_results import fit_sem, regenerate  # noqa: E402
from full_candidate_eval import residual_score, zscore  # noqa: E402

N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
BASE = [20260227, 42, 7, 1234, 999]
SEEDS = BASE + [20_000 + i for i in range(max(0, N_SEEDS - len(BASE)))]
TAU = 3.0
OUT_PATH = "regime_analysis.json"


def load_with_gray(seed):
    df = pd.read_csv("datacenter_dataset.csv")
    agg = {**{m: "median" for m in METRICS}, "fault_type": "first", "is_gray": "max"}
    inc = df.groupby("incident_id", as_index=False).agg(agg)
    X, y, g = inc[METRICS], inc["fault_type"], inc["is_gray"]
    return train_test_split(X, y, g, test_size=0.3, random_state=seed, stratify=y)


def compare(rows, mask, label, results, n_partitions):
    sel = [r for r, m in zip(rows, mask) if m]
    if len(sel) < 10:
        print(f"  {label:34s} n={len(sel):<5} (too few, skipped)")
        return
    x = np.array([r["cf"] for r in sel])
    y = np.array([r["res"] for r in sel])
    n01 = int(((x == 1) & (y == 0)).sum())
    n10 = int(((x == 0) & (y == 1)).sum())
    ex = (n01 + n10) < 25
    p = float(mcnemar([[int(((x == 1) & (y == 1)).sum()), n01],
                       [n10, int(((x == 0) & (y == 0)).sum())]],
                      exact=ex, correction=not ex).pvalue)
    delta = x.mean() - y.mean()
    verdict = ("ours WINS" if delta > 0 and p < 0.05 else
               "residual wins" if delta < 0 and p < 0.05 else "tie")
    print(f"  {label:34s} n={len(sel):<5} ours {x.mean():.3f}  residual {y.mean():.3f}  "
          f"delta {delta:+.3f}  p={p:<10.3g} {verdict}")
    results[label] = {"n": len(sel), "ours": float(x.mean()), "residual": float(y.mean()),
                      "delta": float(delta), "p": p, "verdict": verdict,
                      "bonferroni_p": min(1.0, p * n_partitions)}


def main():
    cag = DatacenterCAG()
    ndesc = {m: len(cag.descendants.get(m, set())) for m in METRICS}
    rows = []

    for k, seed in enumerate(SEEDS, 1):
        regenerate(seed)
        X_tr, X_te, y_tr, y_te, g_tr, g_te = load_with_gray(seed)
        sem = fit_sem(cag, X_tr[y_tr == 0])
        inv = {v: m for m, v in cag.root_to_fault.items()}

        for (_, row), yt, gray in zip(X_te.iterrows(), y_te.values, g_te.values):
            if yt == 0:
                continue
            truth = inv[yt]
            z = {m: zscore(sem, row, m) for m in METRICS}
            cand = [m for m in METRICS if z[m] >= TAU]
            if truth not in cand:
                cand.append(truth)
            cf = max(cand, key=lambda m: (z[m] + sem.simulate_intervention(row, m), z[m]))
            res = max(cand, key=lambda m: (residual_score(sem, cag, row, m), z[m]))
            zrank = sorted(cand, key=lambda m: -z[m]).index(truth) + 1
            rows.append({"cf": int(cf == truth), "res": int(res == truth),
                         "gray": int(gray), "ndesc": ndesc[truth],
                         "zrank": zrank, "ncand": len(cand)})
        if k % 5 == 0:
            print(f"  {k}/{len(SEEDS)} seeds, {len(rows)} incidents", flush=True)

    results = {}
    deep = np.median([r["ndesc"] for r in rows])
    partitions = [
        ("ALL incidents", [True] * len(rows)),
        ("H1 gray failure", [r["gray"] == 1 for r in rows]),
        ("H1 normal signal", [r["gray"] == 0 for r in rows]),
        ("H2 deep root (ndesc>%d)" % deep, [r["ndesc"] > deep for r in rows]),
        ("H2 shallow root", [r["ndesc"] <= deep for r in rows]),
        ("H3 buried root (z-rank>3)", [r["zrank"] > 3 for r in rows]),
        ("H3 obvious root (z-rank<=3)", [r["zrank"] <= 3 for r in rows]),
    ]
    print(f"\n{len(rows)} incidents, {len(partitions)} partitions "
          f"(Bonferroni factor {len(partitions)})\n")
    for label, mask in partitions:
        compare(rows, mask, label, results, len(partitions))

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"seeds": SEEDS, "n": len(rows), "partitions": results}, f, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
