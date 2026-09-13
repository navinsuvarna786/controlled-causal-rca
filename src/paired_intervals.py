#!/usr/bin/env python3
"""
Paired confidence intervals for every partitioned comparison in the paper.

A p-value says whether a difference is distinguishable from zero. It does not
say how large a difference the data could have hidden, which is exactly what a
reader needs for the paper's null findings. This script reports a bootstrap
percentile interval on the paired difference in Top-1 for each partition, so a
null can be stated as a bound rather than as an absence of evidence.

Incidents are resampled with replacement as whole units, which preserves the
pairing: both rules are always evaluated on the same incident.

Writes paired_intervals.json.
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
N_BOOT = 10_000
OUT_PATH = "paired_intervals.json"


def load_with_gray(seed):
    df = pd.read_csv("datacenter_dataset.csv")
    agg = {**{m: "median" for m in METRICS}, "fault_type": "first", "is_gray": "max"}
    inc = df.groupby("incident_id", as_index=False).agg(agg)
    return train_test_split(inc[METRICS], inc["fault_type"], inc["is_gray"],
                            test_size=0.3, random_state=seed,
                            stratify=inc["fault_type"])


def boot_ci(a, b, rng, n_boot=N_BOOT, alpha=0.05):
    """Percentile interval on mean(a) - mean(b), resampling incidents as pairs."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return float(np.percentile(diffs, 100 * alpha / 2)), \
           float(np.percentile(diffs, 100 * (1 - alpha / 2)))


def compare(rows, mask, label, rng, out):
    sel = [r for r, m in zip(rows, mask) if m]
    if len(sel) < 10:
        return
    a = np.array([r["cf"] for r in sel])
    b = np.array([r["res"] for r in sel])
    n01 = int(((a == 1) & (b == 0)).sum())
    n10 = int(((a == 0) & (b == 1)).sum())
    ex = (n01 + n10) < 25
    p = float(mcnemar([[int(((a == 1) & (b == 1)).sum()), n01],
                       [n10, int(((a == 0) & (b == 0)).sum())]],
                      exact=ex, correction=not ex).pvalue)
    lo, hi = boot_ci(a, b, rng)
    d = a.mean() - b.mean()
    print(f"  {label:32s} n={len(sel):<5} ours {a.mean():.3f}  res {b.mean():.3f}  "
          f"d={d:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  width {hi-lo:.3f}  p={p:.3g}")
    out[label] = {"n": len(sel), "ours": float(a.mean()), "residual": float(b.mean()),
                  "diff": float(d), "ci_low": lo, "ci_high": hi,
                  "ci_width": hi - lo, "p": p}


def main():
    rng = np.random.default_rng(20260227)
    cag = DatacenterCAG()
    ndesc = {m: len(cag.descendants.get(m, set())) for m in METRICS}
    rows = []

    for k, seed in enumerate(SEEDS, 1):
        regenerate(seed)
        X_tr, X_te, y_tr, y_te, g_tr, g_te = load_with_gray(seed)
        sem = fit_sem(cag, X_tr[y_tr == 0])
        inv = {v: m for m, v in cag.root_to_fault.items()}
        strong = {e for e, v in sem.edge_r2.items() if v > 0.5}

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
            outs = [(truth, c) for c in cag.dag.get(truth, [])]
            rows.append({
                "cf": int(cf == truth), "res": int(res == truth),
                "gray": int(gray), "ndesc": ndesc[truth],
                "zrank": sorted(cand, key=lambda m: -z[m]).index(truth) + 1,
                "strong": bool(outs) and all(e in strong for e in outs),
            })
        if k % 5 == 0:
            print(f"  {k}/{len(SEEDS)} seeds, {len(rows)} incidents", flush=True)

    out = {}
    deep = np.median([r["ndesc"] for r in rows])
    print(f"\n{len(rows)} incidents, {N_BOOT} bootstrap resamples\n")
    print("PRE-REGISTERED PARTITIONS")
    for label, mask in [
        ("All incidents", [True] * len(rows)),
        ("Gray failure", [r["gray"] == 1 for r in rows]),
        ("Normal signal", [r["gray"] == 0 for r in rows]),
        (f"Deep root (>{deep:.0f} desc)", [r["ndesc"] > deep for r in rows]),
        ("Shallow root", [r["ndesc"] <= deep for r in rows]),
        ("Buried root (rank>3)", [r["zrank"] > 3 for r in rows]),
        ("Prominent root (rank<=3)", [r["zrank"] <= 3 for r in rows]),
    ]:
        compare(rows, mask, label, rng, out)

    print("\nBY RANK OF TRUE ROOT")
    for lo_, hi_, lab in [(1, 1, "rank 1"), (2, 3, "rank 2-3"),
                          (4, 6, "rank 4-6"), (7, 99, "rank 7+")]:
        compare(rows, [lo_ <= r["zrank"] <= hi_ for r in rows], lab, rng, out)

    print("\nBY EDGE FIT OF TRUE ROOT")
    compare(rows, [r["strong"] for r in rows], "Root out-edges R2>0.5", rng, out)
    compare(rows, [not r["strong"] for r in rows], "Otherwise", rng, out)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"seeds": SEEDS, "n_boot": N_BOOT, "partitions": out}, f, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
