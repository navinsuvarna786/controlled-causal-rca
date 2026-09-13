#!/usr/bin/env python3
"""
Seed-clustered intervals, and held-out validation of the candidate-deviation
correction.

Two concerns motivate this script.

First, incidents are not independent. Every incident inside a seed is drawn
from the same generated dataset and scored with the same fitted baselines and
edge models, so incidents are clustered by seed. Resampling incidents, as an
ordinary bootstrap does, ignores that structure and produces intervals that
are too narrow. Here the resampling unit is the seed: seeds are drawn with
replacement and all of their incidents travel together, which is the standard
cluster bootstrap and respects the dependence.

Second, the candidate-deviation correction was motivated by inspecting failure
cases on the base seeds. Reporting its gain on the same data is circular, so
the gain is recomputed on the extension seeds alone, which played no part in
identifying the defect.

Writes cluster_intervals.json.
"""

import contextlib
import io
import json

import numpy as np
import pandas as pd

with contextlib.redirect_stdout(io.StringIO()):
    from datacenter_eval import DatacenterCAG, METRICS  # noqa: E402
from final_results import fit_sem, load, regenerate  # noqa: E402
from full_candidate_eval import residual_score, zscore  # noqa: E402

BASE_SEEDS = [20260227, 42, 7, 1234, 999]
HELDOUT_SEEDS = [20_000 + i for i in range(15)]
SEEDS = BASE_SEEDS + HELDOUT_SEEDS
TAU = 3.0
N_BOOT = 10_000
OUT_PATH = "cluster_intervals.json"


def cluster_ci(df, col_a, col_b, rng, n_boot=N_BOOT, alpha=0.05):
    """Percentile interval on mean(a)-mean(b), resampling SEEDS with replacement."""
    groups = [g for _, g in df.groupby("seed")]
    k = len(groups)
    a_by = [g[col_a].to_numpy(float) for g in groups]
    b_by = [g[col_b].to_numpy(float) for g in groups]
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, k, k)
        a = np.concatenate([a_by[j] for j in pick])
        b = np.concatenate([b_by[j] for j in pick])
        diffs[i] = a.mean() - b.mean()
    return (float(np.percentile(diffs, 100 * alpha / 2)),
            float(np.percentile(diffs, 100 * (1 - alpha / 2))))


def naive_ci(df, col_a, col_b, rng, n_boot=N_BOOT, alpha=0.05):
    """Ordinary incident-level bootstrap, for comparison."""
    a, b = df[col_a].to_numpy(float), df[col_b].to_numpy(float)
    n = len(a)
    idx = rng.integers(0, n, size=(n_boot, n))
    d = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return (float(np.percentile(d, 100 * alpha / 2)),
            float(np.percentile(d, 100 * (1 - alpha / 2))))


def main():
    rng = np.random.default_rng(20260227)
    cag = DatacenterCAG()
    rows = []

    for k, seed in enumerate(SEEDS, 1):
        regenerate(seed)
        _, X_te, _, y_te, X_h = load(seed)
        gray_map = (pd.read_csv("datacenter_dataset.csv")
                    .groupby("incident_id")["is_gray"].max())
        sem = fit_sem(cag, X_h)
        inv = {v: m for m, v in cag.root_to_fault.items()}
        strong = {e for e, v in sem.edge_r2.items() if v > 0.5}

        for (idx_i, row), yt in zip(X_te.iterrows(), y_te.values):
            if yt == 0:
                continue
            truth = inv[yt]
            z = {m: zscore(sem, row, m) for m in METRICS}
            cand = [m for m in METRICS if z[m] >= TAU]
            oracle_added = truth not in cand
            if oracle_added:
                cand.append(truth)
            cf_raw = max(cand, key=lambda m: (sem.simulate_intervention(row, m), z[m]))
            cf_z = max(cand, key=lambda m: (z[m] + sem.simulate_intervention(row, m), z[m]))
            res = max(cand, key=lambda m: (residual_score(sem, cag, row, m), z[m]))
            outs = [(truth, c) for c in cag.dag.get(truth, [])]
            rows.append({
                "seed": seed, "heldout": seed in HELDOUT_SEEDS,
                "oracle_added": int(oracle_added),
                "cf": int(cf_z == truth), "cf_raw": int(cf_raw == truth),
                "res": int(res == truth),
                "anom": int(max(cand, key=lambda m: z[m]) == truth),
                "zrank": sorted(cand, key=lambda m: -z[m]).index(truth) + 1,
                "strong": bool(outs) and all(e in strong for e in outs),
                "ndesc": len(cag.descendants.get(truth, set())),
                "gray": int(gray_map.get(idx_i, 0)),
            })
        if k % 5 == 0:
            print(f"  {k}/{len(SEEDS)} seeds, {len(rows)} incidents", flush=True)

    df = pd.DataFrame(rows)
    out = {"n": len(df), "n_seeds": len(SEEDS)}

    print(f"\n{len(df)} incidents over {len(SEEDS)} seeds, {N_BOOT} resamples\n")
    print("CLUSTER (seed-level) vs NAIVE (incident-level) intervals\n")
    print(f"  {'comparison':34s} {'diff':>8}  {'cluster 95% CI':>22} {'w':>6}   "
          f"{'naive 95% CI':>22} {'w':>6}")

    def line(sub, a, b, label, store):
        d = sub[a].mean() - sub[b].mean()
        clo, chi = cluster_ci(sub, a, b, rng)
        nlo, nhi = naive_ci(sub, a, b, rng)
        print(f"  {label:34s} {d:+8.3f}  [{clo:+.3f}, {chi:+.3f}] {chi-clo:6.3f}   "
              f"[{nlo:+.3f}, {nhi:+.3f}] {nhi-nlo:6.3f}")
        store[label] = {"n": len(sub), "diff": float(d),
                        "cluster_ci": [clo, chi], "cluster_width": chi - clo,
                        "naive_ci": [nlo, nhi], "naive_width": nhi - nlo}

    out["comparisons"] = {}
    line(df, "cf", "res", "ours vs residual (all)", out["comparisons"])
    line(df, "res", "anom", "residual vs anomaly", out["comparisons"])
    line(df, "cf", "anom", "ours vs anomaly", out["comparisons"])
    line(df[df.zrank > 3], "cf", "res", "ours vs residual | buried root", out["comparisons"])
    line(df[df.zrank > 6], "cf", "res", "ours vs residual | deeply buried", out["comparisons"])
    line(df[df.strong], "cf", "res", "ours vs residual | well-fit edges", out["comparisons"])
    line(df[~df.strong], "cf", "res", "ours vs residual | poorly-fit edges", out["comparisons"])
    line(df[df.zrank <= 3], "cf", "res", "ours vs residual | prominent root", out["comparisons"])
    line(df[df.zrank == 1], "cf", "res", "ours vs residual | rank 1", out["comparisons"])
    line(df[(df.zrank >= 2) & (df.zrank <= 3)], "cf", "res", "ours vs residual | rank 2-3", out["comparisons"])
    line(df[(df.zrank >= 4) & (df.zrank <= 6)], "cf", "res", "ours vs residual | rank 4-6", out["comparisons"])
    med = df.ndesc.median()
    line(df[df.ndesc > med], "cf", "res", "ours vs residual | deep root", out["comparisons"])
    line(df[df.ndesc <= med], "cf", "res", "ours vs residual | shallow root", out["comparisons"])
    line(df[df.gray == 1], "cf", "res", "ours vs residual | gray failure", out["comparisons"])
    line(df[df.gray == 0], "cf", "res", "ours vs residual | normal signal", out["comparisons"])

    print("\nCANDIDATE-DEVIATION CORRECTION: same-corpus vs held-out seeds\n")
    out["correction"] = {}
    for lab, sub in (("base seeds (motivating)", df[~df.heldout]),
                     ("held-out seeds", df[df.heldout]),
                     ("all seeds", df)):
        d = sub["cf"].mean() - sub["cf_raw"].mean()
        clo, chi = cluster_ci(sub, "cf", "cf_raw", rng)
        print(f"  {lab:26s} n={len(sub):<5} gain {d:+.4f}  cluster 95% CI "
              f"[{clo:+.4f}, {chi:+.4f}]")
        out["correction"][lab] = {"n": len(sub), "gain": float(d),
                                  "cluster_ci": [clo, chi]}

    print("\nORACLE-ASSISTED INCIDENTS (root added below threshold)\n")
    frac = df.oracle_added.mean()
    print(f"  share of incidents: {frac:.4f}")
    out["oracle"] = {"share": float(frac)}
    for rule in ("cf", "res", "anom"):
        with_o = df[rule].mean()
        without = df[df.oracle_added == 0][rule].mean()
        print(f"  {rule:5s} Top-1 with oracle assist {with_o:.4f}   "
              f"on non-assisted incidents only {without:.4f}")
        out["oracle"][rule] = {"with": float(with_o), "non_assisted": float(without)}

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
