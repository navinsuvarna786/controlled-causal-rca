#!/usr/bin/env python3
"""
Root-cause ranking over the FULL candidate set defined in the paper's Eq. (2):
every metric that becomes anomalous during the incident, not just the handful
of metrics where the generator injects faults.

This distinction turns out to decide the whole comparison. When candidates are
pre-restricted to the known root metrics, downstream symptoms never compete
with their own causes, so anomaly magnitude alone is sufficient and no causal
scoring rule can show a benefit. On the full candidate set the ranker must
reject consequences in favour of causes, which is the task causal reasoning
exists for.

Methods, all over the same DAG, same healthy-only training, same candidates:
  counterfactual : rank by predicted downstream anomaly reduction under
                   do(X_r <- mu_r)
  anomaly        : rank by |z| at the candidate
  residual       : CIRCA-style, z(r) plus weighted parent-conditioned residuals
                   over descendants of r

Correctness is exact identification of the injected root metric.
Writes full_candidate_eval.json.
"""

import contextlib
import io
import json
import sys

import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

with contextlib.redirect_stdout(io.StringIO()):
    from datacenter_eval import DatacenterCAG, METRICS  # noqa: E402
from final_results import fit_sem, load, regenerate  # noqa: E402

N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
BASE = [20260227, 42, 7, 1234, 999]
SEEDS = BASE + [20_000 + i for i in range(max(0, N_SEEDS - len(BASE)))]
TAU = 3.0
OUT_PATH = "full_candidate_eval.json"


def zscore(sem, row, m):
    b = sem.baseline[m]
    return abs(float(row[m]) - b["mean"]) / b["std"]


def residual_score(sem, cag, row, cand):
    """CIRCA-style: candidate deviation plus weighted parent-conditioned
    residuals across its descendants. No do() restoration."""
    score = zscore(sem, row, cand)
    for d in cag.descendants.get(cand, set()):
        parents = cag.parents.get(d, [])
        if not parents:
            continue
        w = max((sem.edge_r2.get((p, d), 0.0) for p in parents), default=0.0)
        if w < 0.05:
            continue
        preds = [sem.models[(p, d)].predict([[float(row[p])]])[0]
                 for p in parents if (p, d) in sem.models]
        if not preds:
            continue
        resid = abs(float(row[d]) - float(np.mean(preds))) / sem.baseline[d]["std"]
        score += w * resid
    return score


def main():
    hits = {k: [] for k in ("counterfactual", "counterfactual_z", "anomaly", "residual")}
    sizes, missed_truth = [], 0

    for k, seed in enumerate(SEEDS, 1):
        regenerate(seed)
        _, X_test, _, y_test, X_healthy = load(seed)
        cag = DatacenterCAG()
        sem = fit_sem(cag, X_healthy)
        inv = {v: m for m, v in cag.root_to_fault.items()}

        for (_, row), yt in zip(X_test.iterrows(), y_test.values):
            if yt == 0:
                continue
            truth = inv[yt]
            z = {m: zscore(sem, row, m) for m in METRICS}
            cand = [m for m in METRICS if z[m] >= TAU]
            if truth not in cand:
                cand.append(truth)
                missed_truth += 1
            sizes.append(len(cand))

            hits["anomaly"].append(int(max(cand, key=lambda m: z[m]) == truth))
            hits["counterfactual"].append(int(
                max(cand, key=lambda m: (sem.simulate_intervention(row, m), z[m])) == truth))
            hits["residual"].append(int(
                max(cand, key=lambda m: (residual_score(sem, cag, row, m), z[m])) == truth))
            # Same information as the residual scorer: own deviation + downstream evidence.
            hits["counterfactual_z"].append(int(
                max(cand, key=lambda m: (z[m] + sem.simulate_intervention(row, m), z[m])) == truth))
        if k % 5 == 0:
            print(f"  {k}/{len(SEEDS)} seeds, {len(sizes)} incidents", flush=True)

    n = len(sizes)
    print(f"\nincidents: {n}   mean candidate-set size: {np.mean(sizes):.1f}"
          f"   true root not anomalous in {missed_truth} ({100*missed_truth/n:.1f}%)\n")
    for key, v in hits.items():
        print(f"  {key:15s} Top-1 = {np.mean(v):.4f}")

    out = {"seeds": SEEDS, "n": n, "mean_candidates": float(np.mean(sizes)),
           "truth_not_anomalous": missed_truth,
           "top1": {k: float(np.mean(v)) for k, v in hits.items()}, "tests": {}}

    print()
    for a, b in (("counterfactual", "anomaly"),
                 ("counterfactual_z", "anomaly"),
                 ("counterfactual_z", "counterfactual"),
                 ("counterfactual_z", "residual"),
                 ("residual", "anomaly")):
        x, y = np.array(hits[a]), np.array(hits[b])
        n01 = int(((x == 1) & (y == 0)).sum())
        n10 = int(((x == 0) & (y == 1)).sum())
        ex = (n01 + n10) < 25
        p = float(mcnemar([[int(((x == 1) & (y == 1)).sum()), n01],
                           [n10, int(((x == 0) & (y == 0)).sum())]],
                          exact=ex, correction=not ex).pvalue)
        print(f"  {a:15s} vs {b:15s} {x.mean():.4f} vs {y.mean():.4f}  "
              f"{n01:>4}/{n10:<4} p={p:<11.4g} {'SIG' if p < 0.05 else 'n.s.'}")
        out["tests"][f"{a}_vs_{b}"] = {"a": float(x.mean()), "b": float(y.mean()),
                                       "n01": n01, "n10": n10, "p": p}

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
