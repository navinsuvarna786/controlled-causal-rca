#!/usr/bin/env python3
"""
Direct measurement of estimator variance for the two leading scoring rules.

The paper explains its headline result by saying that counterfactual
restoration is a higher-variance estimate of the same quantity that a one-hop
residual captures directly. That claim was asserted rather than measured, and
the edge-fit interaction only shows a pattern consistent with it.

This script measures it. The healthy training set is resampled with
replacement, the baselines and per-edge models are refitted on each resample,
and the test incidents are held fixed. Any change in a rule's top-1 pick is
then attributable to the fitted equations alone, which is exactly the
estimator variance the argument appeals to.

Two statistics per rule:
  flip rate  -- fraction of (incident, resample) pairs whose top-1 differs
                from the pick obtained on the full healthy set
  margin SD  -- standard deviation across resamples of the normalized gap
                between the best and second-best candidate, averaged over
                incidents; a stable scorer separates candidates consistently

Writes estimator_variance.json.
"""

import contextlib
import io
import json
import sys

import numpy as np
import pandas as pd

with contextlib.redirect_stdout(io.StringIO()):
    from datacenter_eval import DatacenterCAG, METRICS  # noqa: E402
from final_results import fit_sem, load, regenerate  # noqa: E402
from full_candidate_eval import residual_score, zscore  # noqa: E402

SEEDS = [20260227, 42, 7, 1234, 999]
N_BOOT = 20
TAU = 3.0
OUT_PATH = "estimator_variance.json"


def scores(sem, cag, row, cand):
    """Both rules on one incident, returning score dicts over the candidates."""
    z = {m: zscore(sem, row, m) for m in METRICS}
    cf = {m: z[m] + sem.simulate_intervention(row, m) for m in cand}
    rs = {m: residual_score(sem, cag, row, m) for m in cand}
    return cf, rs


def margin(d):
    """Gap between best and second-best, normalized by the best score."""
    v = sorted(d.values(), reverse=True)
    if len(v) < 2 or abs(v[0]) < 1e-12:
        return 0.0
    return (v[0] - v[1]) / abs(v[0])


def main():
    cag = DatacenterCAG()
    flips = {"cf": [], "res": []}
    margin_sds = {"cf": [], "res": []}

    for si, seed in enumerate(SEEDS, 1):
        regenerate(seed)
        X_tr, X_te, y_tr, y_te, X_healthy = load(seed)
        rng = np.random.default_rng(seed)

        # Reference fit on the full healthy set.
        sem_full = fit_sem(cag, X_healthy)
        ref, cands = {}, {}
        for i, ((_, row), yt) in enumerate(zip(X_te.iterrows(), y_te.values)):
            if yt == 0:
                continue
            z = {m: zscore(sem_full, row, m) for m in METRICS}
            c = [m for m in METRICS if z[m] >= TAU] or [max(METRICS, key=lambda m: z[m])]
            cands[i] = c
            cf, rs = scores(sem_full, cag, row, c)
            ref[i] = (max(cf, key=cf.get), max(rs, key=rs.get))

        # Refit on bootstrap resamples of the healthy training data.
        per_inc = {i: {"cf": [], "res": []} for i in ref}
        for b in range(N_BOOT):
            idx = rng.integers(0, len(X_healthy), len(X_healthy))
            sem_b = fit_sem(cag, X_healthy.iloc[idx])
            for i, ((_, row), yt) in enumerate(zip(X_te.iterrows(), y_te.values)):
                if i not in ref:
                    continue
                cf, rs = scores(sem_b, cag, row, cands[i])
                flips["cf"].append(int(max(cf, key=cf.get) != ref[i][0]))
                flips["res"].append(int(max(rs, key=rs.get) != ref[i][1]))
                per_inc[i]["cf"].append(margin(cf))
                per_inc[i]["res"].append(margin(rs))
        for i in per_inc:
            margin_sds["cf"].append(float(np.std(per_inc[i]["cf"])))
            margin_sds["res"].append(float(np.std(per_inc[i]["res"])))
        print(f"  seed {si}/{len(SEEDS)} done", flush=True)

    out = {}
    print(f"\n{N_BOOT} refits per seed, {len(SEEDS)} seeds, "
          f"{len(flips['cf'])} (incident, refit) pairs\n")
    for key, lab in (("cf", "Counterfactual"), ("res", "Residual")):
        fr = float(np.mean(flips[key]))
        ms = float(np.mean(margin_sds[key]))
        out[key] = {"flip_rate": fr, "margin_sd": ms}
        print(f"  {lab:16s} top-1 flip rate {fr:.4f}   mean margin SD {ms:.4f}")
    ratio_f = out["cf"]["flip_rate"] / max(out["res"]["flip_rate"], 1e-9)
    ratio_m = out["cf"]["margin_sd"] / max(out["res"]["margin_sd"], 1e-9)
    out["flip_ratio"] = ratio_f
    out["margin_ratio"] = ratio_m
    print(f"\n  counterfactual / residual: flip {ratio_f:.2f}x   margin SD {ratio_m:.2f}x")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"seeds": SEEDS, "n_boot": N_BOOT, "results": out}, f, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
