#!/usr/bin/env python3
"""
Significance tests for every comparison the paper makes.

Section V-B promises paired McNemar tests; this script supplies them. All tests
are paired at the incident level and pooled across seeds, which is far better
powered than comparing five per-seed means.

Covers: main-table pairwise comparisons, the concurrent-fault coverage gap, and
robustness degradation under each stressor.

Writes significance_all.json.
"""

import contextlib
import io
import itertools
import json

import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

with contextlib.redirect_stdout(io.StringIO()):
    from datacenter_eval import (  # noqa: E402
        DatacenterCAG, METRICS, PageRankBaseline, PearsonBaseline,
    )
import baselines_causal as bc  # noqa: E402
from final_results import (  # noqa: E402
    fit_sem, load, rank_causal_baseline, rank_sem, regenerate,
)
from multifault_faultonly import load as load_mf, regenerate_multifault  # noqa: E402
from robustness_faultonly import perturb_dag  # noqa: E402
from datacenter_eval import CausalInterventionRCA  # noqa: E402

SEEDS = [20260227, 42, 7, 1234, 999]
MASK_RATE, NOISE_STD, EDGE_REMOVAL = 0.20, 0.06, 0.20
OUT_PATH = "significance_all.json"


def test(a, b, label, results):
    a, b = np.array(a), np.array(b)
    n01 = int(((a == 1) & (b == 0)).sum())
    n10 = int(((a == 0) & (b == 1)).sum())
    tbl = [[int(((a == 1) & (b == 1)).sum()), n01],
           [n10, int(((a == 0) & (b == 0)).sum())]]
    exact = (n01 + n10) < 25
    p = float(mcnemar(tbl, exact=exact, correction=not exact).pvalue)
    sig = "SIG" if p < 0.05 else "n.s."
    print(f"  {label:44s} {a.mean():.3f} vs {b.mean():.3f}  "
          f"{n01:>4}/{n10:<4} p={p:<10.3g} {sig}")
    results[label] = {"a": float(a.mean()), "b": float(b.mean()),
                      "n": len(a), "n01": n01, "n10": n10, "p": p,
                      "significant": p < 0.05}


def main():
    results = {}

    # ---- main table, all pairs -------------------------------------------
    hits = {k: [] for k in ("ours", "ablation", "rr_rca", "ks_rca",
                            "pearson", "pagerank")}
    for seed in SEEDS:
        regenerate(seed)
        Xtr, Xte, ytr, yte, Xh = load(seed)
        cag = DatacenterCAG(); r2f = cag.root_to_fault
        sem = fit_sem(cag, Xh)

        def add(key, ranked, y, mapping):
            for yt, o in zip(y, ranked):
                if yt != 0:
                    hits[key].append(int(mapping[o[0]] == yt))

        add("ours", rank_sem(sem, cag, Xte, True), yte.values, r2f)
        add("ablation", rank_sem(sem, cag, Xte, False), yte.values, r2f)
        (_, bXte, _, byte, _, _, bXh, bbase, bmod, br2, bmet) = bc.fit_artifacts(seed)
        art = (bXte, bXh, bbase, bmod, br2, bmet)
        for nm in ("rr_rca", "ks_rca"):
            add(nm, rank_causal_baseline(nm, bXte.iterrows(), art),
                byte.values, bc.ROOT_TO_FAULT)
        p = PearsonBaseline(r2f); p.fit(Xtr, ytr)
        add("pearson", p.predict_ranked(Xte), yte.values, r2f)
        g = PageRankBaseline(r2f); g.fit(Xtr)
        add("pagerank", g.predict_ranked(Xte), yte.values, r2f)

    print(f"\n=== Main table, all pairs (n={len(hits['ours'])} fault incidents) ===")
    for x, y in itertools.combinations(hits, 2):
        test(hits[x], hits[y], f"{x} vs {y}", results)

    # ---- concurrent faults: coverage gap ---------------------------------
    ours_cov, ks_cov = [], []
    for seed in SEEDS:
        regenerate_multifault(seed)
        X_tr, X_te, y_tr, y_te, y2_tr, y2_te = load_mf(seed)
        cag = DatacenterCAG(); sem = fit_sem(cag, X_tr[y_tr == 0])
        for yt, y2t, o in zip(y_te.values, y2_te.values,
                              rank_sem(sem, cag, X_te, True)):
            if yt == 0 or y2t == 0:
                continue
            ours_cov.append(int(cag.root_to_fault[o[0]] in {yt, y2t}))
        (_, bXte, _, byte, _, by2te, bXh, bbase, bmod, br2, bmet) = bc.fit_artifacts(seed)
        pm = bc.parents_of()
        ks = bc.precompute_ks(bXh, bXte, bmet)
        for i, ((_, row), yt, y2t) in enumerate(zip(bXte.iterrows(),
                                                    byte.values, by2te.values)):
            if yt == 0 or y2t == 0:
                continue
            sc = bc.score_ks_rca(row, ks[i], br2, pm)
            top = max(bc.ROOT_METRICS, key=lambda r: sc[r])
            ks_cov.append(int(bc.ROOT_TO_FAULT[top] in {yt, y2t}))

    print(f"\n=== Concurrent faults, Top-1@coverage (n={len(ours_cov)} multi-fault) ===")
    n = min(len(ours_cov), len(ks_cov))
    test(ours_cov[:n], ks_cov[:n], "ours vs ks_rca (multi-fault coverage)", results)

    # ---- robustness: clean vs each stressor ------------------------------
    cond = {k: [] for k in ("clean", "missing", "noise", "dag")}
    for seed in SEEDS:
        regenerate(seed)
        _, X_te, _, y_te, Xh = load(seed)
        cag = DatacenterCAG(); sem = fit_sem(cag, Xh); r2f = cag.root_to_fault
        rng = np.random.default_rng(seed)

        def add(key, ranked, cg=None):
            m = (cg or cag).root_to_fault
            for yt, o in zip(y_te.values, ranked):
                if yt != 0:
                    cond[key].append(int(m[o[0]] == yt))

        add("clean", rank_sem(sem, cag, X_te, True))
        masked = X_te.copy(deep=True)
        drop = rng.random(masked[METRICS].shape) < MASK_RATE
        vals = np.array(masked[METRICS].to_numpy(dtype=float), copy=True)
        means = np.array([sem.baseline[m]["mean"] for m in METRICS])
        vals[drop] = np.broadcast_to(means, vals.shape)[drop]
        masked[METRICS] = vals
        add("missing", rank_sem(sem, cag, masked, True))
        noisy = X_te.copy(deep=True)
        noisy[METRICS] = noisy[METRICS] + rng.normal(0, NOISE_STD, noisy[METRICS].shape)
        add("noise", rank_sem(sem, cag, noisy, True))
        pert = perturb_dag(cag, sem, EDGE_REMOVAL)
        sp = CausalInterventionRCA(pert, use_rf=False); sp.fit(Xh)
        sp.edge_r2 = {e: v for e, v in sem.edge_r2.items()
                      if e[1] in pert.dag.get(e[0], [])}
        add("dag", rank_sem(sp, pert, X_te, True), pert)

    print(f"\n=== Robustness, clean vs stressor (n={len(cond['clean'])}) ===")
    for k in ("missing", "noise", "dag"):
        test(cond["clean"], cond[k], f"clean vs {k}", results)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
