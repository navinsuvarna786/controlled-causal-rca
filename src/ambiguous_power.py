#!/usr/bin/env python3
"""
Adequately powered test of the paper's central claim.

Counterfactual refinement fires only on incidents whose two leading candidates
are within the margin, which is ~7% of fault incidents. At five seeds that is
about 15 incidents and 9 discordant pairs -- far too few to test. This script
raises the seed count so the gated subset is large enough for a paired test.

Compares the identical fitted model with refinement on versus off, so the only
difference is whether the candidate is chosen by counterfactual restoration or
by anomaly magnitude. Reports exact McNemar on the gated subset.

Writes ambiguous_power.json.
"""

import contextlib
import io
import json
import sys

import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

with contextlib.redirect_stdout(io.StringIO()):
    from datacenter_eval import DatacenterCAG  # noqa: E402
from final_results import fit_sem, load, rank_sem, regenerate  # noqa: E402

N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
BASE_SEEDS = [20260227, 42, 7, 1234, 999]
SEEDS = BASE_SEEDS + [10_000 + i for i in range(max(0, N_SEEDS - len(BASE_SEEDS)))]
MARGIN = 0.25
OUT_PATH = "ambiguous_power.json"


def main():
    on_hits, off_hits, gated_on, gated_off = [], [], [], []

    for k, seed in enumerate(SEEDS, 1):
        regenerate(seed)
        _, X_test, _, y_test, X_healthy = load(seed)
        cag = DatacenterCAG()
        sem = fit_sem(cag, X_healthy)
        r2f = cag.root_to_fault

        rank_on = rank_sem(sem, cag, X_test, True)
        rank_off = rank_sem(sem, cag, X_test, False)

        for (_, row), yt, a, b in zip(X_test.iterrows(), y_test.values, rank_on, rank_off):
            if yt == 0:
                continue
            hit_on = int(r2f[a[0]] == yt)
            hit_off = int(r2f[b[0]] == yt)
            on_hits.append(hit_on)
            off_hits.append(hit_off)
            s = sem.root_anomaly_scores(row)
            order = sorted(s.values(), reverse=True)
            if (order[0] - order[1]) <= MARGIN:
                gated_on.append(hit_on)
                gated_off.append(hit_off)
        if k % 5 == 0:
            print(f"  {k}/{len(SEEDS)} seeds, gated so far: {len(gated_on)}", flush=True)

    def paired(a, b, label):
        a, b = np.array(a), np.array(b)
        n01 = int(((a == 1) & (b == 0)).sum())
        n10 = int(((a == 0) & (b == 1)).sum())
        table = [[int(((a == 1) & (b == 1)).sum()), n01],
                 [n10, int(((a == 0) & (b == 0)).sum())]]
        # Exact binomial: the discordant counts stay small even when pooled.
        res = mcnemar(table, exact=True)
        print(f"\n{label}  (n={len(a)})")
        print(f"  refinement on : {a.mean():.4f}")
        print(f"  refinement off: {b.mean():.4f}   delta {a.mean()-b.mean():+.4f}")
        print(f"  discordant {n01}/{n10}   exact McNemar p = {res.pvalue:.4g}"
              f"   -> {'SIGNIFICANT' if res.pvalue < 0.05 else 'not significant'}")
        return {"n": len(a), "on": float(a.mean()), "off": float(b.mean()),
                "n01": n01, "n10": n10, "p": float(res.pvalue)}

    out = {"seeds": SEEDS,
           "all_fault": paired(on_hits, off_hits, "ALL FAULT INCIDENTS"),
           "gated": paired(gated_on, gated_off, "GATED (ambiguous) SUBSET")}
    out["gate_rate"] = len(gated_on) / len(on_hits)
    print(f"\ngate fires on {out['gate_rate']*100:.1f}% of fault incidents")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
