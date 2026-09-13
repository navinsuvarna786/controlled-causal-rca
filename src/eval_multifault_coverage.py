"""
Multi-fault evaluation: Top-1@primary vs Top-1@coverage.

Top-1@primary    — predicted fault matches `fault_type` (single-fault definition)
Top-1@coverage   — predicted fault matches EITHER `fault_type` OR
                   `fault_type_secondary`. Standard multi-root-cause metric.
Top-2@coverage   — top-2 predictions intersect {primary, secondary}.

Uses V2 counterfactual scoring.

Run from a working dir containing datacenter_dataset.csv (multi-fault) and
the supporting CSV/JSON artifacts. Writes
`results_multifault_coverage.json`.
"""

import json
import math
import pathlib

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from dag import (
    DAG, ROOT_TO_FAULT, FAULT_TO_ROOT, ROOT_METRICS,
    all_metrics, descendants, parents_of,
)

GLOBAL_SEED = 20260227
np.random.seed(GLOBAL_SEED)

DATA_PATH = "datacenter_dataset.csv"


def evaluate_seed(seed, use_rf=False):
    df = pd.read_csv(DATA_PATH)
    metrics = [m for m in all_metrics() if m in df.columns]

    has_secondary = 'fault_type_secondary' in df.columns
    agg = {**{m: 'median' for m in metrics}, 'fault_type': 'first'}
    if has_secondary:
        agg['fault_type_secondary'] = 'first'
    incident_df = df.groupby('incident_id', as_index=False).agg(agg)

    X = incident_df[metrics]
    y = incident_df['fault_type']
    y2 = incident_df['fault_type_secondary'] if has_secondary else pd.Series(
        np.zeros(len(incident_df), dtype=int)
    )

    X_train, X_test, y_train, y_test, y2_train, y2_test = train_test_split(
        X, y, y2, test_size=0.3, stratify=y, random_state=seed,
    )

    X_healthy = X_train[y_train == 0]

    baseline = {
        m: {
            'mean': float(X_healthy[m].mean()),
            'std': float(X_healthy[m].std() + 1e-9),
        }
        for m in metrics
    }

    edge_models = {}
    edge_r2 = {}
    for parent, kids in DAG.items():
        for child in kids:
            if parent not in metrics or child not in metrics:
                continue
            xs = X_healthy[[parent]].values
            ys = X_healthy[child].values
            if use_rf:
                m_ = RandomForestRegressor(
                    n_estimators=80, max_depth=6, random_state=seed
                )
            else:
                m_ = LinearRegression()
            m_.fit(xs, ys)
            edge_models[(parent, child)] = m_
            edge_r2[(parent, child)] = float(m_.score(xs, ys))

    parents_map = parents_of()

    def root_z(row):
        return {
            r: abs(row[r] - baseline[r]['mean']) / baseline[r]['std']
            for r in ROOT_METRICS
        }

    def propagate_baseline(row, root):
        pred = {root: baseline[root]['mean']}
        frontier = [root]
        while frontier:
            par = frontier.pop(0)
            for child in DAG.get(par, []):
                model = edge_models.get((par, child))
                if model is None or par not in pred:
                    continue
                est = float(model.predict([[pred[par]]])[0])
                pred[child] = (
                    0.5 * (pred[child] + est) if child in pred else est
                )
                if child not in frontier:
                    frontier.append(child)
        return pred

    def counterfactual_score(row, candidate):
        pred = propagate_baseline(row, candidate)
        descs = list(descendants(candidate))
        if not descs:
            return abs(
                row[candidate] - baseline[candidate]['mean']
            ) / baseline[candidate]['std']
        improvement = 0.0
        for d in descs:
            par_r2 = [edge_r2.get((p, d), 0.0) for p in parents_map.get(d, [])]
            if not par_r2:
                continue
            w = max(par_r2)
            if w < 0.05:
                continue
            obs = float(row[d])
            std = baseline[d]['std']
            mean_d = baseline[d]['mean']
            est = pred.get(d, obs)
            anomaly_obs = abs(obs - mean_d) / std
            anomaly_post = abs(est - mean_d) / std
            improvement += w * max(0.0, anomaly_obs - anomaly_post)
        return improvement

    healthy_threshold = 1.5
    margin = 0.25
    n_test = len(y_test)

    rankings = []  # list of [metric, metric, ...] ranked best -> worst
    for _, row in X_test.iterrows():
        zs = root_z(row)
        if max(zs.values()) < healthy_threshold:
            rankings.append([])  # healthy
            continue
        ranked = sorted(ROOT_METRICS, key=lambda r: zs[r], reverse=True)
        top1, top2 = ranked[0], ranked[1]
        if (zs[top1] - zs[top2]) > margin:
            best = top1
        else:
            s1 = counterfactual_score(row, top1)
            s2 = counterfactual_score(row, top2)
            best = top1 if s1 >= s2 else top2
        if best == top1:
            rankings.append([top1, top2] + [r for r in ranked[2:]])
        else:
            rankings.append([top2, top1] + [r for r in ranked[2:]])

    y_test_arr = y_test.values
    y2_test_arr = y2_test.values

    top1_primary = 0
    top1_coverage = 0
    top2_coverage = 0
    multi_fault_count = 0
    multi_top1 = 0
    multi_top1_coverage = 0
    multi_top2 = 0
    healthy_count = 0
    correct_healthy = 0

    for i, ranked in enumerate(rankings):
        true_p = int(y_test_arr[i])
        true_s = int(y2_test_arr[i])

        if true_p == 0:
            healthy_count += 1
            if not ranked:
                correct_healthy += 1
            continue

        if not ranked:
            # Predicted healthy but not healthy (false negative)
            continue

        pred1 = ROOT_TO_FAULT[ranked[0]]
        pred_top2 = {ROOT_TO_FAULT[ranked[0]], ROOT_TO_FAULT[ranked[1]]}

        if pred1 == true_p:
            top1_primary += 1

        true_set = {true_p}
        if true_s != 0:
            true_set.add(true_s)
            multi_fault_count += 1
            if pred1 == true_p:
                multi_top1 += 1
            if pred1 in true_set:
                multi_top1_coverage += 1
            if pred_top2 & true_set:
                multi_top2 += 1

        if pred1 in true_set:
            top1_coverage += 1
        if pred_top2 & true_set:
            top2_coverage += 1

    fault_count = sum(1 for r in rankings if r and y_test_arr[rankings.index(r) if rankings.index(r) < n_test else 0])
    nonhealthy = sum(1 for v in y_test_arr if v != 0)

    return {
        'seed': seed,
        'use_rf': use_rf,
        'n_test': n_test,
        'n_nonhealthy': nonhealthy,
        'n_multi_fault': multi_fault_count,
        'top1_primary': top1_primary / nonhealthy if nonhealthy else 0.0,
        'top1_coverage': top1_coverage / nonhealthy if nonhealthy else 0.0,
        'top2_coverage': top2_coverage / nonhealthy if nonhealthy else 0.0,
        'multi_only_top1_primary': multi_top1 / multi_fault_count if multi_fault_count else float('nan'),
        'multi_only_top1_coverage': multi_top1_coverage / multi_fault_count if multi_fault_count else float('nan'),
        'multi_only_top2_coverage': multi_top2 / multi_fault_count if multi_fault_count else float('nan'),
    }


def main():
    seeds = [20260227, 42, 7, 1234, 999]
    print(f"{'seed':>10} {'model':>6} {'T1@p':>7} {'T1@cov':>8} {'T2@cov':>8} "
          f"{'mf-T1@p':>9} {'mf-T1@cov':>11} {'mf-T2@cov':>11} {'n_mf':>6}")
    print("-" * 95)

    all_results = {'linear': [], 'rf': []}
    for seed in seeds:
        for use_rf, label in [(False, 'lin'), (True, 'rf')]:
            r = evaluate_seed(seed, use_rf=use_rf)
            all_results['rf' if use_rf else 'linear'].append(r)
            mfp = r['multi_only_top1_primary']
            mfc = r['multi_only_top1_coverage']
            mft2 = r['multi_only_top2_coverage']
            mfp_s = f"{mfp:.4f}" if not math.isnan(mfp) else "n/a"
            mfc_s = f"{mfc:.4f}" if not math.isnan(mfc) else "n/a"
            mft2_s = f"{mft2:.4f}" if not math.isnan(mft2) else "n/a"
            print(f"{seed:>10} {label:>6} {r['top1_primary']:>7.4f} "
                  f"{r['top1_coverage']:>8.4f} {r['top2_coverage']:>8.4f} "
                  f"{mfp_s:>9} {mfc_s:>11} {mft2_s:>11} {r['n_multi_fault']:>6d}")

    print()
    summary = {}
    for kind, runs in all_results.items():
        for key in [
            'top1_primary', 'top1_coverage', 'top2_coverage',
            'multi_only_top1_primary', 'multi_only_top1_coverage',
            'multi_only_top2_coverage',
        ]:
            vals = [r[key] for r in runs if not (isinstance(r[key], float) and math.isnan(r[key]))]
            if vals:
                summary[f'{kind}_{key}_mean'] = float(np.mean(vals))
                summary[f'{kind}_{key}_std'] = float(np.std(vals))

    for k, v in summary.items():
        print(f"{k:>40}  {v:.4f}")

    out = {'per_seed': all_results, 'summary': summary}
    pathlib.Path('results_multifault_coverage.json').write_text(
        json.dumps(out, indent=2)
    )
    print("\nWritten: results_multifault_coverage.json")


if __name__ == '__main__':
    main()
