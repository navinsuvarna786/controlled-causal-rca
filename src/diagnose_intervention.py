"""
Diagnostic harness for intervention scoring.

Tests three scoring variants against the same DAG + dataset to isolate why
empty DAG matches full DAG and why hybrid scoring degrades ambiguous-subset
accuracy.

Variants:
  V0 — current buggy: alpha*z + beta*|obs - propagate_observed(root)|
  V1 — sign-flip:    alpha*z - beta*|obs - propagate_observed(root)|
  V2 — counterfactual: improvement = sum |z_d|_observed - |z_d|_after_baseline_restore
                       Score by improvement; pick highest.

For each variant: Top-1 overall, Top-1 on ambiguous subset, structure
invocation rate. Margin sweep over [0.05, 0.10, 0.25, 0.5, 0.75, 1.0].

Run: cd kpcode/data/ && python ../src/diagnose_intervention.py
Output: outputs/diagnose/diagnostic_report.json
"""

import json
import math
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from dag import DAG, ROOT_TO_FAULT, ROOT_METRICS, all_metrics, descendants, parents_of

GLOBAL_SEED = 20260227
np.random.seed(GLOBAL_SEED)

DATA_PATH = "datacenter_dataset.csv"


def main():
    df = pd.read_csv(DATA_PATH)
    metrics = [m for m in all_metrics() if m in df.columns]
    incident_df = (
        df.groupby('incident_id', as_index=False)
          .agg({**{m: 'median' for m in metrics}, 'fault_type': 'first'})
    )
    X = incident_df[metrics]
    y = incident_df['fault_type']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=GLOBAL_SEED,
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
            lr = LinearRegression().fit(xs, ys)
            edge_models[(parent, child)] = lr
            edge_r2[(parent, child)] = float(lr.score(xs, ys))

    parents_map = parents_of()

    def root_z(row):
        return {
            r: abs(row[r] - baseline[r]['mean']) / baseline[r]['std']
            for r in ROOT_METRICS
        }

    def propagate_observed(row, root):
        pred = {root: float(row[root])}
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

    def propagate_baseline(row, root):
        """Counterfactual: set root to its healthy mean, propagate forward."""
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

    def score_v0_buggy(row, candidate, alpha=0.65, beta=0.35):
        pred = propagate_observed(row, candidate)
        z = abs(row[candidate] - baseline[candidate]['mean']) / baseline[candidate]['std']
        descs = list(descendants(candidate))
        if not descs:
            return z
        residual_sum, weight_sum = 0.0, 0.0
        for d in descs:
            par_r2 = [edge_r2.get((p, d), 0.0) for p in parents_map.get(d, [])]
            if not par_r2:
                continue
            w = max(par_r2)
            if w < 0.05:
                continue
            obs = float(row[d])
            est = pred.get(d, obs)
            std = baseline[d]['std']
            residual_sum += w * abs(est - obs) / std
            weight_sum += w
        struct = residual_sum / weight_sum if weight_sum > 0 else 0.0
        return alpha * z + beta * struct

    def score_v1_signflip(row, candidate, alpha=0.65, beta=0.35):
        pred = propagate_observed(row, candidate)
        z = abs(row[candidate] - baseline[candidate]['mean']) / baseline[candidate]['std']
        descs = list(descendants(candidate))
        if not descs:
            return z
        residual_sum, weight_sum = 0.0, 0.0
        for d in descs:
            par_r2 = [edge_r2.get((p, d), 0.0) for p in parents_map.get(d, [])]
            if not par_r2:
                continue
            w = max(par_r2)
            if w < 0.05:
                continue
            obs = float(row[d])
            est = pred.get(d, obs)
            std = baseline[d]['std']
            residual_sum += w * abs(est - obs) / std
            weight_sum += w
        struct = residual_sum / weight_sum if weight_sum > 0 else 0.0
        return alpha * z - beta * struct

    def score_v2_counterfactual(row, candidate):
        """Counterfactual improvement: sum of downstream anomaly reduction
        when candidate is restored to healthy baseline.
        """
        pred = propagate_baseline(row, candidate)
        descs = list(descendants(candidate))
        if not descs:
            return abs(row[candidate] - baseline[candidate]['mean']) / baseline[candidate]['std']
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
            est_under_restore = pred.get(d, obs)
            anomaly_observed = abs(obs - baseline[d]['mean']) / std
            anomaly_after_restore = abs(est_under_restore - baseline[d]['mean']) / std
            improvement += w * max(0.0, anomaly_observed - anomaly_after_restore)
        return improvement

    def predict_with(scoring, margin, healthy_threshold=1.5):
        preds = []
        ambiguous_idx = []
        struct_invoked = 0
        nonhealthy = 0
        for i, (_, row) in enumerate(X_test.iterrows()):
            zs = root_z(row)
            if max(zs.values()) < healthy_threshold:
                preds.append(0)
                continue
            nonhealthy += 1
            ranked = sorted(ROOT_METRICS, key=lambda r: zs[r], reverse=True)
            top1, top2 = ranked[0], ranked[1]
            if (zs[top1] - zs[top2]) > margin:
                preds.append(ROOT_TO_FAULT[top1])
                continue
            ambiguous_idx.append(i)
            struct_invoked += 1
            if scoring == 'anomaly_only':
                preds.append(ROOT_TO_FAULT[top1])
            elif scoring == 'v2_counterfactual':
                # Higher improvement = better candidate
                s1 = score_v2_counterfactual(row, top1)
                s2 = score_v2_counterfactual(row, top2)
                preds.append(ROOT_TO_FAULT[top1 if s1 >= s2 else top2])
            elif scoring == 'v0_buggy':
                s1 = score_v0_buggy(row, top1)
                s2 = score_v0_buggy(row, top2)
                # Original code: pick lower score (top1 if s1 <= s2)
                preds.append(ROOT_TO_FAULT[top1 if s1 <= s2 else top2])
            elif scoring == 'v1_signflip':
                s1 = score_v1_signflip(row, top1)
                s2 = score_v1_signflip(row, top2)
                # Higher score = better (z dominates positively, residual penalizes)
                preds.append(ROOT_TO_FAULT[top1 if s1 >= s2 else top2])
            else:
                raise ValueError(scoring)
        return np.array(preds), ambiguous_idx, struct_invoked, nonhealthy

    def evaluate(preds, ambiguous_idx):
        y = y_test.values
        top1 = float(np.mean(preds == y))
        if ambiguous_idx:
            mask = np.array(ambiguous_idx)
            amb_top1 = float(np.mean(preds[mask] == y[mask]))
        else:
            amb_top1 = float('nan')
        return top1, amb_top1

    margins = [0.05, 0.10, 0.25, 0.50, 0.75, 1.00, 2.00]
    scorings = ['anomaly_only', 'v0_buggy', 'v1_signflip', 'v2_counterfactual']

    report = {
        'seed': GLOBAL_SEED,
        'test_size': int(len(y_test)),
        'edge_count_dag': sum(len(v) for v in DAG.values()),
        'mean_edge_r2': float(np.mean(list(edge_r2.values()))),
        'median_edge_r2': float(np.median(list(edge_r2.values()))),
        'sweep': {},
    }

    print(f"{'scoring':<22} {'margin':>8} {'top1':>8} {'amb_top1':>10} {'invoked':>10} {'nonhealthy':>12}")
    print("-" * 80)
    for sc in scorings:
        report['sweep'][sc] = {}
        for m in margins:
            preds, amb_idx, inv, nh = predict_with(sc, m)
            t1, amb = evaluate(preds, amb_idx)
            report['sweep'][sc][f'margin_{m}'] = {
                'top1': t1,
                'ambiguous_top1': amb,
                'invoked_count': inv,
                'invoked_pct': inv / nh if nh else 0.0,
                'nonhealthy_count': nh,
                'ambiguous_count': len(amb_idx),
            }
            amb_str = f"{amb:.4f}" if not math.isnan(amb) else "n/a"
            print(f"{sc:<22} {m:>8.2f} {t1:>8.4f} {amb_str:>10} {inv:>10d} {nh:>12d}")

    out_path = pathlib.Path('diagnostic_report.json')
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport written: {out_path}")


if __name__ == '__main__':
    main()
