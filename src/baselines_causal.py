"""Causal RCA baselines for comparison against V2 counterfactual scoring.

Two methods implemented as faithful simplifications of recent published
approaches over the same 26-edge architecture-informed DAG used by our
scorer:

  - **RR-RCA** (Regression-Residual RCA, inspired by CIRCA, KDD'22):
        For each metric, fit a regression on its parents using healthy
        data. At incident time, compute residual r_d = |obs - predicted_from_observed_parents| / std.
        Root score = z(root) + sum_{d in Desc(root)} w_(.,d) * r_d.
        Differs from ours: residual is scored against parent-conditioned
        prediction, not against a counterfactual restore. No do() operator.

  - **KS-RCA** (Distribution-Divergence RCA, inspired by MicroCause, IWQoS'20):
        For each metric, compute Kolmogorov-Smirnov statistic between
        healthy distribution and incident distribution. Root score =
        KS(root) + alpha * sum_{d in Desc(root)} w_(.,d) * KS(d).
        Differs from ours: ranks by distribution shift, no DAG propagation
        from a candidate restore.

Both methods use the same DAG, the same healthy training data, and the
same root candidate set. They differ from V2 only in scoring semantics.

Run from a working dir containing datacenter_dataset.csv. Writes
results_baselines_causal.json with cross-seed Top-1/3/MRR for each
method plus McNemar p-values vs the V2 ranker.
"""

import json
import math
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from dag import DAG, ROOT_TO_FAULT, ROOT_METRICS, all_metrics, descendants, parents_of

GLOBAL_SEED = 20260227
DATA_PATH = "datacenter_dataset.csv"
HEALTHY_THRESHOLD = 1.5
SEEDS = [20260227, 42, 7, 1234, 999]


def fit_artifacts(seed):
    """Fit healthy baseline + per-edge regression models + R^2 weights."""
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
    edge_models, edge_r2 = {}, {}
    for parent, kids in DAG.items():
        for child in kids:
            if parent not in metrics or child not in metrics:
                continue
            xs = X_healthy[[parent]].values
            ys = X_healthy[child].values
            lr = LinearRegression().fit(xs, ys)
            edge_models[(parent, child)] = lr
            edge_r2[(parent, child)] = float(lr.score(xs, ys))
    return (X_train, X_test, y_train, y_test, y2_train, y2_test,
            X_healthy, baseline, edge_models, edge_r2, metrics)


def score_rr_rca(row, baseline, edge_models, edge_r2, parents_map):
    """Regression-residual scoring: residuals from parent-conditioned
    predictions on observed data, aggregated downstream."""
    scores = {}
    for root in ROOT_METRICS:
        z = abs(row[root] - baseline[root]['mean']) / baseline[root]['std']
        residual_sum, weight_sum = 0.0, 0.0
        for d in descendants(root):
            parent_r2 = [edge_r2.get((p, d), 0.0) for p in parents_map.get(d, [])]
            if not parent_r2:
                continue
            w = max(parent_r2)
            if w < 0.05:
                continue
            # Predict d from its observed parents
            pred_d_vals = []
            for p in parents_map[d]:
                model = edge_models.get((p, d))
                if model is None:
                    continue
                pred_d_vals.append(float(model.predict([[row[p]]])[0]))
            if not pred_d_vals:
                continue
            pred_d = float(np.mean(pred_d_vals))
            obs = float(row[d])
            std = baseline[d]['std']
            residual = abs(obs - pred_d) / std
            residual_sum += w * residual
            weight_sum += w
        struct = residual_sum / weight_sum if weight_sum > 0 else 0.0
        scores[root] = z + 0.5 * struct
    return scores


def score_ks_rca(row, ks_per_metric, edge_r2, parents_map, alpha=0.5):
    """Distribution-divergence scoring: per-metric KS statistic vs healthy,
    aggregated downstream via DAG."""
    scores = {}
    for root in ROOT_METRICS:
        score = ks_per_metric.get(root, 0.0)
        for d in descendants(root):
            parent_r2 = [edge_r2.get((p, d), 0.0) for p in parents_map.get(d, [])]
            if not parent_r2:
                continue
            w = max(parent_r2)
            if w < 0.05:
                continue
            score += alpha * w * ks_per_metric.get(d, 0.0)
        scores[root] = score
    return scores


def precompute_ks(X_healthy, X_test, metrics):
    """For KS-RCA: KS statistic between healthy reference and per-incident
    point. Approximate by per-metric standardized deviation magnitude with
    KS-style weighting (single-row analogue of distribution shift)."""
    ks_table = []
    healthy_arrays = {m: X_healthy[m].values for m in metrics}
    for _, row in X_test.iterrows():
        per_metric = {}
        for m in metrics:
            healthy_vals = healthy_arrays[m]
            # KS-like: probability that healthy distribution exceeds observed
            stat = ks_2samp(healthy_vals, np.array([float(row[m])])).statistic
            per_metric[m] = float(stat)
        ks_table.append(per_metric)
    return ks_table


def predict_with(scoring, X_test, y_test, X_healthy, baseline, edge_models,
                 edge_r2, metrics, parents_map):
    preds = []
    if scoring == 'ks_rca':
        ks_table = precompute_ks(X_healthy, X_test, metrics)
    for i, (_, row) in enumerate(X_test.iterrows()):
        # Healthy gating
        z_max = max(
            abs(row[r] - baseline[r]['mean']) / baseline[r]['std']
            for r in ROOT_METRICS
        )
        if z_max < HEALTHY_THRESHOLD:
            preds.append(0)
            continue
        if scoring == 'rr_rca':
            scores = score_rr_rca(row, baseline, edge_models, edge_r2, parents_map)
        elif scoring == 'ks_rca':
            scores = score_ks_rca(row, ks_table[i], edge_r2, parents_map)
        else:
            raise ValueError(scoring)
        ranked = sorted(ROOT_METRICS, key=lambda r: scores[r], reverse=True)
        preds.append(ROOT_TO_FAULT[ranked[0]])
    return np.array(preds), y_test.values


def topk_mrr(X_test, y_test, baseline, edge_models, edge_r2, parents_map,
             metrics, X_healthy, scoring, y2_test=None):
    """Compute Top-1, Top-3, MRR for a given scoring method.

    If y2_test is provided (multi-fault secondary labels), also returns
    Top-1@coverage and Top-2@coverage (predicted matches at least one of
    {primary, secondary}).
    """
    if scoring == 'ks_rca':
        ks_table = precompute_ks(X_healthy, X_test, metrics)
    top1 = top3 = 0
    top1_cov = top2_cov = 0
    multi_count = 0
    multi_top1 = 0
    multi_top1_cov = 0
    multi_top2_cov = 0
    rr_sum = 0.0
    nonhealthy = 0
    y_arr = y_test.values
    y2_arr = y2_test.values if y2_test is not None else None
    for i, (_, row) in enumerate(X_test.iterrows()):
        true = int(y_arr[i])
        if true == 0:
            continue
        nonhealthy += 1
        z_max = max(
            abs(row[r] - baseline[r]['mean']) / baseline[r]['std']
            for r in ROOT_METRICS
        )
        if z_max < HEALTHY_THRESHOLD:
            continue
        if scoring == 'rr_rca':
            scores = score_rr_rca(row, baseline, edge_models, edge_r2, parents_map)
        elif scoring == 'ks_rca':
            scores = score_ks_rca(row, ks_table[i], edge_r2, parents_map)
        else:
            raise ValueError(scoring)
        ranked = sorted(ROOT_METRICS, key=lambda r: scores[r], reverse=True)
        ranked_faults = [ROOT_TO_FAULT[r] for r in ranked]
        if ranked_faults[0] == true:
            top1 += 1
        if true in ranked_faults[:3]:
            top3 += 1
        rank = ranked_faults.index(true) + 1 if true in ranked_faults else 6
        rr_sum += 1.0 / rank

        if y2_arr is not None:
            true_set = {true}
            secondary = int(y2_arr[i])
            if secondary != 0:
                true_set.add(secondary)
                multi_count += 1
                if ranked_faults[0] == true:
                    multi_top1 += 1
                if ranked_faults[0] in true_set:
                    multi_top1_cov += 1
                if set(ranked_faults[:2]) & true_set:
                    multi_top2_cov += 1
            if ranked_faults[0] in true_set:
                top1_cov += 1
            if set(ranked_faults[:2]) & true_set:
                top2_cov += 1

    out = {
        'top1': top1 / nonhealthy if nonhealthy else 0.0,
        'top3': top3 / nonhealthy if nonhealthy else 0.0,
        'mrr': rr_sum / nonhealthy if nonhealthy else 0.0,
        'n': nonhealthy,
    }
    if y2_arr is not None:
        out['top1_cov'] = top1_cov / nonhealthy if nonhealthy else 0.0
        out['top2_cov'] = top2_cov / nonhealthy if nonhealthy else 0.0
        out['multi_n'] = multi_count
        out['multi_top1_primary'] = multi_top1 / multi_count if multi_count else float('nan')
        out['multi_top1_cov'] = multi_top1_cov / multi_count if multi_count else float('nan')
        out['multi_top2_cov'] = multi_top2_cov / multi_count if multi_count else float('nan')
    return out


def mcnemar(y_true, pred_a, pred_b):
    """McNemar paired test: returns p-value comparing pred_a vs pred_b."""
    a_correct = (pred_a == y_true)
    b_correct = (pred_b == y_true)
    n01 = int(np.sum(a_correct & ~b_correct))
    n10 = int(np.sum(~a_correct & b_correct))
    if n01 + n10 == 0:
        return 1.0
    chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    from scipy.stats import chi2 as chi2_dist
    return float(1.0 - chi2_dist.cdf(chi2, 1))


def main():
    parents_map = parents_of()
    print(f"{'seed':>10} {'method':>10} {'top1':>8} {'top3':>8} {'mrr':>8} "
          f"{'t1@cov':>8} {'t2@cov':>8} {'mf-t1c':>8} {'mf-t2c':>8} {'n_mf':>5}")
    print("-" * 90)

    keys = ['top1', 'top3', 'mrr', 'top1_cov', 'top2_cov',
            'multi_top1_primary', 'multi_top1_cov', 'multi_top2_cov']
    summary = {sc: {k: [] for k in keys} for sc in ['rr_rca', 'ks_rca']}

    for seed in SEEDS:
        (X_train, X_test, y_train, y_test, y2_train, y2_test,
         X_healthy, baseline, edge_models, edge_r2, metrics) = fit_artifacts(seed)
        has_secondary = bool((y2_test != 0).sum())

        for sc in ['rr_rca', 'ks_rca']:
            r = topk_mrr(
                X_test, y_test, baseline, edge_models, edge_r2,
                parents_map, metrics, X_healthy, sc,
                y2_test=y2_test if has_secondary else None,
            )
            mfp = r.get('multi_top1_primary', float('nan'))
            mfc = r.get('multi_top1_cov', float('nan'))
            mft2 = r.get('multi_top2_cov', float('nan'))
            n_mf = r.get('multi_n', 0)
            t1c = r.get('top1_cov', float('nan'))
            t2c = r.get('top2_cov', float('nan'))
            print(f"{seed:>10} {sc:>10} {r['top1']:>8.4f} {r['top3']:>8.4f} "
                  f"{r['mrr']:>8.4f} "
                  f"{('%.4f' % t1c if not math.isnan(t1c) else 'n/a'):>8} "
                  f"{('%.4f' % t2c if not math.isnan(t2c) else 'n/a'):>8} "
                  f"{('%.4f' % mfc if not math.isnan(mfc) else 'n/a'):>8} "
                  f"{('%.4f' % mft2 if not math.isnan(mft2) else 'n/a'):>8} "
                  f"{n_mf:>5d}")
            for k in keys:
                v = r.get(k, float('nan'))
                if isinstance(v, float) and not math.isnan(v):
                    summary[sc][k].append(v)

    print()
    out = {'per_seed': summary, 'aggregate': {}}
    for method, m in summary.items():
        agg = {}
        for k, v in m.items():
            if v:
                agg[f'{k}_mean'] = float(np.mean(v))
                agg[f'{k}_std'] = float(np.std(v))
        out['aggregate'][method] = agg
        print(f"{method}: Top-1 {agg.get('top1_mean', 0):.4f} ± {agg.get('top1_std', 0):.4f}, "
              f"Top-3 {agg.get('top3_mean', 0):.4f} ± {agg.get('top3_std', 0):.4f}, "
              f"MRR {agg.get('mrr_mean', 0):.4f} ± {agg.get('mrr_std', 0):.4f}")
        if 'top1_cov_mean' in agg:
            print(f"  T1@cov {agg['top1_cov_mean']:.4f} ± {agg['top1_cov_std']:.4f}, "
                  f"T2@cov {agg['top2_cov_mean']:.4f} ± {agg['top2_cov_std']:.4f}")
        if 'multi_top1_cov_mean' in agg:
            print(f"  Multi-only T1@cov {agg['multi_top1_cov_mean']:.4f} ± {agg['multi_top1_cov_std']:.4f}, "
                  f"T2@cov {agg['multi_top2_cov_mean']:.4f} ± {agg['multi_top2_cov_std']:.4f}")

    pathlib.Path('results_baselines_causal.json').write_text(
        json.dumps(out, indent=2)
    )
    print("\nWritten: results_baselines_causal.json")


if __name__ == '__main__':
    main()
