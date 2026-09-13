"""Scaling experiment: inference latency as a function of network size N.

Builds synthetic layered DAGs of size N in {28, 64, 128, 256, 512} that
preserve the structural profile of the base spine-leaf DAG (5 layers,
density ~0.034, longest path ~5, ~5 root candidates). For each size,
fits Linear SEM on healthy data and measures per-incident inference
latency for the V2 counterfactual scorer.

Output: outputs/scaling/scaling_results.json + scaling_latency.{png,pdf}
"""

import json
import math
import pathlib
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

GLOBAL_SEED = 20260227
SIZES = [28, 64, 128, 256, 512]
ROOTS_PER_NETWORK = 5
TARGET_DENSITY = 0.034
TARGET_DEPTH = 5
N_HEALTHY = 100
N_INCIDENT = 50
TRIALS = 3


def synth_layered_dag(n_nodes, rng):
    """Generate a layered DAG with ~ROOTS_PER_NETWORK roots, depth=TARGET_DEPTH,
    density ~TARGET_DENSITY. Returns dict parent -> [children]."""
    layers = TARGET_DEPTH + 1
    sizes = []
    sizes.append(ROOTS_PER_NETWORK)
    remaining = n_nodes - ROOTS_PER_NETWORK
    per_layer = max(1, remaining // (layers - 1))
    for _ in range(layers - 2):
        sizes.append(per_layer)
        remaining -= per_layer
    sizes.append(max(1, remaining))

    nodes_by_layer = []
    next_id = 0
    for sz in sizes:
        layer_nodes = [f"n{next_id + i}" for i in range(sz)]
        nodes_by_layer.append(layer_nodes)
        next_id += sz

    target_edges = int(round(TARGET_DENSITY * n_nodes * (n_nodes - 1)))
    dag = {}
    edge_count = 0
    for li in range(layers - 1):
        upper = nodes_by_layer[li]
        lower = nodes_by_layer[li + 1]
        for u in upper:
            n_children = max(1, int(round(target_edges / (n_nodes * (layers - 1) / n_nodes))))
            n_children = min(n_children, len(lower))
            children = list(rng.choice(lower, size=n_children, replace=False))
            dag[u] = children
            edge_count += n_children
            if edge_count >= target_edges:
                break
        if edge_count >= target_edges:
            break

    all_nodes = [n for layer in nodes_by_layer for n in layer]
    roots = nodes_by_layer[0]
    return dag, all_nodes, roots


def descendants_of(dag, node):
    seen = set()
    stack = [node]
    while stack:
        cur = stack.pop()
        for k in dag.get(cur, []):
            if k not in seen:
                seen.add(k)
                stack.append(k)
    return seen


def synthesize_data(all_nodes, n_rows, rng, perturb_root=None):
    """Generate synthetic numeric rows for all nodes."""
    data = {n: rng.normal(0, 1, size=n_rows) for n in all_nodes}
    if perturb_root is not None:
        data[perturb_root] = data[perturb_root] + rng.normal(2.5, 0.5, size=n_rows)
    return pd.DataFrame(data)


def measure_latency(n_nodes, seed):
    rng = np.random.default_rng(seed + n_nodes)
    dag, all_nodes, roots = synth_layered_dag(n_nodes, rng)
    parents_map = {n: [] for n in all_nodes}
    for p, kids in dag.items():
        for k in kids:
            parents_map[k].append(p)

    healthy = synthesize_data(all_nodes, N_HEALTHY, rng)
    baseline = {
        n: {'mean': float(healthy[n].mean()),
            'std': float(healthy[n].std() + 1e-9)}
        for n in all_nodes
    }
    edge_models, edge_r2 = {}, {}
    for parent, kids in dag.items():
        for child in kids:
            if not parents_map[child]:
                continue
            xs = healthy[[parent]].values
            ys = healthy[child].values
            lr = LinearRegression().fit(xs, ys)
            edge_models[(parent, child)] = lr
            edge_r2[(parent, child)] = float(lr.score(xs, ys))

    incident = synthesize_data(
        all_nodes, N_INCIDENT, rng, perturb_root=roots[0],
    )

    def propagate_from_baseline(row, root):
        pred = {root: baseline[root]['mean']}
        frontier = [root]
        while frontier:
            par = frontier.pop(0)
            for child in dag.get(par, []):
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

    def score(row, candidate):
        pred = propagate_from_baseline(row, candidate)
        descs = list(descendants_of(dag, candidate))
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

    # Warm up
    for _, row in incident.head(3).iterrows():
        for r in roots:
            score(row, r)

    # Measure
    start = time.perf_counter()
    for _, row in incident.iterrows():
        for r in roots:
            score(row, r)
    elapsed = time.perf_counter() - start
    per_incident_ms = (elapsed / len(incident)) * 1000.0
    return {
        'n_nodes': n_nodes,
        'n_edges': sum(len(v) for v in dag.values()),
        'n_roots': len(roots),
        'per_incident_ms': per_incident_ms,
        'longest_path': TARGET_DEPTH,
    }


def main():
    out_dir = pathlib.Path('outputs/scaling')
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    print(f"{'N':>6} {'edges':>6} {'roots':>6} {'mean ms':>10} {'std ms':>10}")
    print("-" * 50)
    for n in SIZES:
        trial_ms = []
        n_edges_recorded = None
        for t in range(TRIALS):
            r = measure_latency(n, GLOBAL_SEED + t)
            trial_ms.append(r['per_incident_ms'])
            n_edges_recorded = r['n_edges']
        mean_ms = float(np.mean(trial_ms))
        std_ms = float(np.std(trial_ms))
        results.append({
            'n_nodes': n,
            'n_edges': n_edges_recorded,
            'mean_ms': mean_ms,
            'std_ms': std_ms,
            'trials': trial_ms,
        })
        print(f"{n:>6d} {n_edges_recorded:>6d} {ROOTS_PER_NETWORK:>6d} "
              f"{mean_ms:>10.4f} {std_ms:>10.4f}")

    (out_dir / 'scaling_results.json').write_text(json.dumps(results, indent=2))

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    xs = [r['n_nodes'] for r in results]
    ys = [r['mean_ms'] for r in results]
    yerr = [r['std_ms'] for r in results]
    ax.errorbar(xs, ys, yerr=yerr, marker='o', linewidth=1.5, capsize=3,
                color='black', markerfacecolor='white')
    ax.set_xlabel('Network size $N$ (nodes)')
    ax.set_ylabel('Inference latency (ms / incident)')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.grid(True, which='both', alpha=0.3, linewidth=0.4)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    fig.tight_layout()
    fig.savefig(out_dir / 'scaling_latency.png', dpi=300, bbox_inches='tight')
    fig.savefig(out_dir / 'scaling_latency.pdf', bbox_inches='tight')
    print(f"\nSaved: {out_dir}/scaling_latency.{{png,pdf}}")


if __name__ == '__main__':
    main()
