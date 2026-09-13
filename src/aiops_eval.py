#!/usr/bin/env python3

import json
import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

SEED = 20260227
np.random.seed(SEED)

DATASET_PATH = "aiops_reduced_dataset.csv"
ADJ_PATH = "aiops_adjacency_binary.csv"
LAYERS_PATH = "aiops_node_layers.csv"
OUTPUT_PATH = "results_aiops_realworld_rca.json"

HEALTHY_THRESHOLD = 1.5
MARGIN = 0.25
ALPHA = 0.7
BETA = 0.3


def load_incident_level_dataset():
    df = pd.read_csv(DATASET_PATH)
    _ = pd.read_csv(LAYERS_PATH)

    reserved = {"timestamp", "incident_id", "fault_type", "date"}
    metric_cols = [c for c in df.columns if c not in reserved]

    incident_df = (
        df.groupby("incident_id", as_index=False)
        .agg({**{m: "median" for m in metric_cols}, "fault_type": "first"})
    )

    print("Incident-level dataset shape:", incident_df.shape)
    print("Fault distribution (incident-level):")
    print(incident_df["fault_type"].value_counts().sort_index())

    return incident_df, metric_cols


def build_dag_from_adjacency():
    adj = pd.read_csv(ADJ_PATH, index_col=0)
    nodes = list(adj.index)

    dag = {node: [] for node in nodes}
    parents = {node: [] for node in nodes}

    for parent in nodes:
        row = adj.loc[parent]
        for child in nodes:
            if float(row[child]) > 0.5:
                dag[parent].append(child)
                parents[child].append(parent)

    descendants = {}
    for node in nodes:
        visited = set()
        stack = [node]
        while stack:
            current = stack.pop()
            for child in dag.get(current, []):
                if child not in visited:
                    visited.add(child)
                    stack.append(child)
        descendants[node] = visited

    indegree = {node: len(parents[node]) for node in nodes}
    roots = [node for node, deg in indegree.items() if deg == 0]

    return nodes, dag, parents, descendants, roots


def map_roots_to_faults(incident_df, roots):
    fault_values = sorted(v for v in incident_df["fault_type"].unique() if v != 0)
    root_to_fault = {}

    for root in roots:
        x = incident_df[root].astype(float)
        best_fault = 0
        best_corr = -np.inf

        for fault in fault_values:
            y_bin = (incident_df["fault_type"] == fault).astype(float)
            if x.std() == 0 or y_bin.std() == 0:
                corr_abs = 0.0
            else:
                corr = x.corr(y_bin)
                corr_abs = abs(float(corr)) if pd.notna(corr) else 0.0
            if corr_abs > best_corr:
                best_corr = corr_abs
                best_fault = int(fault)

        root_to_fault[root] = best_fault

    print("\nRoot -> fault mapping:")
    for root in sorted(root_to_fault):
        print(f"  {root} -> {root_to_fault[root]}")

    return root_to_fault


def to_fault_ranking_from_roots(root_ranking, root_to_fault, fault_labels):
    ranked_faults = []
    seen = set()

    for root in root_ranking:
        fault = root_to_fault.get(root, 0)
        if fault != 0 and fault not in seen:
            ranked_faults.append(fault)
            seen.add(fault)

    for fault in fault_labels:
        if fault != 0 and fault not in seen:
            ranked_faults.append(int(fault))

    return ranked_faults


def top1_fault_only_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(y_true[mask] == y_pred[mask]))


def compute_topk_and_mrr(y_true, ranked_faults_all, k=3):
    hits = []
    reciprocal_ranks = []

    for yt, ranked_faults in zip(y_true, ranked_faults_all):
        if yt == 0:
            continue

        topk = ranked_faults[:k]
        hits.append(1.0 if yt in topk else 0.0)

        rr = 0.0
        for idx, pred in enumerate(ranked_faults, start=1):
            if pred == yt:
                rr = 1.0 / idx
                break
        reciprocal_ranks.append(rr)

    topk_acc = float(np.mean(hits)) if hits else 0.0
    mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
    return topk_acc, mrr


def mcnemar_pvalue(y_true, pred_a, pred_b):
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)

    mask = y_true != 0
    y_true = y_true[mask]
    pred_a = pred_a[mask]
    pred_b = pred_b[mask]

    a_correct = pred_a == y_true
    b_correct = pred_b == y_true

    n01 = int(np.sum(a_correct & (~b_correct)))
    n10 = int(np.sum((~a_correct) & b_correct))

    denom = n01 + n10
    if denom == 0:
        return 1.0

    chi2 = ((abs(n01 - n10) - 1.0) ** 2) / denom
    p_value = math.erfc(math.sqrt(max(chi2, 0.0) / 2.0))
    return float(p_value)


class CausalInterventionRCA:
    def __init__(self, nodes, roots, root_to_fault, dag, parents, descendants):
        self.nodes = nodes
        self.roots = roots
        self.root_to_fault = root_to_fault
        self.dag = dag
        self.parents = parents
        self.descendants = descendants

        self.baseline = {}
        self.models = {}
        self.edge_r2 = {}

    def fit(self, train_df, healthy_df):
        source_df = healthy_df if len(healthy_df) > 0 else train_df

        for node in self.nodes:
            std = float(source_df[node].std()) if node in source_df else 0.0
            self.baseline[node] = {
                "mean": float(source_df[node].mean()) if node in source_df else 0.0,
                "std": std + 1e-9,
            }

        for parent, children in self.dag.items():
            for child in children:
                edge_df = source_df[[parent, child]].dropna()
                if len(edge_df) < 2:
                    continue

                x_parent = edge_df[[parent]].to_numpy()
                y_child = edge_df[child].to_numpy()

                model = LinearRegression()
                model.fit(x_parent, y_child)
                self.models[(parent, child)] = model
                self.edge_r2[(parent, child)] = float(model.score(x_parent, y_child))

    def root_anomaly_scores(self, row):
        scores = {}
        for root in self.roots:
            mean = self.baseline[root]["mean"]
            std = self.baseline[root]["std"]
            scores[root] = abs(float(row[root]) - mean) / std
        return scores

    def propagate_from_root(self, row, root_metric):
        predicted = {root_metric: float(row[root_metric])}
        frontier = [root_metric]

        while frontier:
            parent = frontier.pop(0)
            for child in self.dag.get(parent, []):
                model = self.models.get((parent, child))
                if model is None or parent not in predicted:
                    continue

                parent_value = predicted[parent]
                child_est = float(model.predict(np.array([[parent_value]], dtype=float))[0])

                if child in predicted:
                    predicted[child] = 0.5 * (predicted[child] + child_est)
                else:
                    predicted[child] = child_est
                    frontier.append(child)

        return predicted

    def simulate_intervention(self, row, root_metric):
        root_scores = self.root_anomaly_scores(row)
        root_z = root_scores[root_metric]

        predicted = self.propagate_from_root(row, root_metric)
        descendants = list(self.descendants.get(root_metric, set()))

        if not descendants:
            return ALPHA * root_z

        weighted_residual_sum = 0.0
        total_weight = 0.0

        for node in descendants:
            parent_weights = [
                self.edge_r2.get((p, node), 0.0)
                for p in self.parents.get(node, [])
            ]
            node_weight = max(parent_weights) if parent_weights else 0.0
            if node_weight <= 0.0:
                continue

            obs = float(row[node])
            pred = float(predicted.get(node, self.baseline[node]["mean"]))
            std = self.baseline[node]["std"]
            residual = abs(obs - pred) / std

            weighted_residual_sum += node_weight * residual
            total_weight += node_weight

        structural_score = (weighted_residual_sum / total_weight) if total_weight > 0 else 0.0
        return ALPHA * root_z + BETA * structural_score

    def predict_ranked(self, X_test):
        fault_labels = sorted(int(v) for v in pd.unique(X_test["fault_type"]) if v != 0)

        top1_preds = []
        ranked_faults_all = []
        structure_invoked = 0
        nonhealthy_count = 0

        ambiguous_anomaly_preds = []
        ambiguous_final_preds = []
        ambiguous_true = []

        for _, row in X_test.iterrows():
            root_scores = self.root_anomaly_scores(row)
            max_root_z = max(root_scores.values()) if root_scores else 0.0

            if max_root_z < HEALTHY_THRESHOLD:
                top1_preds.append(0)
                ranked_faults_all.append(fault_labels[:])
                continue

            nonhealthy_count += 1

            sorted_roots = sorted(self.roots, key=lambda r: root_scores[r], reverse=True)
            if len(sorted_roots) == 0:
                top1_preds.append(0)
                ranked_faults_all.append(fault_labels[:])
                continue
            if len(sorted_roots) == 1:
                top_root = sorted_roots[0]
                pred_fault = self.root_to_fault.get(top_root, 0)
                top1_preds.append(pred_fault)
                ranked_faults_all.append(to_fault_ranking_from_roots(sorted_roots, self.root_to_fault, fault_labels))
                continue

            top1_root, top2_root = sorted_roots[0], sorted_roots[1]
            z1 = root_scores[top1_root]
            z2 = root_scores[top2_root]

            anomaly_only_fault = self.root_to_fault.get(top1_root, 0)

            if (z1 - z2) > MARGIN:
                chosen_root = top1_root
            else:
                structure_invoked += 1
                s1 = self.simulate_intervention(row, top1_root)
                s2 = self.simulate_intervention(row, top2_root)
                chosen_root = top1_root if s1 <= s2 else top2_root

                ambiguous_anomaly_preds.append(anomaly_only_fault)
                ambiguous_final_preds.append(self.root_to_fault.get(chosen_root, 0))
                ambiguous_true.append(int(row["fault_type"]))

            remaining = [r for r in sorted_roots if r != chosen_root]
            final_root_ranking = [chosen_root] + remaining

            pred_fault = self.root_to_fault.get(chosen_root, 0)
            top1_preds.append(pred_fault)
            ranked_faults_all.append(
                to_fault_ranking_from_roots(final_root_ranking, self.root_to_fault, fault_labels)
            )

        invocation_pct = (100.0 * structure_invoked / nonhealthy_count) if nonhealthy_count > 0 else 0.0

        if len(ambiguous_true) > 0:
            amb_true = np.asarray(ambiguous_true)
            anomaly_acc = float(np.mean(np.asarray(ambiguous_anomaly_preds) == amb_true))
            final_acc = float(np.mean(np.asarray(ambiguous_final_preds) == amb_true))
            ambiguous_improvement = final_acc - anomaly_acc
        else:
            ambiguous_improvement = 0.0

        return (
            np.asarray(top1_preds),
            ranked_faults_all,
            float(invocation_pct),
            float(ambiguous_improvement),
        )


class PearsonBaseline:
    def __init__(self, roots, root_to_fault):
        self.roots = roots
        self.root_to_fault = root_to_fault
        self.root_weights = {}
        self.baseline = {}

    def fit(self, train_df, healthy_df):
        baseline_source = healthy_df if len(healthy_df) > 0 else train_df

        for root in self.roots:
            std = float(baseline_source[root].std()) if root in baseline_source else 0.0
            self.baseline[root] = {
                "mean": float(baseline_source[root].mean()) if root in baseline_source else 0.0,
                "std": std + 1e-9,
            }

            target_fault = self.root_to_fault.get(root, 0)
            y_bin = (train_df["fault_type"] == target_fault).astype(float)
            x_root = train_df[root].astype(float)
            if x_root.std() == 0 or y_bin.std() == 0:
                self.root_weights[root] = 0.0
            else:
                corr = x_root.corr(y_bin)
                self.root_weights[root] = abs(float(corr)) if pd.notna(corr) else 0.0

    def predict_ranked(self, X_test):
        fault_labels = sorted(int(v) for v in pd.unique(X_test["fault_type"]) if v != 0)

        top1_preds = []
        ranked_faults_all = []

        for _, row in X_test.iterrows():
            weighted = {}
            for root in self.roots:
                mean = self.baseline[root]["mean"]
                std = self.baseline[root]["std"]
                z = abs(float(row[root]) - mean) / std
                weighted[root] = z * self.root_weights.get(root, 0.0)

            ranked_roots = sorted(self.roots, key=lambda r: weighted[r], reverse=True)
            top1_fault = self.root_to_fault.get(ranked_roots[0], 0) if ranked_roots else 0

            top1_preds.append(top1_fault)
            ranked_faults_all.append(
                to_fault_ranking_from_roots(ranked_roots, self.root_to_fault, fault_labels)
            )

        return np.asarray(top1_preds), ranked_faults_all


def main():
    incident_df, _ = load_incident_level_dataset()
    nodes, dag, parents, descendants, roots = build_dag_from_adjacency()
    root_to_fault = map_roots_to_faults(incident_df, roots)

    X = incident_df[nodes]
    y = incident_df["fault_type"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=SEED,
        stratify=y,
    )

    train_df = X_train.copy()
    train_df["fault_type"] = y_train.values

    healthy_mask = y_train == 0
    healthy_df = X_train[healthy_mask].copy()

    causal = CausalInterventionRCA(nodes, roots, root_to_fault, dag, parents, descendants)
    causal.fit(train_df, healthy_df)

    X_test_eval = X_test.copy()
    X_test_eval["fault_type"] = y_test.values

    causal_top1, causal_ranked, structure_invocation_pct, ambiguous_improvement = causal.predict_ranked(X_test_eval)

    pearson = PearsonBaseline(roots, root_to_fault)
    pearson.fit(train_df, healthy_df)
    pearson_top1, pearson_ranked = pearson.predict_ranked(X_test_eval)

    causal_top1_acc = top1_fault_only_accuracy(y_test.values, causal_top1)
    pearson_top1_acc = top1_fault_only_accuracy(y_test.values, pearson_top1)

    causal_top3, causal_mrr = compute_topk_and_mrr(y_test.values, causal_ranked, k=3)
    pearson_top3, pearson_mrr = compute_topk_and_mrr(y_test.values, pearson_ranked, k=3)

    mcnemar_p = mcnemar_pvalue(y_test.values, causal_top1, pearson_top1)

    print("\n=== AIOps Real-World RCA Results ===")
    print(f"Causal Top-1: {causal_top1_acc:.4f}")
    print(f"Causal Top-3: {causal_top3:.4f}")
    print(f"Causal MRR: {causal_mrr:.4f}")
    print(f"Pearson Top-1: {pearson_top1_acc:.4f}")
    print(f"Pearson Top-3: {pearson_top3:.4f}")
    print(f"Pearson MRR: {pearson_mrr:.4f}")
    print(f"McNemar p-value: {mcnemar_p:.6f}")
    print(f"Structure Invocation %: {structure_invocation_pct:.2f}")
    print(f"Ambiguous subset improvement: {ambiguous_improvement:.4f}")

    results = {
        "seed": SEED,
        "healthy_threshold": HEALTHY_THRESHOLD,
        "margin": MARGIN,
        "alpha": ALPHA,
        "beta": BETA,
        "dataset_path": DATASET_PATH,
        "adjacency_path": ADJ_PATH,
        "node_layers_path": LAYERS_PATH,
        "num_incidents": int(len(incident_df)),
        "num_metrics": int(len(nodes)),
        "num_roots": int(len(roots)),
        "causal_top1_fault_only": float(causal_top1_acc),
        "causal_top3": float(causal_top3),
        "causal_mrr": float(causal_mrr),
        "pearson_top1_fault_only": float(pearson_top1_acc),
        "pearson_top3": float(pearson_top3),
        "pearson_mrr": float(pearson_mrr),
        "mcnemar_p_value": float(mcnemar_p),
        "structure_invocation_percent": float(structure_invocation_pct),
        "ambiguous_subset_improvement": float(ambiguous_improvement),
        "root_to_fault": {k: int(v) for k, v in root_to_fault.items()},
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
