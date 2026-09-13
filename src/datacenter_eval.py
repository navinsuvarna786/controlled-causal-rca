#!/usr/bin/env python3
"""
Datacenter Causal RCA Evaluation (Deterministic + Full SCM)

Features:
- Stratified train/test split
- Train SEM only on healthy samples
- Linear + RF structural equations
- Intervention-based root cause ranking
- Accuracy + inference speed
- Residual skew/kurtosis diagnostics
"""

import numpy as np
import pandas as pd
import time
import hashlib
import json
import subprocess
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from scipy.stats import skew, kurtosis
import networkx as nx
from statsmodels.stats.contingency_tables import mcnemar

# ===============================
# REPRODUCIBILITY
# ===============================

GLOBAL_SEED = 20260227
np.random.seed(GLOBAL_SEED)

VERSION_STRING = "DATACENTER_RCA_EVAL_V10_FULL_SCM"
VERSION_HASH = hashlib.sha256(VERSION_STRING.encode()).hexdigest()

print(f"\nEvaluation Version: {VERSION_STRING}")
print(f"Version Hash: {VERSION_HASH}")
print(f"Random Seed: {GLOBAL_SEED}\n")

# ===============================
# LOAD DATA
# ===============================

DATA_PATH = "datacenter_dataset.csv"
df = pd.read_csv(DATA_PATH)

# ===============================
# METRICS (28 EXACT)
# ===============================

METRICS = [
    'spine_packet_loss_rate','spine_queue_depth','spine_cpu_util','spine_link_util',
    'leaf_packet_loss_rate','leaf_queue_depth','leaf_cpu_util',
    'leaf_link_util_uplink','leaf_link_util_downlink','leaf_buffer_drops',
    'tor_congestion_index','server_retransmit_rate','server_conn_timeout_rate',
    'server_app_latency','vm_density',
    'optical_rx_power','optical_ber','transceiver_temp',
    'vxlan_encap_error_rate','vxlan_decap_error_rate','vtep_reachability',
    'bgp_route_flaps','bgp_convergence_time','ecmp_path_imbalance','routing_loop_detected',
    'bgp_update_rate','arp_request_rate','lldp_frame_rate'
]

if len(METRICS) != 28:
    raise ValueError(f"Expected 28 metrics, found {len(METRICS)}")

if not set(METRICS).issubset(df.columns):
    missing = sorted(set(METRICS) - set(df.columns))
    raise ValueError(f"Dataset missing expected metrics: {missing}")

if 'incident_id' not in df.columns or 'fault_type' not in df.columns:
    raise ValueError("Dataset must include incident_id and fault_type columns")

print("Dataset Shape:", df.shape)
print("\nFault Distribution (row-level):")
print(df["fault_type"].value_counts().sort_index())

healthy_row_count = int((df['fault_type'] == 0).sum())

print("\nReproducibility Metadata:")
print("Version String:", VERSION_STRING)
print("Version Hash:", VERSION_HASH)
print("Random Seed:", GLOBAL_SEED)
print("Dataset Row Count:", len(df))
print("Healthy Sample Count:", healthy_row_count)

# ===============================
# CAUSAL DAG
# ===============================

class DatacenterCAG:
    def __init__(self):
        self.metric_names = METRICS
        self.dag = self._build_dag()
        print("Total DAG edges:", sum(len(v) for v in self.dag.values()))
        self.descendants = {m: self._get_descendants(m) for m in METRICS}
        self.root_to_fault = {
            'optical_rx_power': 1,
            'leaf_link_util_downlink': 2,
            'vtep_reachability': 3,
            'bgp_route_flaps': 4,
            'bgp_update_rate': 5
        }
        self.root_metrics = list(self.root_to_fault.keys())
        self.parents = {m: [] for m in METRICS}
        for parent, children in self.dag.items():
            for child in children:
                self.parents[child].append(parent)

    def _build_dag(self):
        # Single source of truth: dag.DAG, sourced from
        # data/datacenter_adjacency_binary.csv (26 edges, 25 connected nodes).
        from dag import DAG as _DAG
        return {p: list(kids) for p, kids in _DAG.items()}

    def _get_descendants(self, node):
        desc = set()
        stack = [node]
        while stack:
            n = stack.pop()
            for child in self.dag.get(n, []):
                if child not in desc:
                    desc.add(child)
                    stack.append(child)
        return desc

# ===============================
# CAUSAL INTERVENTION RCA
# ===============================

class CausalInterventionRCA:

    def __init__(self, cag, use_rf=False):
        self.cag = cag
        self.use_rf = use_rf
        self.baseline = {}
        self.models = {}
        self.edge_r2 = {}
        self.healthy_threshold = 1.6 if use_rf else 1.4

    def fit(self, X_healthy):
        if X_healthy.shape[0] == 0:
            raise ValueError(
                "No healthy samples in training split (y_train == 0 required) before SEM fitting."
            )

        # Baseline stats
        for m in METRICS:
            self.baseline[m] = {
                "mean": X_healthy[m].mean(),
                "std": X_healthy[m].std() + 1e-9
            }

        # Structural equations
        for parent, children in self.cag.dag.items():
            for child in children:
                Xp = X_healthy[[parent]].values
                yc = X_healthy[child].values

                if self.use_rf:
                    model = RandomForestRegressor(
                        n_estimators=50,
                        max_depth=8,
                        random_state=GLOBAL_SEED
                    )
                else:
                    model = LinearRegression()

                model.fit(Xp, yc)
                self.models[(parent, child)] = model

    def root_anomaly_scores(self, row):
        return {
            root: abs(row[root] - self.baseline[root]["mean"]) / self.baseline[root]["std"]
            for root in self.cag.root_metrics
        }

    def propagate_from_root(self, row, root_metric):
        predicted = {root_metric: float(row[root_metric])}
        frontier = [root_metric]

        while frontier:
            parent = frontier.pop(0)
            for child in self.cag.dag.get(parent, []):
                model = self.models.get((parent, child))
                if model is None or parent not in predicted:
                    continue

                parent_value = predicted[parent]
                child_estimate = float(model.predict([[parent_value]])[0])

                if child in predicted:
                    predicted[child] = 0.5 * (predicted[child] + child_estimate)
                else:
                    predicted[child] = child_estimate
                    frontier.append(child)

        return predicted

    def anomaly_score(self, row):
        scores = {}
        for m in METRICS:
            z = abs(row[m] - self.baseline[m]["mean"]) / self.baseline[m]["std"]
            scores[m] = z
        return scores

    def propagate_from_baseline(self, row, root_metric):
        """Counterfactual propagation: set root_metric to its healthy mean,
        then propagate forward through the DAG using fitted edge models.
        Returns dict of predicted descendant values under the restore.
        """
        predicted = {root_metric: self.baseline[root_metric]["mean"]}
        frontier = [root_metric]
        while frontier:
            parent = frontier.pop(0)
            for child in self.cag.dag.get(parent, []):
                model = self.models.get((parent, child))
                if model is None or parent not in predicted:
                    continue
                parent_value = predicted[parent]
                child_estimate = float(model.predict([[parent_value]])[0])
                if child in predicted:
                    predicted[child] = 0.5 * (predicted[child] + child_estimate)
                else:
                    predicted[child] = child_estimate
                    frontier.append(child)
        return predicted

    def simulate_intervention(self, row, candidate):
        """Counterfactual improvement score: restore candidate to baseline,
        propagate forward, sum the predicted reduction in downstream
        anomaly. Higher = better candidate.
        """
        predicted_under_restore = self.propagate_from_baseline(row, candidate)
        descendants = list(self.cag.descendants.get(candidate, set()))

        root_z = abs(
            row[candidate] - self.baseline[candidate]["mean"]
        ) / self.baseline[candidate]["std"]

        if not descendants:
            return root_z

        improvement = 0.0
        for d in descendants:
            parent_r2_values = [
                self.edge_r2.get((p, d), 0.0)
                for p in self.cag.parents.get(d, [])
            ]

            if not parent_r2_values:
                continue

            node_weight = max(parent_r2_values)

            if node_weight < 0.05:
                continue

            obs = float(row[d])
            std = self.baseline[d]["std"]
            mean_d = self.baseline[d]["mean"]
            est_under_restore = predicted_under_restore.get(d, obs)

            anomaly_observed = abs(obs - mean_d) / std
            anomaly_after_restore = abs(est_under_restore - mean_d) / std
            improvement += node_weight * max(
                0.0, anomaly_observed - anomaly_after_restore
            )

        return improvement

    def predict(self, X):
        preds = []

        for _, row in X.iterrows():
            root_scores = self.root_anomaly_scores(row)

            # Healthy gating
            if max(root_scores.values()) < self.healthy_threshold:
                preds.append(0)
                continue

            # Sort roots by anomaly (descending)
            sorted_roots = sorted(
                self.cag.root_metrics,
                key=lambda r: root_scores[r],
                reverse=True
            )

            top1 = sorted_roots[0]
            top2 = sorted_roots[1]

            z1 = root_scores[top1]
            z2 = root_scores[top2]

            # Margin threshold for ambiguity
            margin = 0.25

            # If anomaly clearly dominant, choose directly
            if (z1 - z2) > margin:
                preds.append(self.metric_to_fault(top1))
                continue

            # Otherwise refine between top-2 by counterfactual improvement.
            # Higher improvement = candidate's restore explains more downstream
            # anomaly = better root cause.
            score1 = self.simulate_intervention(row, top1)
            score2 = self.simulate_intervention(row, top2)

            if score1 >= score2:
                preds.append(self.metric_to_fault(top1))
            else:
                preds.append(self.metric_to_fault(top2))

        return np.array(preds)

    def predict_with_margin(self, X, margin):
        preds = []
        structure_invoked = 0
        total_nonhealthy = 0

        for _, row in X.iterrows():
            root_scores = self.root_anomaly_scores(row)

            if max(root_scores.values()) < self.healthy_threshold:
                preds.append(0)
                continue

            total_nonhealthy += 1

            sorted_roots = sorted(
                self.cag.root_metrics,
                key=lambda r: root_scores[r],
                reverse=True
            )

            top1 = sorted_roots[0]
            top2 = sorted_roots[1]

            z1 = root_scores[top1]
            z2 = root_scores[top2]

            if (z1 - z2) > margin:
                preds.append(self.metric_to_fault(top1))
            else:
                structure_invoked += 1

                score1 = self.simulate_intervention(row, top1)
                score2 = self.simulate_intervention(row, top2)

                if score1 >= score2:
                    preds.append(self.metric_to_fault(top1))
                else:
                    preds.append(self.metric_to_fault(top2))

        usage_rate = (
            structure_invoked / total_nonhealthy
            if total_nonhealthy > 0 else 0.0
        )

        return np.array(preds), usage_rate

    def residual_diagnostics(self, X_eval):
        diagnostics = []
        for (parent, child), model in self.models.items():
            Xp = X_eval[[parent]].values
            yc = X_eval[child].values
            yp = model.predict(Xp)
            resid = yc - yp

            s = skew(resid)
            k = kurtosis(resid)
            nonlinear = (abs(s) > 1.0) or (abs(k) > 3.0)
            diagnostics.append({
                'edge': f"{parent}->{child}",
                'skew': float(s),
                'kurtosis': float(k),
                'nonlinear': nonlinear
            })
        return diagnostics

    def metric_to_fault(self, metric):
        return self.cag.root_to_fault.get(metric, 0)

# ===============================
# INCIDENT-LEVEL AGGREGATION
# ===============================

incident_fault_cardinality = df.groupby('incident_id')['fault_type'].nunique()
if (incident_fault_cardinality > 1).any():
    raise ValueError("Each incident_id must map to exactly one fault_type for incident-level evaluation")

incident_df = (
    df.groupby('incident_id', as_index=False)
      .agg({**{m: 'median' for m in METRICS}, 'fault_type': 'first'})
)

print("\nIncident-Level Shape:", incident_df.shape)
print("Fault Distribution (incident-level):")
print(incident_df['fault_type'].value_counts().sort_index())

# ===============================
# STRATIFIED SPLIT (INCIDENT LEVEL)
# ===============================

X = incident_df[METRICS]
y = incident_df["fault_type"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=GLOBAL_SEED,
    stratify=y
)

healthy_mask = (y_train == 0)
X_healthy = X_train[healthy_mask].copy()

# Train only on healthy
X_train_healthy = X_healthy
if X_train_healthy.shape[0] == 0:
    raise ValueError("No healthy samples available in training split. Increase healthy baseline proportion.")

cag = DatacenterCAG()

audit_rows = []
edge_r2_map = {}
for parent, children in cag.dag.items():
    for child in children:
        edge_df = X_healthy[[parent, child]].dropna()

        if edge_df.shape[0] < 2:
            coef = float('nan')
            r2 = float('nan')
            corr = float('nan')
            residual_std = float('nan')
        else:
            Xp = edge_df[[parent]].values
            yc = edge_df[child].values

            lr = LinearRegression()
            lr.fit(Xp, yc)
            yp = lr.predict(Xp)

            coef = float(lr.coef_[0])
            r2 = float(lr.score(Xp, yc))
            corr = float(edge_df[parent].corr(edge_df[child]))
            residual_std = float(np.std(yc - yp))

        audit_rows.append({
            'parent': parent,
            'child': child,
            'coef': coef,
            'r2': r2,
            'corr': corr,
            'residual_std': residual_std
        })
        edge_r2_map[(parent, child)] = float(r2)

audit_df = pd.DataFrame(audit_rows).sort_values(by='r2', ascending=False, na_position='last')

print("\n=== Healthy Structural Fit Audit ===")
print("Parent -> Child | Coef | R² | Corr | Residual Std")
for row in audit_df.itertuples(index=False):
    print(
        f"{row.parent} -> {row.child} | "
        f"{row.coef:.4f} | {row.r2:.4f} | {row.corr:.4f} | {row.residual_std:.4f}"
    )

r2_values = audit_df['r2'].dropna().values
mean_r2 = float(np.mean(r2_values)) if len(r2_values) > 0 else float('nan')
median_r2 = float(np.median(r2_values)) if len(r2_values) > 0 else float('nan')
pct_r2_gt_03 = float(np.mean(r2_values > 0.3) * 100.0) if len(r2_values) > 0 else float('nan')
pct_r2_gt_05 = float(np.mean(r2_values > 0.5) * 100.0) if len(r2_values) > 0 else float('nan')

print("\n=== Healthy Structural Fit Summary ===")
print(f"Mean R² across all edges: {mean_r2:.4f}")
print(f"Median R²: {median_r2:.4f}")
print(f"% of edges with R² > 0.3: {pct_r2_gt_03:.2f}%")
print(f"% of edges with R² > 0.5: {pct_r2_gt_05:.2f}%")

# ===============================
# LINEAR SEM
# ===============================

linear_model = CausalInterventionRCA(cag, use_rf=False)
linear_model.fit(X_train_healthy)
linear_model.edge_r2 = edge_r2_map

start = time.time()
y_pred_linear = linear_model.predict(X_test)
linear_time = (time.time() - start) / len(X_test) * 1000

linear_acc = accuracy_score(y_test, y_pred_linear)

print("\nLinear SEM Top-1 Accuracy:", round(linear_acc,4))
print("Linear SEM Inference (ms/sample):", round(linear_time,3))

# ===============================
# RF SEM
# ===============================

rf_model = CausalInterventionRCA(cag, use_rf=True)
rf_model.fit(X_train_healthy)
rf_model.edge_r2 = edge_r2_map

start = time.time()
y_pred_rf = rf_model.predict(X_test)
rf_time = (time.time() - start) / len(X_test) * 1000

rf_acc = accuracy_score(y_test, y_pred_rf)

print("\nRF SEM Top-1 Accuracy:", round(rf_acc,4))
print("RF SEM Inference (ms/sample):", round(rf_time,3))

print("\n=== Margin Sensitivity Sweep ===")

margins = [0.25, 0.10, 0.05, 0.01]

for m in margins:
    preds_lin, usage_lin = linear_model.predict_with_margin(X_test, m)
    preds_rf, usage_rf = rf_model.predict_with_margin(X_test, m)

    acc_lin = accuracy_score(y_test, preds_lin)
    acc_rf = accuracy_score(y_test, preds_rf)

    print(f"\nMargin: {m}")
    print(f"Linear Top-1: {acc_lin:.4f}, Structure Used: {usage_lin*100:.2f}%")
    print(f"RF Top-1: {acc_rf:.4f}, Structure Used: {usage_rf*100:.2f}%")

print("\n=== Ambiguous Case Structural Impact (Margin = 0.25) ===")

margin = 0.25

ambiguous_indices = []
anomaly_only_preds = []
hybrid_preds = []
true_labels = []

for idx, row in X_test.iterrows():
    root_scores = linear_model.root_anomaly_scores(row)

    if max(root_scores.values()) < linear_model.healthy_threshold:
        continue

    sorted_roots = sorted(
        linear_model.cag.root_metrics,
        key=lambda r: root_scores[r],
        reverse=True
    )

    top1 = sorted_roots[0]
    top2 = sorted_roots[1]

    z1 = root_scores[top1]
    z2 = root_scores[top2]

    if (z1 - z2) <= margin:
        ambiguous_indices.append(idx)

        # Anomaly-only prediction
        anomaly_pred = linear_model.metric_to_fault(top1)
        anomaly_only_preds.append(anomaly_pred)

        # Hybrid prediction (using existing predict logic)
        hybrid_pred = linear_model.predict(pd.DataFrame([row]))[0]
        hybrid_preds.append(hybrid_pred)

        true_labels.append(y_test.loc[idx])

if len(ambiguous_indices) > 0:
    anomaly_acc = accuracy_score(true_labels, anomaly_only_preds)
    hybrid_acc = accuracy_score(true_labels, hybrid_preds)

    print(f"Ambiguous Samples: {len(ambiguous_indices)}")
    print(f"Anomaly-only Top-1 (ambiguous subset): {anomaly_acc:.4f}")
    print(f"Hybrid Top-1 (ambiguous subset): {hybrid_acc:.4f}")
else:
    print("No ambiguous samples found at this margin.")

print("\n=== Root Anomaly Compression Experiment ===")

compression_factors = [1.0, 0.7, 0.5, 0.3]
margin = 0.25

for lam in compression_factors:
    X_mod = X_test.copy()

    # Compress only root metrics
    for root in linear_model.cag.root_metrics:
        mu = linear_model.baseline[root]["mean"]
        X_mod[root] = mu + lam * (X_mod[root] - mu)

    anomaly_only_preds = []
    hybrid_preds = []
    ambiguous_count = 0
    total_nonhealthy = 0

    for idx, row in X_mod.iterrows():
        root_scores = linear_model.root_anomaly_scores(row)

        if max(root_scores.values()) < linear_model.healthy_threshold:
            anomaly_only_preds.append(0)
            hybrid_preds.append(0)
            continue

        total_nonhealthy += 1

        sorted_roots = sorted(
            linear_model.cag.root_metrics,
            key=lambda r: root_scores[r],
            reverse=True
        )

        top1 = sorted_roots[0]
        top2 = sorted_roots[1]

        z1 = root_scores[top1]
        z2 = root_scores[top2]

        # Anomaly-only
        anomaly_only_preds.append(linear_model.metric_to_fault(top1))

        # Hybrid
        if (z1 - z2) > margin:
            hybrid_preds.append(linear_model.metric_to_fault(top1))
        else:
            ambiguous_count += 1
            score1 = linear_model.simulate_intervention(row, top1)
            score2 = linear_model.simulate_intervention(row, top2)

            if score1 >= score2:
                hybrid_preds.append(linear_model.metric_to_fault(top1))
            else:
                hybrid_preds.append(linear_model.metric_to_fault(top2))

    anomaly_acc = accuracy_score(y_test, anomaly_only_preds)
    hybrid_acc = accuracy_score(y_test, hybrid_preds)

    ambiguity_rate = (
        ambiguous_count / total_nonhealthy
        if total_nonhealthy > 0 else 0.0
    )

    print(f"\nλ = {lam}")
    print(f"Anomaly-only Top-1: {anomaly_acc:.4f}")
    print(f"Hybrid Top-1: {hybrid_acc:.4f}")
    print(f"Ambiguity Rate: {ambiguity_rate*100:.2f}%")

# ===============================
# RF CLASSIFIER UPPER BOUND
# ===============================

rf_cls = RandomForestClassifier(
    n_estimators=60,
    max_depth=4,
    max_features=4,
    min_samples_leaf=2,
    random_state=GLOBAL_SEED,
    n_jobs=-1
)
rf_cls.fit(X_train, y_train)
y_pred_rf_cls = rf_cls.predict(X_test)
rf_cls_acc = accuracy_score(y_test, y_pred_rf_cls)

print("\nRF Classifier Top-1 (Upper Bound):", round(rf_cls_acc,4))

# ===============================
# RESIDUAL DIAGNOSTICS
# ===============================

print("\nResidual Diagnostics (Linear SEM)")
linear_diag = linear_model.residual_diagnostics(X_test)
for row in linear_diag:
    flag = " <-- Non-linearity indicator" if row['nonlinear'] else ""
    print(f"{row['edge']}: skew={row['skew']:.2f}, kurtosis={row['kurtosis']:.2f}{flag}")

print("\nResidual Diagnostics (RF SEM)")
rf_diag = rf_model.residual_diagnostics(X_test)
for row in rf_diag:
    flag = " <-- Non-linearity indicator" if row['nonlinear'] else ""
    print(f"{row['edge']}: skew={row['skew']:.2f}, kurtosis={row['kurtosis']:.2f}{flag}")

linear_flags = sum(d['nonlinear'] for d in linear_diag)
rf_flags = sum(d['nonlinear'] for d in rf_diag)

print("\nResidual Non-linearity Flags:")
print(f"Linear SEM: {linear_flags}/{len(linear_diag)} edges flagged")
print(f"RF SEM: {rf_flags}/{len(rf_diag)} edges flagged")

# ===============================
# ADDITIONAL MANUSCRIPT MODULES
# ===============================

ROOT_METRICS = list(cag.root_to_fault.keys())

def _predict_ranked_method(self, X):
    ranked_output = []
    for _, row in X.iterrows():
        root_scores = self.root_anomaly_scores(row)
        sorted_roots = sorted(
            self.cag.root_metrics,
            key=lambda r: root_scores[r],
            reverse=True
        )

        if max(root_scores.values()) < self.healthy_threshold:
            ranked_output.append(sorted_roots)
            continue

        top1 = sorted_roots[0]
        top2 = sorted_roots[1]
        z1 = root_scores[top1]
        z2 = root_scores[top2]
        margin = 0.25

        if (z1 - z2) > margin:
            best_root = top1
        else:
            score1 = self.simulate_intervention(row, top1)
            score2 = self.simulate_intervention(row, top2)
            best_root = top1 if score1 >= score2 else top2

        remaining = [r for r in sorted_roots if r != best_root]
        ranked_output.append([best_root] + remaining)

    return ranked_output

CausalInterventionRCA.predict_ranked = _predict_ranked_method

def ranked_to_top1_predictions(model, X_eval, ranked_roots):
    preds = []
    for (_, row), ranking in zip(X_eval.iterrows(), ranked_roots):
        root_scores = model.root_anomaly_scores(row)
        if max(root_scores.values()) < model.healthy_threshold:
            preds.append(0)
        else:
            preds.append(model.metric_to_fault(ranking[0]))
    return np.array(preds)

def compute_topk_and_mrr(y_true, ranked_roots, root_to_fault, k=3):
    hits = []
    reciprocal_ranks = []

    for yt, ranking in zip(y_true, ranked_roots):
        if yt == 0:
            continue

        ranked_faults = [root_to_fault[r] for r in ranking]
        topk = ranked_faults[:k]
        hits.append(1.0 if yt in topk else 0.0)

        rr = 0.0
        for idx, pred_fault in enumerate(ranked_faults, start=1):
            if pred_fault == yt:
                rr = 1.0 / idx
                break
        reciprocal_ranks.append(rr)

    topk_acc = float(np.mean(hits)) if hits else 0.0
    mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
    return topk_acc, mrr

class PearsonBaseline:
    def __init__(self, root_to_fault):
        self.root_to_fault = root_to_fault
        self.root_metrics = list(root_to_fault.keys())
        self.root_correlations = {}
        self.correlation_weight = {}
        self.baseline_mean = {}
        self.baseline_std = {}
        self.ranked_roots = []

    def fit(self, X_train_data, y_train_data):
        y_train_series = pd.Series(y_train_data).reset_index(drop=True)
        X_train_df = X_train_data.reset_index(drop=True)

        for root in self.root_metrics:
            root_values = X_train_df[root]
            mapped_fault = self.root_to_fault[root]
            y_binary = (y_train_series == mapped_fault).astype(float)
            corr = root_values.corr(y_binary)
            self.root_correlations[root] = float(abs(corr) if pd.notna(corr) else 0.0)
            self.correlation_weight[root] = self.root_correlations[root]
            self.baseline_mean[root] = float(root_values.mean())
            self.baseline_std[root] = float(root_values.std() + 1e-9)

        self.ranked_roots = sorted(self.root_metrics, key=lambda r: self.root_correlations[r], reverse=True)

    def predict_ranked(self, X_eval):
        ranked_output = []
        for _, row in X_eval.iterrows():
            weighted_scores = {}
            for root in self.root_metrics:
                z = abs(float(row[root]) - self.baseline_mean[root]) / self.baseline_std[root]
                weighted_scores[root] = z * self.correlation_weight[root]
            ranked_roots = sorted(weighted_scores, key=weighted_scores.get, reverse=True)
            ranked_output.append(ranked_roots)
        return ranked_output

    def predict(self, X_eval):
        ranked = self.predict_ranked(X_eval)
        return np.array([self.root_to_fault[r[0]] for r in ranked])

class PageRankBaseline:
    def __init__(self, root_to_fault, alpha=0.85):
        self.root_to_fault = root_to_fault
        self.root_metrics = list(root_to_fault.keys())
        self.alpha = alpha
        self.pagerank_scores = {}
        self.ranked_roots = []

    def fit(self, X_train_data):
        corr_matrix = X_train_data[self.root_metrics].corr().abs().fillna(0.0)
        graph = nx.Graph()
        graph.add_nodes_from(self.root_metrics)

        for i, src in enumerate(self.root_metrics):
            for dst in self.root_metrics[i + 1:]:
                weight = float(corr_matrix.loc[src, dst])
                if weight > 0.05:
                    graph.add_edge(src, dst, weight=weight)

        self.pagerank_scores = nx.pagerank(graph, alpha=self.alpha, weight='weight')
        self.ranked_roots = sorted(self.root_metrics, key=lambda r: self.pagerank_scores[r], reverse=True)

    def predict_ranked(self, X_eval):
        return [self.ranked_roots[:] for _ in range(len(X_eval))]

    def predict(self, X_eval):
        ranked = self.predict_ranked(X_eval)
        return np.array([self.root_to_fault[r[0]] for r in ranked])

def mcnemar_test(y_true, pred_a, pred_b):
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)

    a_correct = pred_a == y_true
    b_correct = pred_b == y_true

    both_correct = int(np.sum(a_correct & b_correct))
    a_only = int(np.sum(a_correct & (~b_correct)))
    b_only = int(np.sum((~a_correct) & b_correct))
    neither = int(np.sum((~a_correct) & (~b_correct)))

    table = [[both_correct, a_only], [b_only, neither]]
    result = mcnemar(table, exact=False, correction=True)
    return float(result.statistic), float(result.pvalue)

def compute_mean_rank(y_true, ranked_roots, root_to_fault):
    ranks = []
    for yt, ranking in zip(y_true, ranked_roots):
        if yt == 0:
            continue
        ranked_faults = [root_to_fault[r] for r in ranking]
        if yt in ranked_faults:
            ranks.append(ranked_faults.index(yt) + 1)
    return float(np.mean(ranks)) if ranks else float('nan')

def top1_fault_only_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(y_true[mask] == y_pred[mask]))

# Ranked outputs for SEM models
linear_ranked = linear_model.predict_ranked(X_test)
rf_ranked = rf_model.predict_ranked(X_test)

# Ensure Top-1 from ranked pipeline exactly matches existing behavior
linear_top1_from_ranked = ranked_to_top1_predictions(linear_model, X_test, linear_ranked)
rf_top1_from_ranked = ranked_to_top1_predictions(rf_model, X_test, rf_ranked)

if not np.array_equal(linear_top1_from_ranked, y_pred_linear):
    raise AssertionError("Linear ranked Top-1 predictions do not match existing predict() outputs")
if not np.array_equal(rf_top1_from_ranked, y_pred_rf):
    raise AssertionError("RF ranked Top-1 predictions do not match existing predict() outputs")

linear_top3, linear_mrr = compute_topk_and_mrr(y_test.values, linear_ranked, cag.root_to_fault, k=3)
rf_top3, rf_mrr = compute_topk_and_mrr(y_test.values, rf_ranked, cag.root_to_fault, k=3)

print("\nRanked Metrics (fault-only Top-3/MRR)")
print("Linear SEM Top-1:", round(linear_acc, 4))
print("Linear SEM Top-3:", round(linear_top3, 4))
print("Linear SEM MRR:", round(linear_mrr, 4))
print("RF SEM Top-1:", round(rf_acc, 4))
print("RF SEM Top-3:", round(rf_top3, 4))
print("RF SEM MRR:", round(rf_mrr, 4))

# Pearson baseline
pearson_baseline = PearsonBaseline(cag.root_to_fault)
pearson_baseline.fit(X_train, y_train)
pearson_pred = pearson_baseline.predict(X_test)
pearson_ranked = pearson_baseline.predict_ranked(X_test)
pearson_top1 = top1_fault_only_accuracy(y_test.values, pearson_pred)
pearson_top3, pearson_mrr = compute_topk_and_mrr(y_test.values, pearson_ranked, cag.root_to_fault, k=3)

# PageRank baseline
pagerank_baseline = PageRankBaseline(cag.root_to_fault, alpha=0.85)
pagerank_baseline.fit(X_train)
pagerank_pred = pagerank_baseline.predict(X_test)
pagerank_ranked = pagerank_baseline.predict_ranked(X_test)
pagerank_top1 = top1_fault_only_accuracy(y_test.values, pagerank_pred)
pagerank_top3, pagerank_mrr = compute_topk_and_mrr(y_test.values, pagerank_ranked, cag.root_to_fault, k=3)

print("\nCorrelation / Graph Baselines")
print("Pearson Baseline Top-1:", round(pearson_top1, 4))
print("Pearson Baseline Top-3:", round(pearson_top3, 4))
print("Pearson Baseline MRR:", round(pearson_mrr, 4))
print("PageRank Baseline Top-1:", round(pagerank_top1, 4))
print("PageRank Baseline Top-3:", round(pagerank_top3, 4))
print("PageRank Baseline MRR:", round(pagerank_mrr, 4))

print("\n=== BASELINE DEBUG SUMMARY ===")
print("Root correlations:", {k: round(v, 6) for k, v in pearson_baseline.root_correlations.items()})
print("PageRank scores:", {k: round(v, 6) for k, v in pagerank_baseline.pagerank_scores.items()})
print("Pearson Top-1:", round(pearson_top1, 4))
print("PageRank Top-1:", round(pagerank_top1, 4))
print("Pearson Top-1 != 0.1111:", abs(pearson_top1 - 0.1111) > 1e-6)
print("PageRank Top-1 != 0.1111:", abs(pagerank_top1 - 0.1111) > 1e-6)
print("Pearson Top-1 > random (0.17):", pearson_top1 > 0.17)
print("PageRank Top-1 > random (0.17):", pagerank_top1 > 0.17)

# McNemar statistical tests
mc_linear_vs_pearson_stat, mc_linear_vs_pearson_p = mcnemar_test(y_test.values, y_pred_linear, pearson_pred)
mc_linear_vs_pagerank_stat, mc_linear_vs_pagerank_p = mcnemar_test(y_test.values, y_pred_linear, pagerank_pred)
mc_rf_vs_pearson_stat, mc_rf_vs_pearson_p = mcnemar_test(y_test.values, y_pred_rf, pearson_pred)
mc_rf_vs_pagerank_stat, mc_rf_vs_pagerank_p = mcnemar_test(y_test.values, y_pred_rf, pagerank_pred)

print("\nMcNemar Tests")
print(f"Linear SEM vs Pearson: statistic={mc_linear_vs_pearson_stat:.4f}, p={mc_linear_vs_pearson_p:.6f}")
print(f"Linear SEM vs PageRank: statistic={mc_linear_vs_pagerank_stat:.4f}, p={mc_linear_vs_pagerank_p:.6f}")
print(f"RF SEM vs Pearson: statistic={mc_rf_vs_pearson_stat:.4f}, p={mc_rf_vs_pearson_p:.6f}")
print(f"RF SEM vs PageRank: statistic={mc_rf_vs_pagerank_stat:.4f}, p={mc_rf_vs_pagerank_p:.6f}")

# MTTR simulation (analytical)
manual_baseline = 4.2
scm_expected = compute_mean_rank(y_test.values, linear_ranked, cag.root_to_fault)
mttr_reduction_percent = ((manual_baseline - scm_expected) / manual_baseline) * 100.0

print("\nMTTR Simulation")
print("Manual baseline investigations:", round(manual_baseline, 4))
print("SCM expected investigations:", round(scm_expected, 4))
print("MTTR reduction (%):", round(mttr_reduction_percent, 2))

# Results artifact export
results_payload = {
    "dataset_version_hash": hashlib.sha256("DATACENTER_RCA_V10_DETERMINISTIC".encode()).hexdigest(),
    "evaluation_version_hash": VERSION_HASH,
    "random_seed": GLOBAL_SEED,
    "top1_linear": float(linear_acc),
    "top3_linear": float(linear_top3),
    "mrr_linear": float(linear_mrr),
    "top1_rf": float(rf_acc),
    "top3_rf": float(rf_top3),
    "mrr_rf": float(rf_mrr),
    "top1_pearson": float(pearson_top1),
    "top1_pagerank": float(pagerank_top1),
    "mcnemar_pvalues": {
        "linear_vs_pearson": float(mc_linear_vs_pearson_p),
        "linear_vs_pagerank": float(mc_linear_vs_pagerank_p),
        "rf_vs_pearson": float(mc_rf_vs_pearson_p),
        "rf_vs_pagerank": float(mc_rf_vs_pagerank_p)
    },
    "mttr_reduction_percent": float(mttr_reduction_percent),
    "linear_inference_ms": float(linear_time),
    "rf_inference_ms": float(rf_time)
}

with open("results_datacenter_rca.json", "w", encoding="utf-8") as f:
    json.dump(results_payload, f, indent=2)

print("\n=== Cross-Seed Stability Evaluation ===")

seeds = [20260227, 42, 7, 1234, 999]

linear_scores = []
rf_scores = []
pearson_scores = []
ambiguous_improvements = []

for seed in seeds:
    print(f"\nRunning seed: {seed}")

    # Regenerate dataset
    subprocess.run([
        sys.executable,
        "datacenter_dataset_generator.py",
        "--seed",
        str(seed)
    ], check=True)

    # Reload dataset
    df_seed = pd.read_csv("datacenter_dataset.csv")

    # Incident aggregation (reuse existing logic)
    df_incident = (
        df_seed.groupby('incident_id', as_index=False)
              .agg({**{m: 'median' for m in METRICS}, 'fault_type': 'first'})
    )

    X_seed = df_incident[METRICS]
    y_seed = df_incident["fault_type"]

    X_train_seed, X_test_seed, y_train_seed, y_test_seed = train_test_split(
        X_seed, y_seed,
        test_size=0.3,
        stratify=y_seed,
        random_state=seed
    )

    X_train_seed_healthy = X_train_seed[y_train_seed == 0]

    # Train SEM models (reuse existing classes)
    lin_model = CausalInterventionRCA(cag, use_rf=False)
    lin_model.fit(X_train_seed_healthy)

    rf_model_seed = CausalInterventionRCA(cag, use_rf=True)
    rf_model_seed.fit(X_train_seed_healthy)

    edge_r2_seed = {}
    for parent, children in cag.dag.items():
        for child in children:
            edge_df = X_train_seed_healthy[[parent, child]].dropna()
            if edge_df.shape[0] < 2:
                r2_val = float('nan')
            else:
                Xp = edge_df[[parent]].values
                yc = edge_df[child].values
                lr_seed = LinearRegression()
                lr_seed.fit(Xp, yc)
                r2_val = float(lr_seed.score(Xp, yc))
            edge_r2_seed[(parent, child)] = r2_val

    lin_model.edge_r2 = edge_r2_seed
    rf_model_seed.edge_r2 = edge_r2_seed

    # Pearson baseline (reuse existing)
    pearson_model = PearsonBaseline(cag.root_to_fault)
    pearson_model.fit(X_train_seed, y_train_seed)

    # Hybrid predictions
    lin_preds = lin_model.predict(X_test_seed)
    rf_preds_seed = rf_model_seed.predict(X_test_seed)
    pearson_preds = pearson_model.predict(X_test_seed)

    linear_scores.append(accuracy_score(y_test_seed, lin_preds))
    rf_scores.append(accuracy_score(y_test_seed, rf_preds_seed))
    pearson_scores.append(accuracy_score(y_test_seed, pearson_preds))

    # Ambiguous improvement
    margin = 0.25
    ambiguous_count = 0
    anomaly_correct = 0
    hybrid_correct = 0

    for idx, row in X_test_seed.iterrows():
        root_scores = lin_model.root_anomaly_scores(row)

        if max(root_scores.values()) < lin_model.healthy_threshold:
            continue

        sorted_roots = sorted(
            lin_model.cag.root_metrics,
            key=lambda r: root_scores[r],
            reverse=True
        )

        top1, top2 = sorted_roots[:2]
        z1, z2 = root_scores[top1], root_scores[top2]

        if (z1 - z2) <= margin:
            ambiguous_count += 1

            anomaly_pred = lin_model.metric_to_fault(top1)
            hybrid_pred = lin_model.predict(pd.DataFrame([row]))[0]

            if anomaly_pred == y_test_seed.loc[idx]:
                anomaly_correct += 1
            if hybrid_pred == y_test_seed.loc[idx]:
                hybrid_correct += 1

    if ambiguous_count > 0:
        improvement = (
            hybrid_correct / ambiguous_count
            - anomaly_correct / ambiguous_count
        )
        ambiguous_improvements.append(improvement)
    else:
        ambiguous_improvements.append(0.0)

# Print summary
print("\n=== Cross-Seed Summary ===")
print(f"Linear Top-1: {np.mean(linear_scores):.4f} ± {np.std(linear_scores):.4f}")
print(f"RF Top-1: {np.mean(rf_scores):.4f} ± {np.std(rf_scores):.4f}")
print(f"Pearson Top-1: {np.mean(pearson_scores):.4f} ± {np.std(pearson_scores):.4f}")
print(f"Ambiguous Improvement: {np.mean(ambiguous_improvements):.4f} ± {np.std(ambiguous_improvements):.4f}")

print("\n=== Cross-Seed Robustness Consolidation ===")

seeds = [20260227, 42, 7, 1234, 999]

clean_linear = []
clean_rf = []
pearson_clean = []

missing_linear = []
noise_linear = []
dag_linear = []

for seed in seeds:

    subprocess.run([
        sys.executable,
        "datacenter_dataset_generator.py",
        "--seed",
        str(seed)
    ], check=True)

    df_seed = pd.read_csv("datacenter_dataset.csv")

    df_incident = (
        df_seed.groupby('incident_id', as_index=False)
        .agg({**{m: 'median' for m in METRICS}, 'fault_type': 'first'})
    )

    X_seed = df_incident[METRICS]
    y_seed = df_incident["fault_type"]

    X_train_seed, X_test_seed, y_train_seed, y_test_seed = train_test_split(
        X_seed, y_seed,
        test_size=0.3,
        stratify=y_seed,
        random_state=seed
    )

    X_train_healthy = X_train_seed[y_train_seed == 0]

    lin = CausalInterventionRCA(cag, use_rf=False)
    lin.fit(X_train_healthy)

    rf = CausalInterventionRCA(cag, use_rf=True)
    rf.fit(X_train_healthy)

    pearson = PearsonBaseline(cag.root_to_fault)
    pearson.fit(X_train_seed, y_train_seed)

    # Clean accuracy
    clean_linear.append(accuracy_score(y_test_seed, lin.predict(X_test_seed)))
    clean_rf.append(accuracy_score(y_test_seed, rf.predict(X_test_seed)))
    pearson_clean.append(accuracy_score(y_test_seed, pearson.predict(X_test_seed)))

    # Missing Data 20%
    X_missing = X_test_seed.copy()
    rng = np.random.default_rng(seed)
    mask = rng.random(X_missing.shape) < 0.2
    for i, m in enumerate(METRICS):
        mean_val = lin.baseline[m]["mean"]
        X_missing.loc[mask[:, i], m] = mean_val

    missing_linear.append(accuracy_score(y_test_seed, lin.predict(X_missing)))

    # Noise 0.06
    X_noise = X_test_seed.copy()
    for m in METRICS:
        std = lin.baseline[m]["std"]
        X_noise[m] += rng.normal(0, 0.06 * std, size=len(X_noise))

    noise_linear.append(accuracy_score(y_test_seed, lin.predict(X_noise)))

    # DAG removal 20%
    edges = list(cag.dag.items())
    flat_edges = [(p, c) for p, children in edges for c in children]
    k = int(len(flat_edges) * 0.2)
    removed = set(flat_edges[:k])

    cag_mod = DatacenterCAG()
    for p, c in removed:
        if c in cag_mod.dag.get(p, []):
            cag_mod.dag[p].remove(c)

    lin_mod = CausalInterventionRCA(cag_mod, use_rf=False)
    lin_mod.fit(X_train_healthy)

    dag_linear.append(accuracy_score(y_test_seed, lin_mod.predict(X_test_seed)))

# Summary
def summarize(arr):
    return f"{np.mean(arr):.4f} ± {np.std(arr):.4f}"

print("\n--- CLEAN ---")
print("Linear:", summarize(clean_linear))
print("RF:", summarize(clean_rf))
print("Pearson:", summarize(pearson_clean))

print("\n--- Missing (20%) ---")
print("Linear:", summarize(missing_linear))

print("\n--- Noise (0.06) ---")
print("Linear:", summarize(noise_linear))

print("\n--- DAG Removal (20%) ---")
print("Linear:", summarize(dag_linear))

# Consolidation artifact export (non-destructive)
consolidation_section = {
    "seeds": seeds,
    "margin": 0.25,
    "missing_rate": 0.2,
    "noise_std": 0.06,
    "dag_removal_rate": 0.2,
    "clean": {
        "linear": {"mean": float(np.mean(clean_linear)), "std": float(np.std(clean_linear))},
        "rf": {"mean": float(np.mean(clean_rf)), "std": float(np.std(clean_rf))},
        "pearson": {"mean": float(np.mean(pearson_clean)), "std": float(np.std(pearson_clean))}
    },
    "missing_20": {
        "linear": {"mean": float(np.mean(missing_linear)), "std": float(np.std(missing_linear))}
    },
    "noise_006": {
        "linear": {"mean": float(np.mean(noise_linear)), "std": float(np.std(noise_linear))}
    },
    "dag_removal_20": {
        "linear": {"mean": float(np.mean(dag_linear)), "std": float(np.std(dag_linear))}
    }
}

with open("results_datacenter_rca.json", "r", encoding="utf-8") as f:
    existing_results = json.load(f)

extended_results = dict(existing_results)
extended_results["cross_seed_robustness_consolidation"] = consolidation_section

with open("results_datacenter_rca_with_consolidation.json", "w", encoding="utf-8") as f:
    json.dump(extended_results, f, indent=2)

print("\nConsolidation artifact written: results_datacenter_rca_with_consolidation.json")
