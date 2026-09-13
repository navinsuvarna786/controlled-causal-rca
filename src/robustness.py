#!/usr/bin/env python3

import json
import copy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression

from datacenter_eval import (
	DatacenterCAG,
	CausalInterventionRCA,
	GLOBAL_SEED,
	METRICS,
)


DATA_PATH = "datacenter_dataset.csv"
MASK_RATES = [0.0, 0.1, 0.2, 0.3]
NOISE_LEVELS = [0.045, 0.06, 0.08]
REMOVAL_LEVELS = [0.1, 0.2]
STRUCTURAL_MODES = ["full", "root_only", "empty"]
OUT_PATH = "robustness.json"


def compute_topk_and_mrr(y_true, ranked_roots, root_to_fault, k=3):
	hits = []
	reciprocal_ranks = []

	for yt, ranking in zip(y_true, ranked_roots):
		if yt == 0:
			continue

		ranked_faults = [root_to_fault[r] for r in ranking]
		hits.append(1.0 if yt in ranked_faults[:k] else 0.0)

		rr = 0.0
		for idx, pred_fault in enumerate(ranked_faults, start=1):
			if pred_fault == yt:
				rr = 1.0 / idx
				break
		reciprocal_ranks.append(rr)

	topk_acc = float(np.mean(hits)) if hits else 0.0
	mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
	return topk_acc, mrr


def prepare_data():
	df = pd.read_csv(DATA_PATH)

	incident_df = (
		df.groupby('incident_id', as_index=False)
		  .agg({**{m: 'median' for m in METRICS}, 'fault_type': 'first'})
	)

	X = incident_df[METRICS]
	y = incident_df["fault_type"]

	X_train, X_test, y_train, y_test = train_test_split(
		X, y,
		test_size=0.3,
		random_state=GLOBAL_SEED,
		stratify=y
	)

	X_train_healthy = X_train[y_train == 0]
	if X_train_healthy.shape[0] == 0:
		raise ValueError("No healthy samples available in training split.")

	return X_train, X_test, y_train, y_test, X_train_healthy


def _compute_edge_r2_map(cag, X_train_healthy):
	edge_r2_map = {}
	for parent, children in cag.dag.items():
		for child in children:
			edge_df = X_train_healthy[[parent, child]].dropna()
			if edge_df.shape[0] < 2:
				r2_val = 0.0
			else:
				Xp = edge_df[[parent]].values
				yc = edge_df[child].values
				lr = LinearRegression()
				lr.fit(Xp, yc)
				r2_val = float(lr.score(Xp, yc))
			edge_r2_map[(parent, child)] = r2_val if np.isfinite(r2_val) else 0.0
	return edge_r2_map


def _rebuild_parents_and_descendants(cag):
	cag.parents = {m: [] for m in METRICS}
	for parent, children in cag.dag.items():
		for child in children:
			cag.parents[child].append(parent)

	cag.descendants = {m: cag._get_descendants(m) for m in METRICS}
	return cag


def _fit_model_with_edge_r2(cag, use_rf, X_train_healthy):
	model = CausalInterventionRCA(cag, use_rf=use_rf)
	model.fit(X_train_healthy)
	model.edge_r2 = _compute_edge_r2_map(cag, X_train_healthy)
	return model


def evaluate_with_missing(mask_rate, X_test, y_test, linear_model, rf_model):
	X_masked = X_test.copy(deep=True)
	rng = np.random.default_rng(GLOBAL_SEED + int(mask_rate * 100))

	n_metrics = len(METRICS)
	n_mask = int(round(mask_rate * n_metrics))

	if n_mask > 0:
		for row_idx in X_masked.index:
			mask_cols_idx = rng.choice(n_metrics, size=n_mask, replace=False)
			for col_idx in mask_cols_idx:
				metric = METRICS[col_idx]
				X_masked.at[row_idx, metric] = linear_model.baseline[metric]["mean"]

	y_pred_linear = linear_model.predict(X_masked)
	y_pred_rf = rf_model.predict(X_masked)

	linear_top1 = float(accuracy_score(y_test, y_pred_linear))
	rf_top1 = float(accuracy_score(y_test, y_pred_rf))

	linear_ranked = linear_model.predict_ranked(X_masked)
	rf_ranked = rf_model.predict_ranked(X_masked)

	linear_top3, linear_mrr = compute_topk_and_mrr(
		y_test.values,
		linear_ranked,
		linear_model.cag.root_to_fault,
		k=3
	)
	rf_top3, rf_mrr = compute_topk_and_mrr(
		y_test.values,
		rf_ranked,
		rf_model.cag.root_to_fault,
		k=3
	)

	return {
		"linear_top1": linear_top1,
		"rf_top1": rf_top1,
		"linear_top3": float(linear_top3),
		"rf_top3": float(rf_top3),
		"linear_mrr": float(linear_mrr),
		"rf_mrr": float(rf_mrr),
	}


def _top1_from_ranked_with_threshold(X_eval, y_true, model, ranked_roots):
	preds = []
	for (_, row), ranking in zip(X_eval.iterrows(), ranked_roots):
		root_scores = model.root_anomaly_scores(row)
		if max(root_scores.values()) < model.healthy_threshold:
			preds.append(0)
		else:
			preds.append(model.metric_to_fault(ranking[0]))
	return float(accuracy_score(y_true, np.array(preds)))


def evaluate_with_noise(noise_std, X_test, y_test, linear_model, rf_model):
	rng = np.random.default_rng(GLOBAL_SEED + int(noise_std * 1000))
	X_noisy = X_test.copy(deep=True)
	X_noisy[METRICS] = X_noisy[METRICS] + rng.normal(0, noise_std, X_noisy[METRICS].shape)

	linear_ranked = linear_model.predict_ranked(X_noisy)
	rf_ranked = rf_model.predict_ranked(X_noisy)

	linear_top1 = _top1_from_ranked_with_threshold(X_noisy, y_test, linear_model, linear_ranked)
	rf_top1 = _top1_from_ranked_with_threshold(X_noisy, y_test, rf_model, rf_ranked)

	linear_top3, linear_mrr = compute_topk_and_mrr(
		y_test.values,
		linear_ranked,
		linear_model.cag.root_to_fault,
		k=3
	)
	rf_top3, rf_mrr = compute_topk_and_mrr(
		y_test.values,
		rf_ranked,
		rf_model.cag.root_to_fault,
		k=3
	)

	return {
		"linear_top1": float(linear_top1),
		"rf_top1": float(rf_top1),
		"linear_top3": float(linear_top3),
		"rf_top3": float(rf_top3),
		"linear_mrr": float(linear_mrr),
		"rf_mrr": float(rf_mrr),
	}


def perturb_dag(original_cag, removal_fraction):
	perturbed_cag = copy.deepcopy(original_cag)

	edges_with_importance = []
	for parent, children in perturbed_cag.dag.items():
		for child in children:
			importance = len(perturbed_cag.descendants.get(child, set()))
			edges_with_importance.append((parent, child, importance))

	edges_with_importance.sort(key=lambda item: (-item[2], item[0], item[1]))
	number_to_remove = int(len(edges_with_importance) * removal_fraction)
	if number_to_remove <= 0:
		return _rebuild_parents_and_descendants(perturbed_cag)

	edges_to_remove = {
		(parent, child)
		for parent, child, _ in edges_with_importance[:number_to_remove]
	}

	for parent, child in edges_to_remove:
		if child in perturbed_cag.dag.get(parent, []):
			perturbed_cag.dag[parent].remove(child)

	return _rebuild_parents_and_descendants(perturbed_cag)


def evaluate_dag_ablation(removal_fraction, base_cag, X_train_healthy, X_test, y_test):
	perturbed_cag = perturb_dag(base_cag, removal_fraction)

	linear_model = _fit_model_with_edge_r2(perturbed_cag, use_rf=False, X_train_healthy=X_train_healthy)
	rf_model = _fit_model_with_edge_r2(perturbed_cag, use_rf=True, X_train_healthy=X_train_healthy)

	y_pred_linear = linear_model.predict(X_test)
	y_pred_rf = rf_model.predict(X_test)

	linear_top1 = float(accuracy_score(y_test, y_pred_linear))
	rf_top1 = float(accuracy_score(y_test, y_pred_rf))

	linear_ranked = linear_model.predict_ranked(X_test)
	rf_ranked = rf_model.predict_ranked(X_test)

	linear_top3, linear_mrr = compute_topk_and_mrr(
		y_test.values,
		linear_ranked,
		perturbed_cag.root_to_fault,
		k=3
	)
	rf_top3, rf_mrr = compute_topk_and_mrr(
		y_test.values,
		rf_ranked,
		perturbed_cag.root_to_fault,
		k=3
	)

	return {
		"linear_top1": linear_top1,
		"rf_top1": rf_top1,
		"linear_top3": float(linear_top3),
		"rf_top3": float(rf_top3),
		"linear_mrr": float(linear_mrr),
		"rf_mrr": float(rf_mrr),
	}


def build_structural_variant(original_cag, mode):
	variant_cag = copy.deepcopy(original_cag)

	if mode == "full":
		return _rebuild_parents_and_descendants(variant_cag)

	if mode == "root_only":
		root_set = set(variant_cag.root_metrics)
		variant_cag.dag = {
			parent: children[:] if parent in root_set else []
			for parent, children in variant_cag.dag.items()
		}
		return _rebuild_parents_and_descendants(variant_cag)

	if mode == "empty":
		variant_cag.dag = {parent: [] for parent in variant_cag.dag.keys()}
		return _rebuild_parents_and_descendants(variant_cag)

	raise ValueError(f"Unsupported structural mode: {mode}")


def evaluate_structural_variant(mode, base_cag, X_train_healthy, X_test, y_test):
	variant_cag = build_structural_variant(base_cag, mode)

	linear_model = _fit_model_with_edge_r2(variant_cag, use_rf=False, X_train_healthy=X_train_healthy)
	rf_model = _fit_model_with_edge_r2(variant_cag, use_rf=True, X_train_healthy=X_train_healthy)

	y_pred_linear = linear_model.predict(X_test)
	y_pred_rf = rf_model.predict(X_test)

	linear_top1 = float(accuracy_score(y_test, y_pred_linear))
	rf_top1 = float(accuracy_score(y_test, y_pred_rf))

	linear_ranked = linear_model.predict_ranked(X_test)
	rf_ranked = rf_model.predict_ranked(X_test)

	linear_top3, linear_mrr = compute_topk_and_mrr(
		y_test.values,
		linear_ranked,
		variant_cag.root_to_fault,
		k=3
	)
	rf_top3, rf_mrr = compute_topk_and_mrr(
		y_test.values,
		rf_ranked,
		variant_cag.root_to_fault,
		k=3
	)

	return {
		"linear_top1": linear_top1,
		"rf_top1": rf_top1,
		"linear_top3": float(linear_top3),
		"rf_top3": float(rf_top3),
		"linear_mrr": float(linear_mrr),
		"rf_mrr": float(rf_mrr),
	}


def main():
	X_train, X_test, y_train, y_test, X_train_healthy = prepare_data()

	cag = DatacenterCAG()

	linear_model = _fit_model_with_edge_r2(cag, use_rf=False, X_train_healthy=X_train_healthy)
	rf_model = _fit_model_with_edge_r2(cag, use_rf=True, X_train_healthy=X_train_healthy)

	mask_results = {}
	for mask_rate in MASK_RATES:
		metrics = evaluate_with_missing(mask_rate, X_test, y_test, linear_model, rf_model)
		mask_results[f"{mask_rate:.1f}"] = metrics

	print("\n=== Missing Data Robustness ===")
	print("Mask % | Linear Top1 | RF Top1 | Linear Top3 | RF Top3")
	for mask_rate in MASK_RATES:
		key = f"{mask_rate:.1f}"
		res = mask_results[key]
		print(
			f"{int(mask_rate*100):>5}% | "
			f"{res['linear_top1']:.4f}      | "
			f"{res['rf_top1']:.4f} | "
			f"{res['linear_top3']:.4f}       | "
			f"{res['rf_top3']:.4f}"
		)

	noise_results = {}
	for noise_std in NOISE_LEVELS:
		metrics = evaluate_with_noise(noise_std, X_test, y_test, linear_model, rf_model)
		noise_results[f"{noise_std:.3f}".rstrip('0').rstrip('.')] = metrics

	print("\n=== Noise Sensitivity ===")
	print("Noise STD | Linear Top1 | RF Top1 | Linear Top3 | RF Top3")
	for noise_std in NOISE_LEVELS:
		key = f"{noise_std:.3f}".rstrip('0').rstrip('.')
		res = noise_results[key]
		print(
			f"{noise_std:>8.3f} | "
			f"{res['linear_top1']:.4f}      | "
			f"{res['rf_top1']:.4f} | "
			f"{res['linear_top3']:.4f}       | "
			f"{res['rf_top3']:.4f}"
		)

	dag_ablation_results_importance = {}
	for removal_fraction in REMOVAL_LEVELS:
		metrics = evaluate_dag_ablation(
			removal_fraction,
			cag,
			X_train_healthy,
			X_test,
			y_test
		)
		dag_ablation_results_importance[f"{removal_fraction:.1f}"] = metrics

	print("\n=== Importance-Based DAG Ablation ===")
	print("Removal % | Linear Top1 | RF Top1 | Linear Top3 | RF Top3")
	for removal_fraction in REMOVAL_LEVELS:
		key = f"{removal_fraction:.1f}"
		res = dag_ablation_results_importance[key]
		print(
			f"{int(removal_fraction*100):>8}% | "
			f"{res['linear_top1']:.4f}      | "
			f"{res['rf_top1']:.4f} | "
			f"{res['linear_top3']:.4f}       | "
			f"{res['rf_top3']:.4f}"
		)

	baseline_linear_top1 = mask_results["0.0"]["linear_top1"]
	baseline_rf_top1 = mask_results["0.0"]["rf_top1"]
	ablation_10 = dag_ablation_results_importance.get("0.1", {})
	ablation_20 = dag_ablation_results_importance.get("0.2", {})

	linear_drop_10 = baseline_linear_top1 - float(ablation_10.get("linear_top1", baseline_linear_top1))
	linear_drop_20 = baseline_linear_top1 - float(ablation_20.get("linear_top1", baseline_linear_top1))
	rf_drop_10 = baseline_rf_top1 - float(ablation_10.get("rf_top1", baseline_rf_top1))
	rf_drop_20 = baseline_rf_top1 - float(ablation_20.get("rf_top1", baseline_rf_top1))

	if max(linear_drop_10, linear_drop_20, rf_drop_10, rf_drop_20) < 0.02:
		print("DAG removal did not significantly impact performance — investigate structural dependency.")

	structural_extreme_results = {}
	for mode in STRUCTURAL_MODES:
		structural_extreme_results[mode] = evaluate_structural_variant(
			mode,
			cag,
			X_train_healthy,
			X_test,
			y_test
		)

	print("\n=== Extreme Structural Ablation ===")
	print("Mode | Linear Top1 | RF Top1 | Linear Top3 | RF Top3")
	for mode in STRUCTURAL_MODES:
		res = structural_extreme_results[mode]
		print(
			f"{mode:>9} | "
			f"{res['linear_top1']:.4f}      | "
			f"{res['rf_top1']:.4f} | "
			f"{res['linear_top3']:.4f}       | "
			f"{res['rf_top3']:.4f}"
		)

	full_res = structural_extreme_results["full"]
	empty_res = structural_extreme_results["empty"]
	full_top1_avg = 0.5 * (full_res["linear_top1"] + full_res["rf_top1"])
	empty_top1_avg = 0.5 * (empty_res["linear_top1"] + empty_res["rf_top1"])
	absolute_drop = full_top1_avg - empty_top1_avg

	if absolute_drop <= 0.05:
		print("WARNING: Structural dependency appears weak — investigate intervention logic.")
	elif absolute_drop > 0.10:
		print("Structural reasoning materially contributes to performance.")

	payload = {
		"seed": GLOBAL_SEED,
		"mask_results": mask_results,
		"noise_results": noise_results,
		"dag_ablation_results_importance": dag_ablation_results_importance,
		"structural_extreme_results": structural_extreme_results,
	}
	with open(OUT_PATH, "w", encoding="utf-8") as f:
		json.dump(payload, f, indent=2)


if __name__ == "__main__":
	main()
