#!/usr/bin/env python3
"""
Fault-realism sensitivity sweep.

Re-runs the ranking evaluation on generator configurations whose fault
prevalence and fault-class mix are matched to published datacenter failure
measurements, instead of the balanced/uniform defaults:

  Gill, Jain and Nagappan, "Understanding Network Failures in Data Centers"
  (SIGCOMM 2011)
    Table 3  -- 28.6% of logged link events had measurable traffic impact,
                i.e. an event corpus that is ~71% non-impacting.
    Fig. 6   -- device failures are strongly skewed across types
                (38/28/15/9/4/4 percent), not uniform.
    Sec. 4.6 -- 41% of correlated failure groups contain more than one
                failure; only 10% contain more than four.

Reported metrics are fault-only so they stay comparable when prevalence
changes; the all-incident Top-1 used for the headline table is also emitted.

Usage (from a directory holding the generator + DAG CSVs):
    python sensitivity_realism.py
Output: sensitivity_realism.json  + a markdown table on stdout.
"""

import json
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Importing the evaluation module executes its top-level script once. That is
# unavoidable here (the module is a script, not a library) and its side effect
# of regenerating datacenter_dataset.csv is why every config below rebuilds the
# CSV immediately before it is read.
from datacenter_eval import (  # noqa: E402
    CausalInterventionRCA,
    DatacenterCAG,
    METRICS,
    PearsonBaseline,
)

SEEDS = [20260227, 42, 7, 1234, 999]
DATA_PATH = "datacenter_dataset.csv"
OUT_PATH = "sensitivity_realism.json"

# Gill et al. Fig. 6 device-failure shares (38/28/15/9/4/4), with the two 4%
# tail types merged into one class so the vector matches the generator's five
# architectural fault classes.
GILL_CLASS_SKEW = "38,28,15,9,10"

# Gill et al. Table 3: 13,330 of 46,676 logged link events (28.6%) had
# measurable traffic impact.
GILL_HEALTHY_RATIO = 0.714

CONFIGS = [
    {
        "name": "balanced-uniform",
        "label": "Balanced, uniform classes (paper default)",
        "args": [],
    },
    {
        "name": "gill-prevalence",
        "label": "Literature fault prevalence (28.6% fault)",
        "args": ["--healthy-ratio", str(GILL_HEALTHY_RATIO)],
    },
    {
        "name": "gill-skew",
        "label": "Literature class skew (38/28/15/9/10)",
        "args": ["--class-weights", GILL_CLASS_SKEW],
    },
    {
        "name": "gill-combined",
        "label": "Literature prevalence + class skew",
        "args": [
            "--healthy-ratio", str(GILL_HEALTHY_RATIO),
            "--class-weights", GILL_CLASS_SKEW,
        ],
    },
]


def generate(seed, extra_args):
    subprocess.run(
        [sys.executable, "datacenter_dataset_generator.py", "--seed", str(seed)]
        + extra_args,
        check=True,
        stdout=subprocess.DEVNULL,
    )


def split(seed):
    df = pd.read_csv(DATA_PATH)
    incident_df = (
        df.groupby("incident_id", as_index=False)
          .agg({**{m: "median" for m in METRICS}, "fault_type": "first"})
    )
    X, y = incident_df[METRICS], incident_df["fault_type"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    return X_train, X_test, y_train, y_test, X_train[y_train == 0]


def fit_sem(cag, use_rf, X_train_healthy):
    model = CausalInterventionRCA(cag, use_rf=use_rf)
    model.fit(X_train_healthy)
    edge_r2 = {}
    for parent, children in cag.dag.items():
        for child in children:
            edge_df = X_train_healthy[[parent, child]].dropna()
            if edge_df.shape[0] < 2:
                edge_r2[(parent, child)] = 0.0
                continue
            lr = LinearRegression().fit(edge_df[[parent]].values, edge_df[child].values)
            r2 = float(lr.score(edge_df[[parent]].values, edge_df[child].values))
            edge_r2[(parent, child)] = r2 if np.isfinite(r2) else 0.0
    model.edge_r2 = edge_r2
    return model


def score(y_true, ranked, root_to_fault, k=3):
    """Fault-only Top-1, Top-k and MRR. Healthy incidents are skipped so the
    numbers stay comparable across configurations with different prevalence."""
    top1, topk, rr = [], [], []
    for yt, ranking in zip(y_true, ranked):
        if yt == 0:
            continue
        faults = [root_to_fault[r] for r in ranking]
        top1.append(1.0 if faults[0] == yt else 0.0)
        topk.append(1.0 if yt in faults[:k] else 0.0)
        rr.append(next((1.0 / i for i, f in enumerate(faults, 1) if f == yt), 0.0))
    n = len(top1)
    return {
        "top1": float(np.mean(top1)) if n else 0.0,
        "top3": float(np.mean(topk)) if n else 0.0,
        "mrr": float(np.mean(rr)) if n else 0.0,
        "n_fault": n,
    }


def all_incident_top1(X_eval, y_true, model, ranked):
    """Paper's headline Top-1: healthy incidents count, gated by the anomaly
    threshold. Prevalence-sensitive by construction, reported for continuity."""
    preds = []
    for (_, row), ranking in zip(X_eval.iterrows(), ranked):
        scores = model.root_anomaly_scores(row)
        preds.append(
            0 if max(scores.values()) < model.healthy_threshold
            else model.metric_to_fault(ranking[0])
        )
    return float(accuracy_score(y_true, np.array(preds)))


def run_config(cfg):
    per_seed = []
    for seed in SEEDS:
        generate(seed, cfg["args"])
        X_train, X_test, y_train, y_test, X_healthy = split(seed)
        cag = DatacenterCAG()

        row = {"seed": seed, "n_healthy_train": int(X_healthy.shape[0])}
        for tag, use_rf in (("linear", False), ("rf", True)):
            model = fit_sem(cag, use_rf, X_healthy)
            ranked = model.predict_ranked(X_test)
            row[tag] = score(y_test.values, ranked, cag.root_to_fault)
            row[tag]["top1_all_incident"] = all_incident_top1(
                X_test, y_test, model, ranked
            )

        pearson = PearsonBaseline(cag.root_to_fault)
        pearson.fit(X_train, y_train)
        row["pearson"] = score(y_test.values, pearson.predict_ranked(X_test),
                               cag.root_to_fault)
        per_seed.append(row)

    agg = {}
    for tag in ("linear", "rf", "pearson"):
        for metric in ("top1", "top3", "mrr"):
            vals = [r[tag][metric] for r in per_seed]
            agg[f"{tag}_{metric}_mean"] = float(np.mean(vals))
            agg[f"{tag}_{metric}_std"] = float(np.std(vals, ddof=0))
    agg["mean_fault_incidents"] = float(np.mean([r["linear"]["n_fault"] for r in per_seed]))
    return {"config": cfg, "per_seed": per_seed, "aggregate": agg}


def main():
    results = []
    for cfg in CONFIGS:
        print(f"\n=== {cfg['label']} ===", flush=True)
        res = run_config(cfg)
        a = res["aggregate"]
        print(f"  Linear  Top-1 {a['linear_top1_mean']:.3f} +/- {a['linear_top1_std']:.3f}")
        print(f"  RF      Top-1 {a['rf_top1_mean']:.3f} +/- {a['rf_top1_std']:.3f}")
        print(f"  Pearson Top-1 {a['pearson_top1_mean']:.3f} +/- {a['pearson_top1_std']:.3f}")
        results.append(res)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"seeds": SEEDS, "results": results}, f, indent=2)

    print("\n\n| Configuration | Fault inc. | Linear Top-1 | RF Top-1 | Pearson Top-1 | RF MRR |")
    print("|---|---|---|---|---|---|")
    for res in results:
        a, label = res["aggregate"], res["config"]["label"]
        print(
            f"| {label} | {a['mean_fault_incidents']:.0f} "
            f"| {a['linear_top1_mean']:.3f} ± {a['linear_top1_std']:.3f} "
            f"| {a['rf_top1_mean']:.3f} ± {a['rf_top1_std']:.3f} "
            f"| {a['pearson_top1_mean']:.3f} ± {a['pearson_top1_std']:.3f} "
            f"| {a['rf_mrr_mean']:.3f} ± {a['rf_mrr_std']:.3f} |"
        )
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
