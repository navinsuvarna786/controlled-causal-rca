#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np
import pandas as pd


AIOPS_SUMMARY_PATH = Path("aiops_structural_summary.json")
DC_SUMMARY_PATH = Path("datacenter_structural_summary.json")
AIOPS_ADJ_PATH = Path("aiops_adjacency_binary.csv")
DC_ADJ_PATH = Path("datacenter_adjacency_binary.csv")
OUTPUT_PATH = Path("structural_similarity_report.json")


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _safe_similarity_relative(aiops_value: float, dc_value: float) -> float:
    if dc_value == 0:
        return 0.0
    return 1.0 - abs(aiops_value - dc_value) / dc_value


def _safe_similarity_symmetric(aiops_value: float, dc_value: float) -> float:
    scale = max(aiops_value, dc_value)
    if scale == 0:
        return 0.0
    return 1.0 - abs(aiops_value - dc_value) / scale


def main():
    with AIOPS_SUMMARY_PATH.open("r", encoding="utf-8") as f:
        aiops_summary = json.load(f)

    with DC_SUMMARY_PATH.open("r", encoding="utf-8") as f:
        dc_summary = json.load(f)

    aiops_adj = pd.read_csv(AIOPS_ADJ_PATH, index_col=0)
    dc_adj = pd.read_csv(DC_ADJ_PATH, index_col=0)

    aiops_depth = float(aiops_summary["longest_path_length"])
    dc_depth = float(dc_summary["longest_path_length"])

    aiops_avg_degree = float(aiops_adj.sum(axis=1).mean() + aiops_adj.sum(axis=0).mean())
    dc_avg_degree = float(dc_adj.sum(axis=1).mean() + dc_adj.sum(axis=0).mean())

    aiops_roots = float(aiops_summary["number_of_roots"])
    aiops_nodes = float(aiops_summary["node_count"])
    dc_roots = float(dc_summary["number_of_roots"])
    dc_nodes = float(dc_summary["node_count"])

    aiops_leaves = float(aiops_summary["number_of_leaves"])
    dc_leaves = float(dc_summary["number_of_leaves"])

    aiops_density = float(aiops_summary["density"])
    dc_density = float(dc_summary["density"])

    depth_similarity = _safe_similarity_relative(aiops_depth, dc_depth)
    degree_similarity = _safe_similarity_relative(aiops_avg_degree, dc_avg_degree)
    root_ratio_similarity = 1.0 - abs(_safe_div(aiops_roots, aiops_nodes) - _safe_div(dc_roots, dc_nodes))
    leaf_ratio_similarity = 1.0 - abs(_safe_div(aiops_leaves, aiops_nodes) - _safe_div(dc_leaves, dc_nodes))
    density_similarity = _safe_similarity_symmetric(aiops_density, dc_density)

    structural_similarity_score_no_density = float(
        np.mean(
            [
                depth_similarity,
                degree_similarity,
                root_ratio_similarity,
                leaf_ratio_similarity,
            ]
        )
    )

    structural_similarity_score_with_density = float(
        np.mean(
            [
                depth_similarity,
                degree_similarity,
                root_ratio_similarity,
                leaf_ratio_similarity,
                density_similarity,
            ]
        )
    )

    report = {
        "depth_similarity": float(depth_similarity),
        "degree_similarity": float(degree_similarity),
        "root_ratio_similarity": float(root_ratio_similarity),
        "leaf_ratio_similarity": float(leaf_ratio_similarity),
        "density_similarity": float(density_similarity),
        "structural_similarity_score_no_density": structural_similarity_score_no_density,
        "structural_similarity_score_with_density": structural_similarity_score_with_density,
        "inputs": {
            "aiops_structural_summary": str(AIOPS_SUMMARY_PATH),
            "datacenter_structural_summary": str(DC_SUMMARY_PATH),
            "aiops_adjacency_binary": str(AIOPS_ADJ_PATH),
            "datacenter_adjacency_binary": str(DC_ADJ_PATH),
        },
    }

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
