#!/usr/bin/env python3

import json
import re
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

SEED = 42
np.random.seed(SEED)

OUTPUT_DATASET = "aiops_reduced_dataset.csv"
OUTPUT_SUBGRAPH_JSON = "aiops_reduced_subgraph.json"
OUTPUT_SUBGRAPH_PNG = "aiops_reduced_subgraph.png"


def _find_data_paths():
    candidates = [
        Path("aiops_challenge_2020_stage1/AIOps挑战赛数据"),
        Path("aiops_data/AIOps挑战赛数据"),
        Path("AIOps挑战赛数据"),
    ]
    fault_candidates = [
        Path("aiops_challenge_2020_stage1/故障整理（预赛）.csv"),
        Path("aiops_data/故障整理（预赛）.csv"),
        Path("故障整理（预赛）.csv"),
    ]

    data_root = next((p for p in candidates if p.exists()), None)
    fault_file = next((p for p in fault_candidates if p.exists()), None)

    if data_root is None:
        raise FileNotFoundError("Could not find AIOps metric data folder (AIOps挑战赛数据).")
    if fault_file is None:
        raise FileNotFoundError("Could not find 故障整理（预赛）.csv.")

    return data_root, fault_file


def _parse_epoch_or_datetime(series):
    numeric = pd.to_numeric(series, errors="coerce")
    dt_from_ms = pd.to_datetime(numeric, unit="ms", errors="coerce")
    dt_from_text = pd.to_datetime(series, errors="coerce")
    return dt_from_ms.fillna(dt_from_text)


def _read_csv_chunks_from_zip(zip_file, member_name, chunksize=120_000):
    encodings = ["utf-8", "gbk", "latin1"]

    for enc in encodings:
        try:
            with zip_file.open(member_name) as fh:
                reader = pd.read_csv(
                    fh,
                    chunksize=chunksize,
                    encoding=enc,
                    on_bad_lines="skip",
                    low_memory=True,
                )
                for chunk in reader:
                    yield chunk
            return
        except UnicodeDecodeError:
            continue
        except pd.errors.ParserError:
            continue
        except Exception:
            continue


def _extract_long_metrics_from_chunk(chunk, csv_stem, date_label):
    if chunk is None or chunk.empty:
        return []

    lower_map = {c.lower(): c for c in chunk.columns}
    time_col = None
    for candidate in ["timestamp", "starttime", "time", "ts"]:
        if candidate in lower_map:
            time_col = lower_map[candidate]
            break

    if time_col is None:
        return []

    ts = _parse_epoch_or_datetime(chunk[time_col]).dt.floor("min")

    frames = []

    # Infrastructure-style records with dynamic metric name + value
    if "value" in chunk.columns and "name" in chunk.columns:
        value = pd.to_numeric(chunk["value"], errors="coerce")
        metric_name = chunk["name"].astype(str).str.strip()
        metric = csv_stem + "__" + metric_name

        long_df = pd.DataFrame(
            {
                "timestamp": ts,
                "metric": metric,
                "value": value,
                "date": date_label,
            }
        ).dropna(subset=["timestamp", "metric", "value"])

        if not long_df.empty:
            grouped = (
                long_df.groupby(["timestamp", "metric", "date"], as_index=False)["value"]
                .median()
            )
            frames.append(grouped)

    # Generic numeric columns for business / trace files
    id_like = {
        time_col,
        "itemid",
        "bomc_id",
        "traceId",
        "id",
        "pid",
        "cmdb_id",
        "serviceName",
        "dsName",
        "callType",
        "name",
        "container",
        "object",
        "kpi",
    }

    numeric_candidates = []
    for col in chunk.columns:
        if col in id_like:
            continue
        col_numeric = pd.to_numeric(chunk[col], errors="coerce")
        if col_numeric.notna().sum() > 0:
            numeric_candidates.append((col, col_numeric))

    for col, col_numeric in numeric_candidates:
        metric = f"{csv_stem}__{col}"
        long_df = pd.DataFrame(
            {
                "timestamp": ts,
                "metric": metric,
                "value": col_numeric,
                "date": date_label,
            }
        ).dropna(subset=["timestamp", "value"])

        if long_df.empty:
            continue

        grouped = (
            long_df.groupby(["timestamp", "metric", "date"], as_index=False)["value"]
            .median()
        )
        frames.append(grouped)

    return frames


def load_data():
    data_root, _ = _find_data_paths()
    zip_files = sorted(data_root.glob("2020_*.zip"))
    if len(zip_files) == 0:
        raise FileNotFoundError(f"No 2020_*.zip files found under {data_root}")

    print(f"\nDiscovered {len(zip_files)} ZIP files for processing.")
    print("Processing all available 2020_*.zip files.\n")

    print("Processing ZIP files (multi-day mode):")
    for zp in zip_files:
        print(" -", zp.name)

    long_parts = []

    for idx, zip_path in enumerate(zip_files, start=1):
        print(f"[{idx}/{len(zip_files)}] Processing {zip_path.name}")

        date_match = re.search(r"(2020_\d{2}_\d{2})", zip_path.name)
        date_label = date_match.group(1) if date_match else zip_path.stem

        with zipfile.ZipFile(zip_path, "r") as zf:
            members = sorted(
                m for m in zf.namelist()
                if m.lower().endswith(".csv") and not Path(m).name.startswith("~$")
            )

            for member in members:
                csv_stem = Path(member).stem
                for chunk in _read_csv_chunks_from_zip(zf, member):
                    frames = _extract_long_metrics_from_chunk(chunk, csv_stem, date_label)
                    if frames:
                        long_parts.extend(frames)

    if not long_parts:
        raise RuntimeError("No metric rows could be loaded from zip files.")

    long_df = pd.concat(long_parts, ignore_index=True)
    long_df = (
        long_df.groupby(["timestamp", "metric", "date"], as_index=False)["value"]
        .median()
    )

    df_metrics_full = (
        long_df.pivot_table(index="timestamp", columns="metric", values="value", aggfunc="median")
        .sort_index()
        .reset_index()
    )
    df_metrics_full["date"] = df_metrics_full["timestamp"].dt.strftime("%Y_%m_%d")

    return df_metrics_full


def _parse_duration_to_timedelta(duration_series):
    txt = duration_series.astype(str).str.lower().str.strip()
    mins = pd.to_numeric(txt.str.extract(r"([0-9]+\.?[0-9]*)\s*min", expand=False), errors="coerce")
    secs = pd.to_numeric(txt.str.extract(r"([0-9]+\.?[0-9]*)\s*s", expand=False), errors="coerce")
    hrs = pd.to_numeric(txt.str.extract(r"([0-9]+\.?[0-9]*)\s*h", expand=False), errors="coerce")

    minutes_total = mins.fillna(0.0) + secs.fillna(0.0) / 60.0 + hrs.fillna(0.0) * 60.0
    minutes_total = minutes_total.replace(0.0, np.nan).fillna(5.0)
    return pd.to_timedelta(minutes_total, unit="m")


def align_incidents(df_metrics_full):
    _, fault_file = _find_data_paths()

    faults = pd.read_csv(fault_file)

    faults["log_time_parsed"] = pd.to_datetime(faults["log_time"], errors="coerce")
    faults["start_time_parsed"] = pd.to_datetime(faults.get("start_time"), errors="coerce")
    faults["window_start"] = faults["start_time_parsed"].fillna(faults["log_time_parsed"])

    duration_td = _parse_duration_to_timedelta(faults.get("duration", pd.Series(dtype=str)))
    faults["window_end"] = faults["window_start"] + duration_td

    # ----------------------------------------
    # Timezone normalization (Faults UTC+8 → UTC)
    # ----------------------------------------

    # Assume fault timestamps are China local time (UTC+8)
    faults["window_start"] = faults["window_start"] - pd.Timedelta(hours=8)
    faults["window_end"] = faults["window_end"] - pd.Timedelta(hours=8)

    # Match metric granularity
    faults["window_start"] = faults["window_start"].dt.floor("min")
    faults["window_end"] = faults["window_end"].dt.ceil("min")

    # ============================================
    # TIMESTAMP DEEP DIAGNOSTIC (DEBUG ONLY)
    # ============================================

    print("\n=== Timestamp Deep Diagnostic ===")

    # Metric samples
    print("\nMetric sample timestamps (first 5):")
    print(df_metrics_full["timestamp"].head())

    print("Metric timestamp dtype:", df_metrics_full["timestamp"].dtype)
    print("Metric timezone:", df_metrics_full["timestamp"].dt.tz)

    print("Metric min/max:",
          df_metrics_full["timestamp"].min(),
          "to",
          df_metrics_full["timestamp"].max())

    # Fault samples
    print("\nFault window_start sample (first 5):")
    print(faults["window_start"].head())

    print("Fault timestamp dtype:", faults["window_start"].dtype)
    print("Fault timezone:", faults["window_start"].dt.tz)

    print("Fault window min/max:",
          faults["window_start"].min(),
          "to",
          faults["window_end"].max())

    # Compute offset difference between first metric and first fault
    try:
        metric_first = df_metrics_full["timestamp"].iloc[0]
        fault_first = faults["window_start"].iloc[0]
        print("\nDifference between first metric and first fault:",
              fault_first - metric_first)
    except Exception as e:
        print("Offset comparison failed:", e)

    print("=============================================\n")

    # ----------------------------------------
    # Alignment Diagnostics (DEBUG ONLY)
    # ----------------------------------------

    print("\n=== Alignment Diagnostics ===")

    print("Metric timestamp range:",
          df_metrics_full["timestamp"].min(),
          "to",
          df_metrics_full["timestamp"].max())

    print("Fault window_start range:",
          faults["window_start"].min(),
          "to",
          faults["window_start"].max())

    print("Fault window_end range:",
          faults["window_end"].min(),
          "to",
          faults["window_end"].max())

    # Check if any global overlap exists
    metric_min = df_metrics_full["timestamp"].min()
    metric_max = df_metrics_full["timestamp"].max()
    fault_min = faults["window_start"].min()
    fault_max = faults["window_end"].max()

    global_overlap = not (fault_max < metric_min or fault_min > metric_max)
    print("Global time overlap exists:", global_overlap)

    # Per-fault window match counts
    match_counts = []

    for row in faults.itertuples(index=False):
        if pd.isna(row.window_start) or pd.isna(row.window_end):
            continue

        mask = (
            (df_metrics_full["timestamp"] >= row.window_start)
            & (df_metrics_full["timestamp"] <= row.window_end)
        )

        match_counts.append(int(mask.sum()))

    print("Total faults:", len(match_counts))
    print("Faults with >=1 matching metric row:",
          sum(1 for x in match_counts if x > 0))
    print("Faults with 0 matching rows:",
          sum(1 for x in match_counts if x == 0))

    print("Sample first 10 match counts:", match_counts[:10])
    print("================================\n")

    fault_desc = faults["fault_desrcibtion"].astype(str).fillna("unknown")
    labels = sorted(fault_desc.unique())
    fault_type_map = {name: idx + 1 for idx, name in enumerate(labels)}
    faults["fault_type"] = fault_desc.map(fault_type_map).astype(int)

    faults = faults.sort_values(["window_start", "index"], kind="mergesort").reset_index(drop=True)
    faults["incident_window_id"] = np.arange(1, len(faults) + 1, dtype=int)

    aligned = df_metrics_full.copy()
    aligned["fault_type"] = 0
    aligned["incident_id"] = 0

    for row in faults.itertuples(index=False):
        if pd.isna(row.window_start) or pd.isna(row.window_end):
            continue

        mask = (
            (aligned["timestamp"] >= row.window_start)
            & (aligned["timestamp"] <= row.window_end)
            & (aligned["fault_type"] == 0)
        )

        if mask.any():
            aligned.loc[mask, "fault_type"] = int(row.fault_type)
            aligned.loc[mask, "incident_id"] = int(row.incident_window_id)

    return aligned, fault_type_map


def filter_metrics(df_aligned):
    feature_cols = [c for c in df_aligned.columns if c not in {"timestamp", "date", "fault_type", "incident_id"}]
    original_metric_count = len(feature_cols)

    missing_frac = df_aligned[feature_cols].isna().mean()
    after_missing_cols = [c for c in feature_cols if missing_frac[c] <= 0.30]

    if not after_missing_cols:
        raise RuntimeError("No metrics remain after missing-value filtering.")

    variance = df_aligned[after_missing_cols].var(skipna=True)
    after_variance_cols = [c for c in after_missing_cols if variance[c] > 1e-12]

    if not after_variance_cols:
        raise RuntimeError("No metrics remain after variance filtering.")

    df_work = df_aligned[["timestamp", "date", "fault_type", "incident_id"] + after_variance_cols].copy()
    medians = df_work[after_variance_cols].median()
    df_work[after_variance_cols] = df_work[after_variance_cols].fillna(medians)

    # ----------------------------------------
    # Unsupervised metric selection (variance-based)
    # ----------------------------------------

    variance_series = df_work[after_variance_cols].var(skipna=True)
    variance_series = variance_series.sort_values(ascending=False)

    def infer_layer(metric_name):
        if metric_name.startswith("db_"):
            return "database"
        if metric_name.startswith("mw_"):
            return "middleware"
        if metric_name.startswith("os_"):
            return "infrastructure"
        return "unknown"

    layer_groups = {
        "database": [],
        "middleware": [],
        "infrastructure": [],
        "unknown": [],
    }

    for metric in variance_series.index.tolist():
        layer_groups[infer_layer(metric)].append(metric)

    selected = []
    for layer in ["database", "middleware", "infrastructure", "unknown"]:
        layer_metrics = layer_groups[layer]
        selected.extend(layer_metrics[: min(10, len(layer_metrics))])

    if len(selected) < 20:
        selected_set = set(selected)
        for metric in variance_series.index.tolist():
            if metric in selected_set:
                continue
            selected.append(metric)
            selected_set.add(metric)
            if len(selected) >= 20:
                break

    summary = {
        "original_metric_count": int(original_metric_count),
        "after_filtering_count": int(len(after_variance_cols)),
        "after_selection_count": int(len(selected)),
        "after_mi_count": int(len(selected)),
        "selection_method": "top_variance_unsupervised",
        "top_variances": variance_series.head(10),
    }

    return df_work, selected, summary


def _partial_corr_from_data(X):
    model = LedoitWolf().fit(X)
    precision = model.precision_

    diag = np.sqrt(np.diag(precision))
    denom = np.outer(diag, diag)

    partial = -precision / np.where(denom == 0.0, 1e-12, denom)
    np.fill_diagonal(partial, 1.0)
    return partial


def discover_structure(df_filtered, selected_metrics):
    healthy = df_filtered[df_filtered["fault_type"] == 0]
    if healthy.empty:
        raise RuntimeError("No healthy rows available for structure discovery.")

    healthy_X = healthy[selected_metrics].to_numpy(dtype=float)
    model = LedoitWolf().fit(healthy_X)
    precision = model.precision_

    metric_names = selected_metrics
    print("Using normalized partial correlations")

    threshold_values = [0.30, 0.20, 0.15, 0.10]

    # ----------------------------------------
    # Layer mapping
    # ----------------------------------------
    def infer_layer(metric_name):
        if metric_name.startswith("db_"):
            return "database"
        if metric_name.startswith("mw_"):
            return "middleware"
        if metric_name.startswith("os_"):
            return "infrastructure"
        return "unknown"

    layer_order = {
        "database": 3,
        "middleware": 2,
        "infrastructure": 1,
        "unknown": 0,
    }

    baseline_dag = None
    baseline_stats = None

    for threshold in threshold_values:
        print("\n==============================")
        print(f"Threshold = {threshold}")
        print("==============================")

        partial = np.zeros_like(precision, dtype=float)
        np.fill_diagonal(partial, 1.0)

        edges = []
        adjacency = {metric: [] for metric in metric_names}

        for i in range(len(metric_names)):
            for j in range(i + 1, len(metric_names)):

                theta_ij = precision[i, j]
                theta_ii = precision[i, i]
                theta_jj = precision[j, j]

                # Avoid division by zero
                if theta_ii <= 0 or theta_jj <= 0:
                    continue

                # Compute normalized partial correlation
                rho_ij = -theta_ij / np.sqrt(theta_ii * theta_jj)
                partial[i, j] = rho_ij
                partial[j, i] = rho_ij

                if not np.isfinite(rho_ij):
                    continue

                if abs(rho_ij) >= threshold:
                    source = metric_names[i]
                    target = metric_names[j]

                    edges.append({
                        "source": source,
                        "target": target,
                        "weight": float(rho_ij)
                    })

                    adjacency[source].append(target)
                    adjacency[target].append(source)

        undirected = nx.Graph()
        undirected.add_nodes_from(selected_metrics)
        for edge in edges:
            undirected.add_edge(
                edge["source"],
                edge["target"],
                weight=float(edge["weight"])
            )

        component_nodes = sorted(undirected.nodes())

        # ----------------------------------------
        # Convert to layer-constrained DAG
        # ----------------------------------------
        dag = nx.DiGraph()
        dag.add_nodes_from(component_nodes)

        component_set = set(component_nodes)
        for edge in edges:
            u = edge["source"]
            v = edge["target"]
            w = edge["weight"]

            if u not in component_set or v not in component_set:
                continue

            layer_u = infer_layer(u)
            layer_v = infer_layer(v)

            if layer_order[layer_u] > layer_order[layer_v]:
                dag.add_edge(u, v, weight=w)
            elif layer_order[layer_v] > layer_order[layer_u]:
                dag.add_edge(v, u, weight=w)
            else:
                # same layer → orient by absolute variance (higher variance upstream)
                var_u = df_filtered[u].var()
                var_v = df_filtered[v].var()
                if var_u >= var_v:
                    dag.add_edge(u, v, weight=w)
                else:
                    dag.add_edge(v, u, weight=w)

        print("Node count:", dag.number_of_nodes())
        print("Edge count:", dag.number_of_edges())
        print("Density:", round(nx.density(dag), 4))
        print("Is DAG:", nx.is_directed_acyclic_graph(dag))

        degrees = dict(dag.degree())
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
        print("Top 5 nodes:", top_nodes)

        oracle_nodes = [n for n in dag.nodes() if n.startswith("db_")]
        print("Oracle present:", len(oracle_nodes) > 0)

        degree_map = dict(dag.degree())
        top5 = sorted(degree_map.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
        density = nx.density(dag)

        edges_output = [
            {"source": u, "target": v, "weight": float(data["weight"])}
            for u, v, data in dag.edges(data=True)
        ]

        stats = {
            "node_count": int(dag.number_of_nodes()),
            "edge_count": int(dag.number_of_edges()),
            "density": float(density),
            "top5_highest_degree_nodes": [{"node": k, "degree": int(v)} for k, v in top5],
            "threshold_partial_corr": threshold,
            "edges_output": edges_output,
        }

        if threshold == 0.10:
            baseline_dag = dag
            baseline_stats = stats

    if baseline_dag is None:
        baseline_dag = dag
        baseline_stats = stats

    # Structural artifact exports from frozen baseline DAG
    node_order = list(baseline_dag.nodes())

    adjacency_binary = nx.to_pandas_adjacency(
        baseline_dag,
        nodelist=node_order,
        dtype=int,
        weight=None,
    )
    adjacency_binary.to_csv("aiops_adjacency_binary.csv")

    adjacency_weighted = nx.to_pandas_adjacency(
        baseline_dag,
        nodelist=node_order,
        dtype=float,
        weight="weight",
    )
    adjacency_weighted.to_csv("aiops_adjacency_weighted.csv")

    node_layers = pd.DataFrame(
        {
            "node": node_order,
            "layer": [infer_layer(node) for node in node_order],
        }
    )
    node_layers.to_csv("aiops_node_layers.csv", index=False)

    topo_order = list(nx.topological_sort(baseline_dag))
    Path("aiops_topological_order.txt").write_text("\n".join(topo_order) + "\n", encoding="utf-8")

    longest_path_nodes = nx.dag_longest_path(baseline_dag) if baseline_dag.number_of_nodes() > 0 else []
    longest_path_length = len(longest_path_nodes) - 1 if longest_path_nodes else 0

    structural_summary = {
        "node_count": int(baseline_dag.number_of_nodes()),
        "edge_count": int(baseline_dag.number_of_edges()),
        "density": float(nx.density(baseline_dag)),
        "average_in_degree": float(np.mean([d for _, d in baseline_dag.in_degree()])) if baseline_dag.number_of_nodes() > 0 else 0.0,
        "average_out_degree": float(np.mean([d for _, d in baseline_dag.out_degree()])) if baseline_dag.number_of_nodes() > 0 else 0.0,
        "longest_path_length": int(longest_path_length),
        "longest_path_nodes": longest_path_nodes,
        "number_of_roots": int(sum(1 for _, d in baseline_dag.in_degree() if d == 0)),
        "number_of_leaves": int(sum(1 for _, d in baseline_dag.out_degree() if d == 0)),
    }
    with open("aiops_structural_summary.json", "w", encoding="utf-8") as f:
        json.dump(structural_summary, f, ensure_ascii=False, indent=2)

    return baseline_dag, baseline_stats


def build_reduced_dataset(df_filtered, dag):
    selected_nodes = sorted(dag.nodes())

    reduced = df_filtered[["timestamp", "fault_type", "incident_id"] + selected_nodes].copy()

    max_fault_incident = int(reduced["incident_id"].max())
    healthy_mask = reduced["incident_id"] == 0

    healthy_ids = np.arange(max_fault_incident + 1, max_fault_incident + 1 + healthy_mask.sum(), dtype=int)
    reduced.loc[healthy_mask, "incident_id"] = healthy_ids

    agg = (
        reduced.groupby("incident_id", as_index=False)
        .agg({**{m: "median" for m in selected_nodes}, "fault_type": "first"})
    )

    ordered_cols = ["incident_id", "fault_type"] + selected_nodes
    return agg[ordered_cols]


def save_outputs(df_reduced, dag, stats):
    df_reduced.to_csv(OUTPUT_DATASET, index=False)

    adjacency = {n: sorted(list(dag.successors(n))) for n in sorted(dag.nodes())}
    payload = {
        "nodes": sorted(dag.nodes()),
        "edges": [{"source": u, "target": v, "weight": float(d.get("weight", 0.0))} for u, v, d in dag.edges(data=True)],
        "adjacency": adjacency,
        "graph_statistics": stats,
        "seed": SEED,
    }

    with open(OUTPUT_SUBGRAPH_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(dag, seed=SEED)
    nx.draw_networkx_nodes(dag, pos, node_size=900, node_color="#AED6F1")
    nx.draw_networkx_edges(dag, pos, arrows=True, arrowstyle="-|>", arrowsize=18, width=1.8)
    nx.draw_networkx_labels(dag, pos, font_size=8)
    plt.title("AIOps Reduced Causal Subgraph")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_SUBGRAPH_PNG, dpi=220)
    plt.close()


def main():
    print("=== AIOps Reduced Causal Subgraph Discovery ===")
    print(f"Random seed: {SEED}")

    df_metrics_full = load_data()
    df_aligned, fault_type_map = align_incidents(df_metrics_full)

    df_filtered, selected_metrics, metric_summary = filter_metrics(df_aligned)

    dag, graph_stats = discover_structure(df_filtered, selected_metrics)
    df_reduced = build_reduced_dataset(df_filtered, dag)

    save_outputs(df_reduced, dag, graph_stats)

    print("\n=== Summary Report ===")
    print(f"Total original metrics: {metric_summary['original_metric_count']}")
    print(f"Metrics after cleaning: {metric_summary['after_filtering_count']}")
    print(f"Metrics after MI filtering: {metric_summary['after_mi_count']}")
    print(f"Final node count: {graph_stats['node_count']}")
    print(f"Final edge count: {graph_stats['edge_count']}")
    print(f"Graph density: {graph_stats['density']:.4f}")

    top5 = graph_stats["top5_highest_degree_nodes"]
    print("Top 5 influential nodes:")
    for item in top5:
        print(f"  - {item['node']} (degree={item['degree']})")

    print("\nFault type mapping:")
    for fault_name, fault_id in sorted(fault_type_map.items(), key=lambda kv: kv[1]):
        print(f"  {fault_id}: {fault_name}")

    print("\nOutputs written:")
    print(f"  - {OUTPUT_DATASET}")
    print(f"  - {OUTPUT_SUBGRAPH_JSON}")
    print(f"  - {OUTPUT_SUBGRAPH_PNG}")


if __name__ == "__main__":
    main()
