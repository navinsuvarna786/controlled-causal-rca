#!/usr/bin/env python3
"""
Datacenter Spine-Leaf Causal RCA Dataset Generator (Reproducible)

Design Goals:
- Same 28 metrics as v4
- Same DAG topology
- Deterministic (fixed seed)
- Logged version hash
- No multi-fault
- Minimal gray failures (5%)
- Calibrated for:
    RF classifier: ~88–92% Top-1
    Linear SEM: ~70–75% Top-1
    RF SEM: ~75–80% Top-1
"""

import numpy as np
import pandas as pd
import hashlib
import argparse
import random
from scipy.special import expit

# ===============================
# CLI SEED HANDLING
# ===============================

DEFAULT_SEED = 20260227

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
parser.add_argument(
    "--multi-fault-prob", type=float, default=0.0,
    help="Probability that a fault incident contains a SECOND simultaneous "
         "primary fault from a distinct fault_type. Default 0.0 = single-fault.",
)
parser.add_argument(
    "--output-csv", type=str, default="datacenter_dataset.csv",
    help="Output CSV path (use a separate name for multi-fault variants).",
)
parser.add_argument(
    "--healthy-ratio", type=float, default=0.40,
    help="Fraction of incidents with no injected fault. Default 0.40. "
         "Gill et al. (SIGCOMM'11, Table 3) report 28.6%% of logged link "
         "events had measurable traffic impact, i.e. a healthy ratio of ~0.71.",
)
parser.add_argument(
    "--gray-prob", type=float, default=0.05,
    help="Fraction of fault incidents with partially masked anomaly signal.",
)
parser.add_argument(
    "--class-weights", type=str, default=None,
    help="Comma-separated relative frequencies for the 5 fault classes. "
         "Default None = uniform. Gill et al. (SIGCOMM'11, Fig. 6) observe a "
         "strongly skewed device-failure distribution (38/28/15/9/10).",
)
args = parser.parse_args()

GLOBAL_SEED = args.seed
MULTI_FAULT_PROB = args.multi_fault_prob
OUTPUT_CSV = args.output_csv

if args.class_weights is None:
    CLASS_WEIGHTS = None
else:
    CLASS_WEIGHTS = np.array([float(w) for w in args.class_weights.split(",")], dtype=float)
    if CLASS_WEIGHTS.size != 5 or (CLASS_WEIGHTS < 0).any() or CLASS_WEIGHTS.sum() <= 0:
        raise ValueError("--class-weights needs 5 non-negative values summing to > 0")
    CLASS_WEIGHTS = CLASS_WEIGHTS / CLASS_WEIGHTS.sum()

np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)

print(f"Random Seed Used: {GLOBAL_SEED}")

# ---------------------------
# REPRODUCIBILITY SETTINGS
# ---------------------------

VERSION_STRING = "DATACENTER_RCA_V10_DETERMINISTIC"
VERSION_HASH = hashlib.sha256(VERSION_STRING.encode()).hexdigest()

print(f"\nDataset Version: {VERSION_STRING}")
print(f"Version Hash: {VERSION_HASH}")
print(f"Random Seed: {GLOBAL_SEED}\n")

# ---------------------------
# CALIBRATION PARAMETERS
# ---------------------------

FAULT_STRENGTH = 1.2
NOISE_STD = 0.06
GRAY_PROB = args.gray_prob
NUM_INCIDENTS = 240
ROWS_PER_INCIDENT = 30
NUM_ROWS = NUM_INCIDENTS * ROWS_PER_INCIDENT
HEALTHY_INCIDENT_RATIO = args.healthy_ratio

# ---------------------------
# METRIC LIST (28 EXACT)
# ---------------------------

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

def softplus(x):
    return np.log1p(np.exp(x))

def build_incident_plan(rng):
    num_healthy_incidents = int(np.ceil(NUM_INCIDENTS * HEALTHY_INCIDENT_RATIO))
    num_fault_incidents = NUM_INCIDENTS - num_healthy_incidents

    if CLASS_WEIGHTS is None:
        # Uniform: deterministic round-robin so every class gets equal support.
        fault_types = [((idx % 5) + 1) for idx in range(num_fault_incidents)]
    else:
        # Skewed: largest-remainder allocation keeps counts deterministic for a
        # given seed while matching the requested class frequencies.
        exact = CLASS_WEIGHTS * num_fault_incidents
        counts = np.floor(exact).astype(int)
        for cls in np.argsort(-(exact - counts))[: num_fault_incidents - counts.sum()]:
            counts[cls] += 1
        fault_types = [cls + 1 for cls, n in enumerate(counts) for _ in range(n)]
    rng.shuffle(fault_types)

    incident_fault_types = [0] * num_healthy_incidents + fault_types
    rng.shuffle(incident_fault_types)

    fault_incident_indices = [idx for idx, f in enumerate(incident_fault_types) if f != 0]
    max_gray_incidents = int(np.floor(GRAY_PROB * num_fault_incidents))

    gray_incident_mask = np.zeros(NUM_INCIDENTS, dtype=bool)
    if max_gray_incidents > 0:
        gray_indices = rng.choice(fault_incident_indices, size=max_gray_incidents, replace=False)
        gray_incident_mask[gray_indices] = True

    partial_mask_incident_mask = np.zeros(NUM_INCIDENTS, dtype=bool)
    for fault_id in range(1, 6):
        per_fault_indices = [
            idx for idx, fault in enumerate(incident_fault_types)
            if fault == fault_id
        ]
        num_masked = int(np.floor(0.20 * len(per_fault_indices)))
        if num_masked > 0:
            masked_indices = rng.choice(per_fault_indices, size=num_masked, replace=False)
            partial_mask_incident_mask[masked_indices] = True

    return (
        incident_fault_types,
        gray_incident_mask,
        partial_mask_incident_mask,
        num_healthy_incidents,
        num_fault_incidents
    )

def validate_dataset_constraints(df, gray_incident_mask, num_fault_incidents):
    if list(df.columns[:len(METRICS)]) != METRICS:
        raise ValueError("Metric schema mismatch: expected fixed 28 metrics in canonical order")

    healthy_ratio = (df['fault_type'] == 0).mean()
    if abs(healthy_ratio - HEALTHY_INCIDENT_RATIO) > 0.05:
        raise ValueError(
            f"Healthy ratio {healthy_ratio:.4f} deviates from requested "
            f"{HEALTHY_INCIDENT_RATIO:.4f} by more than 0.05"
        )

    fault_values = set(df['fault_type'].unique())
    expected_fault_values = {0} | {
        cls + 1 for cls in range(5)
        if CLASS_WEIGHTS is None or CLASS_WEIGHTS[cls] > 0
    }
    if fault_values != expected_fault_values:
        raise ValueError(
            f"fault_type values mismatch: got {sorted(fault_values)}, expected {sorted(expected_fault_values)}"
        )

    gray_rate_fault_incidents = float(gray_incident_mask.sum()) / max(num_fault_incidents, 1)
    if gray_rate_fault_incidents > GRAY_PROB + 1e-9:
        raise ValueError(
            f"Gray-failure incidence too high: {gray_rate_fault_incidents:.4f} > {GRAY_PROB:.4f}"
        )

    return healthy_ratio, gray_rate_fault_incidents

def generate_dataset():
    rng = np.random.default_rng(GLOBAL_SEED)
    rows = []
    (
        incident_fault_types,
        gray_incident_mask,
        partial_mask_incident_mask,
        num_healthy_incidents,
        num_fault_incidents
    ) = build_incident_plan(rng)

    for incident_id in range(NUM_INCIDENTS):

        fault_type = incident_fault_types[incident_id]
        is_gray = bool(gray_incident_mask[incident_id])
        is_partial_masked = bool(partial_mask_incident_mask[incident_id])
        severity = 0.5 if is_gray else 1.0

        traffic_pressure = rng.normal(0, 0.35)
        control_plane_stress = rng.normal(0, 0.30)
        optics_drift = rng.normal(0, 0.45)
        overlay_instability = rng.normal(0, 0.20)
        shared_root_variability = rng.normal(0, 0.35)

        distractor_root = None
        distractor_magnitude = 0.0
        secondary_fault_type = 0
        secondary_root = None
        secondary_strength = 0.0
        root_metrics = ['optical_rx', 'leaf_down', 'vtep', 'bgp_flaps', 'bgp_update']
        primary_root_by_fault = {
            1: 'optical_rx',
            2: 'leaf_down',
            3: 'vtep',
            4: 'bgp_flaps',
            5: 'bgp_update',
        }
        fault_by_root = {v: k for k, v in primary_root_by_fault.items()}

        if fault_type != 0:
            primary_root = primary_root_by_fault[fault_type]

            if MULTI_FAULT_PROB > 0.0 and rng.random() < MULTI_FAULT_PROB:
                # Multi-fault: inject a SECOND primary-strength fault from a
                # different fault_type. Both faults get full FAULT_STRENGTH.
                second_candidates = [r for r in root_metrics if r != primary_root]
                secondary_root = rng.choice(second_candidates)
                secondary_fault_type = fault_by_root[secondary_root]
                secondary_strength = (
                    FAULT_STRENGTH * severity * rng.uniform(0.85, 1.00)
                )
            elif rng.random() < 0.52:
                # Single fault with weak distractor (legacy behavior).
                candidates = [r for r in root_metrics if r != primary_root]
                distractor_root = rng.choice(candidates)
                distractor_magnitude = (
                    FAULT_STRENGTH * severity * rng.uniform(0.42, 0.60)
                )

        for t in range(ROWS_PER_INCIDENT):

            diurnal = 0.5 + 0.3*np.sin(2*np.pi*(t/ROWS_PER_INCIDENT))
            temperature = 25 + rng.normal(0,1)

            effective_traffic_pressure = traffic_pressure
            if fault_type == 1:
                effective_traffic_pressure += rng.normal(0, 0.25)
            elif fault_type != 0:
                effective_traffic_pressure += rng.normal(0, 0.20)

            # -------- ROOT INJECTION --------

            optical_rx = -2.0 + 0.45*optics_drift + 0.10*effective_traffic_pressure
            leaf_down = 0.5*diurnal + 0.35*effective_traffic_pressure
            vtep = 1.0 - 0.08*overlay_instability - 0.05*control_plane_stress
            bgp_flaps = 0.1 + 0.25*control_plane_stress + 0.08*effective_traffic_pressure
            bgp_update = 0.2 + 0.20*control_plane_stress + 0.10*effective_traffic_pressure

            optical_rx += 0.30 * shared_root_variability
            leaf_down += 0.28 * shared_root_variability
            vtep -= 0.06 * shared_root_variability
            bgp_flaps += 0.22 * shared_root_variability
            bgp_update += 0.24 * shared_root_variability

            if fault_type == 1:
                visibility_scale = rng.uniform(0.70, 1.00)
                primary_strength = FAULT_STRENGTH * severity * visibility_scale * (0.5 if is_partial_masked else 1.0)
                optical_rx -= primary_strength

                coupling_leaf = FAULT_STRENGTH * severity * rng.uniform(0.30, 0.50)
                coupling_bgp_update = FAULT_STRENGTH * severity * rng.uniform(0.30, 0.50)
                leaf_down += rng.normal(1.00 * coupling_leaf, 0.25 * coupling_leaf)
                bgp_update += rng.normal(0.95 * coupling_bgp_update, 0.25 * coupling_bgp_update)
            elif fault_type == 2:
                visibility_scale = rng.uniform(0.70, 1.00)
                primary_strength = FAULT_STRENGTH * severity * visibility_scale * (0.5 if is_partial_masked else 1.0)
                leaf_down += primary_strength

                coupling_optics = FAULT_STRENGTH * severity * rng.uniform(0.30, 0.50)
                coupling_bgp_flaps = FAULT_STRENGTH * severity * rng.uniform(0.30, 0.50)
                optical_rx += rng.normal(-0.95 * coupling_optics, 0.25 * coupling_optics)
                bgp_flaps += rng.normal(1.00 * coupling_bgp_flaps, 0.25 * coupling_bgp_flaps)
            elif fault_type == 3:
                visibility_scale = rng.uniform(0.70, 1.00)
                primary_strength = FAULT_STRENGTH * 0.3 * severity * visibility_scale * (0.5 if is_partial_masked else 1.0)
                vtep -= primary_strength

                coupling_leaf = FAULT_STRENGTH * severity * rng.uniform(0.30, 0.50)
                coupling_bgp_update = FAULT_STRENGTH * severity * rng.uniform(0.30, 0.50)
                leaf_down += rng.normal(0.95 * coupling_leaf, 0.25 * coupling_leaf)
                bgp_update += rng.normal(0.95 * coupling_bgp_update, 0.25 * coupling_bgp_update)
            elif fault_type == 4:
                visibility_scale = rng.uniform(0.70, 1.00)
                primary_strength = FAULT_STRENGTH * severity * visibility_scale * (0.5 if is_partial_masked else 1.0)
                bgp_flaps += primary_strength

                coupling_optics = FAULT_STRENGTH * severity * rng.uniform(0.30, 0.50)
                coupling_vtep = FAULT_STRENGTH * severity * rng.uniform(0.30, 0.50)
                optical_rx += rng.normal(-0.95 * coupling_optics, 0.25 * coupling_optics)
                vtep += rng.normal(-0.35 * coupling_vtep, 0.20 * coupling_vtep)
            elif fault_type == 5:
                visibility_scale = rng.uniform(0.70, 1.00)
                primary_strength = FAULT_STRENGTH * severity * visibility_scale * (0.5 if is_partial_masked else 1.0)
                bgp_update += primary_strength

                coupling_leaf = FAULT_STRENGTH * severity * rng.uniform(0.30, 0.50)
                coupling_bgp_flaps = FAULT_STRENGTH * severity * rng.uniform(0.30, 0.50)
                leaf_down += rng.normal(1.00 * coupling_leaf, 0.25 * coupling_leaf)
                bgp_flaps += rng.normal(0.95 * coupling_bgp_flaps, 0.25 * coupling_bgp_flaps)

            if distractor_root is not None:
                if distractor_root == 'optical_rx':
                    optical_rx += rng.normal(-0.9 * distractor_magnitude, 0.25 * distractor_magnitude)
                elif distractor_root == 'leaf_down':
                    leaf_down += rng.normal(0.9 * distractor_magnitude, 0.25 * distractor_magnitude)
                elif distractor_root == 'vtep':
                    vtep += rng.normal(-0.25 * distractor_magnitude, 0.15 * distractor_magnitude)
                elif distractor_root == 'bgp_flaps':
                    bgp_flaps += rng.normal(0.9 * distractor_magnitude, 0.25 * distractor_magnitude)
                elif distractor_root == 'bgp_update':
                    bgp_update += rng.normal(0.9 * distractor_magnitude, 0.25 * distractor_magnitude)

            if secondary_root is not None:
                # Inject second primary fault at near-full magnitude with the
                # same coupling pattern used for the primary injection.
                if secondary_root == 'optical_rx':
                    optical_rx -= secondary_strength
                    coupling_leaf = FAULT_STRENGTH * severity * rng.uniform(0.30, 0.50)
                    coupling_bgp_update = FAULT_STRENGTH * severity * rng.uniform(0.30, 0.50)
                    leaf_down += rng.normal(1.00 * coupling_leaf, 0.25 * coupling_leaf)
                    bgp_update += rng.normal(0.95 * coupling_bgp_update, 0.25 * coupling_bgp_update)
                elif secondary_root == 'leaf_down':
                    leaf_down += secondary_strength
                    coupling_optics = FAULT_STRENGTH * severity * rng.uniform(0.30, 0.50)
                    coupling_bgp_flaps = FAULT_STRENGTH * severity * rng.uniform(0.30, 0.50)
                    optical_rx += rng.normal(-0.95 * coupling_optics, 0.25 * coupling_optics)
                    bgp_flaps += rng.normal(1.00 * coupling_bgp_flaps, 0.25 * coupling_bgp_flaps)
                elif secondary_root == 'vtep':
                    vtep -= secondary_strength * 0.3
                    coupling_leaf = FAULT_STRENGTH * severity * rng.uniform(0.30, 0.50)
                    coupling_bgp_update = FAULT_STRENGTH * severity * rng.uniform(0.30, 0.50)
                    leaf_down += rng.normal(0.95 * coupling_leaf, 0.25 * coupling_leaf)
                    bgp_update += rng.normal(0.95 * coupling_bgp_update, 0.25 * coupling_bgp_update)
                elif secondary_root == 'bgp_flaps':
                    bgp_flaps += secondary_strength
                    coupling_optics = FAULT_STRENGTH * severity * rng.uniform(0.30, 0.50)
                    coupling_vtep = FAULT_STRENGTH * severity * rng.uniform(0.30, 0.50)
                    optical_rx += rng.normal(-0.95 * coupling_optics, 0.25 * coupling_optics)
                    vtep += rng.normal(-0.35 * coupling_vtep, 0.20 * coupling_vtep)
                elif secondary_root == 'bgp_update':
                    bgp_update += secondary_strength
                    coupling_leaf = FAULT_STRENGTH * severity * rng.uniform(0.30, 0.50)
                    coupling_bgp_flaps = FAULT_STRENGTH * severity * rng.uniform(0.30, 0.50)
                    leaf_down += rng.normal(1.00 * coupling_leaf, 0.25 * coupling_leaf)
                    bgp_flaps += rng.normal(0.95 * coupling_bgp_flaps, 0.25 * coupling_bgp_flaps)

            # -------- PROPAGATION --------

            optical_ber = expit(-optical_rx)
            spine_link = 0.3*diurnal
            spine_queue = 100 + 200*softplus(spine_link-0.6)
            amplification_spine_loss = 1 + rng.normal(0, 0.15)
            spine_loss = amplification_spine_loss * (0.05*optical_ber + 0.01*spine_queue)
            spine_cpu = 0.2 + 0.2*softplus(bgp_update-0.2) + 0.05*spine_link

            leaf_uplink = 0.4*diurnal
            leaf_queue = 80 + 150*softplus(leaf_down-0.5)
            amplification_leaf_loss = rng.uniform(1.5, 2.0)
            leaf_loss_nonlinear = softplus(1.8 * (leaf_queue / 120.0 - 1.0))
            leaf_loss = amplification_leaf_loss * (0.03*optical_ber + 0.015*leaf_queue + 0.45*leaf_loss_nonlinear)
            leaf_cpu = 0.15 + 0.12*softplus(leaf_down-0.5) + 0.1*softplus(bgp_update-0.2)

            amplification_tor_index = 1 + rng.normal(0, 0.15)
            tor_index = amplification_tor_index * (0.1 + 0.45*softplus(leaf_down-0.5))
            amplification_retrans = rng.uniform(1.5, 2.0)
            retrans_nonlinear = softplus(2.0 * (leaf_loss / 6.0 - 1.0))
            retrans = amplification_retrans * (0.012*leaf_loss + 0.008*optical_ber + 0.25*retrans_nonlinear)
            timeout = 0.1*retrans
            amplification_latency = rng.uniform(1.5, 2.0)
            latency_nonlinear = softplus(2.2 * (tor_index - 0.6)) + softplus(2.0 * (retrans - 0.6))
            latency = amplification_latency * (5 + 18*tor_index + 8*retrans + 6*latency_nonlinear)

            vx_encap = 0.01*(1-vtep)
            vx_decap = 0.008*(1-vtep)

            amplification_bgp_conv = 1 + rng.normal(0, 0.15)
            bgp_conv = amplification_bgp_conv * (0.5*bgp_flaps)
            ecmp = 0.3*bgp_flaps
            amplification_routing_loop = 1 + rng.normal(0, 0.15)
            routing_loop = amplification_routing_loop * (0.4*bgp_flaps)

            arp = 0.2 + 0.2*softplus(bgp_update-0.2)
            lldp = 0.1 + 0.2*softplus(bgp_update-0.2)

            vm_density = 8 + 3*diurnal

            metric_values = [
                spine_loss, spine_queue, spine_cpu, spine_link,
                leaf_loss, leaf_queue, leaf_cpu,
                leaf_uplink, leaf_down, 0.01*leaf_queue,
                tor_index, retrans, timeout,
                latency, vm_density,
                optical_rx, optical_ber, temperature,
                vx_encap, vx_decap, vtep,
                bgp_flaps, bgp_conv, ecmp, routing_loop,
                bgp_update, arp, lldp
            ]

            noisy = [v + rng.normal(0, NOISE_STD) for v in metric_values]

            row = dict(zip(METRICS, noisy))
            row['fault_type'] = fault_type
            row['fault_type_secondary'] = secondary_fault_type
            # Exported so evaluations can isolate gray failures, where the fault
            # is real but its telemetry signature is only half strength.
            row['is_gray'] = int(is_gray)
            row['incident_id'] = incident_id
            rows.append(row)

    df = pd.DataFrame(rows)
    healthy_ratio, gray_rate_fault_incidents = validate_dataset_constraints(
        df, gray_incident_mask, num_fault_incidents
    )

    metadata = {
        'num_rows': len(df),
        'num_incidents': NUM_INCIDENTS,
        'num_healthy_incidents': num_healthy_incidents,
        'num_fault_incidents': num_fault_incidents,
        'healthy_ratio_rows': healthy_ratio,
        'gray_rate_fault_incidents': gray_rate_fault_incidents,
        'fault_type_distribution_rows': df['fault_type'].value_counts().sort_index().to_dict()
    }
    return df, metadata


if __name__ == "__main__":
    df, metadata = generate_dataset()
    df.to_csv(OUTPUT_CSV, index=False)

    import json
    import networkx as nx

    edge_list = [
        ("optical_rx_power", "optical_ber"),
        ("optical_ber", "spine_packet_loss_rate"),
        ("optical_ber", "leaf_packet_loss_rate"),
        ("optical_ber", "server_retransmit_rate"),
        ("spine_link_util", "spine_queue_depth"),
        ("spine_link_util", "spine_cpu_util"),
        ("spine_queue_depth", "spine_packet_loss_rate"),
        ("spine_packet_loss_rate", "leaf_packet_loss_rate"),
        ("leaf_link_util_downlink", "leaf_queue_depth"),
        ("leaf_link_util_downlink", "tor_congestion_index"),
        ("leaf_link_util_downlink", "leaf_cpu_util"),
        ("leaf_queue_depth", "leaf_packet_loss_rate"),
        ("leaf_queue_depth", "leaf_buffer_drops"),
        ("leaf_packet_loss_rate", "server_retransmit_rate"),
        ("server_retransmit_rate", "server_conn_timeout_rate"),
        ("server_retransmit_rate", "server_app_latency"),
        ("tor_congestion_index", "server_app_latency"),
        ("vtep_reachability", "vxlan_encap_error_rate"),
        ("vtep_reachability", "vxlan_decap_error_rate"),
        ("bgp_route_flaps", "bgp_convergence_time"),
        ("bgp_route_flaps", "ecmp_path_imbalance"),
        ("bgp_route_flaps", "routing_loop_detected"),
        ("bgp_update_rate", "spine_cpu_util"),
        ("bgp_update_rate", "leaf_cpu_util"),
        ("bgp_update_rate", "arp_request_rate"),
        ("bgp_update_rate", "lldp_frame_rate"),
    ]

    dag = nx.DiGraph()
    dag.add_nodes_from(METRICS)
    dag.add_edges_from(edge_list)

    node_order = list(METRICS)

    adjacency_binary = nx.to_pandas_adjacency(
        dag,
        nodelist=node_order,
        dtype=int,
        weight=None,
    )
    adjacency_binary.to_csv("datacenter_adjacency_binary.csv")

    weighted_dag = dag.copy()
    for u, v in weighted_dag.edges():
        weighted_dag[u][v]["weight"] = 1.0

    adjacency_weighted = nx.to_pandas_adjacency(
        weighted_dag,
        nodelist=node_order,
        dtype=float,
        weight="weight",
    )
    adjacency_weighted.to_csv("datacenter_adjacency_weighted.csv")

    def infer_layer(metric_name):
        if metric_name.startswith("spine_"):
            return "core"
        if metric_name.startswith("leaf_"):
            return "access"
        if metric_name.startswith("server_"):
            return "compute"
        if metric_name.startswith("optical_"):
            return "physical"
        if metric_name.startswith("bgp_"):
            return "control_plane"
        if metric_name.startswith("vxlan_"):
            return "overlay"
        return "infrastructure"

    node_layers = pd.DataFrame(
        {
            "node": node_order,
            "layer": [infer_layer(node) for node in node_order],
        }
    )
    node_layers.to_csv("datacenter_node_layers.csv", index=False)

    topological_order = list(nx.topological_sort(dag))
    with open("datacenter_topological_order.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(topological_order) + "\n")

    longest_path_nodes = list(nx.dag_longest_path(dag))
    longest_path_length = len(longest_path_nodes) - 1 if longest_path_nodes else 0

    structural_summary = {
        "node_count": int(dag.number_of_nodes()),
        "edge_count": int(dag.number_of_edges()),
        "density": float(nx.density(dag)),
        "average_in_degree": float(np.mean([d for _, d in dag.in_degree()])) if dag.number_of_nodes() > 0 else 0.0,
        "average_out_degree": float(np.mean([d for _, d in dag.out_degree()])) if dag.number_of_nodes() > 0 else 0.0,
        "longest_path_length": int(longest_path_length),
        "longest_path_nodes": longest_path_nodes,
        "number_of_roots": int(sum(1 for _, d in dag.in_degree() if d == 0)),
        "number_of_leaves": int(sum(1 for _, d in dag.out_degree() if d == 0)),
    }

    with open("datacenter_structural_summary.json", "w", encoding="utf-8") as f:
        json.dump(structural_summary, f, ensure_ascii=False, indent=2)

    print(f"Dataset saved: {OUTPUT_CSV}")
    print(f"Rows: {metadata['num_rows']}, Incidents: {metadata['num_incidents']}")
    print(f"Healthy incidents: {metadata['num_healthy_incidents']} ({metadata['healthy_ratio_rows']:.2%} of rows)")
    print(f"Gray-failure rate (fault incidents): {metadata['gray_rate_fault_incidents']:.2%}")
    print(f"fault_type distribution (rows): {metadata['fault_type_distribution_rows']}")
