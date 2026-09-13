"""Canonical DAG for spine-leaf datacenter RCA.

Sourced from data/datacenter_adjacency_binary.csv (26 edges, 28 nodes).
This is the single source of truth — eval, diagnostic, and multi-fault
coverage scorer all import from here.

The 28th node count includes 3 isolated metrics (transceiver_temp,
vm_density, leaf_link_util_uplink) that participate in no edge. The DAG
edge structure connects 25 of 28 metrics.
"""

ROOT_TO_FAULT = {
    'optical_rx_power': 1,
    'leaf_link_util_downlink': 2,
    'vtep_reachability': 3,
    'bgp_route_flaps': 4,
    'bgp_update_rate': 5,
}

FAULT_TO_ROOT = {v: k for k, v in ROOT_TO_FAULT.items()}

ROOT_METRICS = list(ROOT_TO_FAULT.keys())

DAG = {
    'optical_rx_power': ['optical_ber'],
    'optical_ber': [
        'spine_packet_loss_rate',
        'leaf_packet_loss_rate',
        'server_retransmit_rate',
    ],
    'spine_link_util': ['spine_queue_depth', 'spine_cpu_util'],
    'spine_queue_depth': ['spine_packet_loss_rate'],
    'spine_packet_loss_rate': ['leaf_packet_loss_rate'],
    'leaf_link_util_downlink': [
        'leaf_queue_depth',
        'leaf_cpu_util',
        'tor_congestion_index',
    ],
    'leaf_queue_depth': ['leaf_packet_loss_rate', 'leaf_buffer_drops'],
    'leaf_packet_loss_rate': ['server_retransmit_rate'],
    'server_retransmit_rate': [
        'server_conn_timeout_rate',
        'server_app_latency',
    ],
    'tor_congestion_index': ['server_app_latency'],
    'bgp_route_flaps': [
        'bgp_convergence_time',
        'ecmp_path_imbalance',
        'routing_loop_detected',
    ],
    'bgp_update_rate': [
        'spine_cpu_util',
        'leaf_cpu_util',
        'arp_request_rate',
        'lldp_frame_rate',
    ],
    'vtep_reachability': [
        'vxlan_encap_error_rate',
        'vxlan_decap_error_rate',
    ],
}


def all_metrics():
    nodes = set()
    for parent, kids in DAG.items():
        nodes.add(parent)
        nodes.update(kids)
    return sorted(nodes)


def edge_count():
    return sum(len(v) for v in DAG.values())


def descendants(node):
    seen = set()
    stack = [node]
    while stack:
        cur = stack.pop()
        for k in DAG.get(cur, []):
            if k not in seen:
                seen.add(k)
                stack.append(k)
    return seen


def parents_of():
    parents = {m: [] for m in all_metrics()}
    for parent, kids in DAG.items():
        for child in kids:
            parents[child].append(parent)
    return parents


if __name__ == '__main__':
    print(f'Edges: {edge_count()}')
    print(f'Nodes (in DAG): {len(all_metrics())}')
    print(f'Roots: {ROOT_METRICS}')
