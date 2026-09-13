#!/usr/bin/env python3
"""
Generate Figure 2 from dag_structure_for_fig2.json.
Outputs: PNG, PDF
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches


INPUT_JSON = 'dag_structure_for_fig2.json'
OUTPUT_PNG = 'figure2_datacenter_dag.png'
OUTPUT_PDF = 'figure2_datacenter_dag.pdf'

DISPLAY_ALIASES = {
    'arp_request_rate': 'arp_req',
    'bgp_convergence_time': 'bgp\nconv',
    'bgp_route_flaps': 'bgp\nflaps',
    'bgp_update_rate': 'bgp\nupd',
    'ecmp_path_imbalance': 'ecmp\nimbal',
    'leaf_buffer_drops': 'leaf\nbuffer',
    'leaf_cpu_util': 'leaf\ncpu',
    'leaf_link_util_downlink': 'leaf_link\ndown',
    'leaf_link_util_uplink': 'leaf_link\nup',
    'leaf_packet_loss_rate': 'leaf_pkt\nloss',
    'leaf_queue_depth': 'leaf\nqueue',
    'lldp_frame_rate': 'lldp\nframe',
    'optical_rx_power': 'optical\nrx',
    'optical_ber': 'optical\nber',
    'routing_loop_detected': 'route\nloop',
    'server_app_latency': 'srv_app\nlat',
    'server_conn_timeout_rate': 'srv\ntimeout',
    'server_retransmit_rate': 'srv\nretrans',
    'spine_cpu_util': 'spn\ncpu',
    'spine_link_util': 'spn\nlink',
    'spine_packet_loss_rate': 'spn_pkt\nloss',
    'spine_queue_depth': 'spn\nqueue',
    'tor_congestion_index': 'tor\ncong',
    'transceiver_temp': 'xcvr\ntemp',
    'vm_density': 'vm\ndensity',
    'vtep_reachability': 'vtep\nreach',
    'vxlan_decap_error_rate': 'vxlan\ndecap',
    'vxlan_encap_error_rate': 'vxlan\nencap',
}


def format_node_label(node_name):
    parts = node_name.split('_')

    if len(parts) <= 1:
        return node_name
    if len(parts) == 2:
        return "\n".join(parts)

    split_idx = (len(parts) + 1) // 2
    line1 = "_".join(parts[:split_idx])
    line2 = "_".join(parts[split_idx:])
    return f"{line1}\n{line2}"


def measure_label_size_data(ax, label, fontsize=5.8, linespacing=0.88):
    temp = ax.text(
        0.0,
        0.0,
        label,
        ha='center',
        va='center',
        fontsize=fontsize,
        linespacing=linespacing,
        visible=False,
    )
    renderer = ax.figure.canvas.get_renderer()
    bbox = temp.get_window_extent(renderer=renderer)
    temp.remove()

    inv = ax.transData.inverted()
    (x0, y0) = inv.transform((bbox.x0, bbox.y0))
    (x1, y1) = inv.transform((bbox.x1, bbox.y1))

    text_w = abs(x1 - x0)
    text_h = abs(y1 - y0)

    width = text_w * 1.24 + 0.016
    height = text_h * 1.45 + 0.016

    width = min(max(width, 0.16), 0.36)
    height = min(max(height, 0.17), 0.34)
    return width, height


def place_layer_nodes(nodes_in_order, node_sizes, y_value, x_min=0.0, x_max=1.0, gap=0.02):
    if not nodes_in_order:
        return {}, {}

    widths = [node_sizes[n][0] for n in nodes_in_order]
    heights = [node_sizes[n][1] for n in nodes_in_order]

    available = x_max - x_min
    min_gap = 0.008

    required = sum(widths) + gap * (len(widths) - 1)
    if required > available:
        gap = min_gap
        required = sum(widths) + gap * (len(widths) - 1)

    if required > available:
        scale = (available - gap * (len(widths) - 1)) / max(sum(widths), 1e-9)
        scale = max(min(scale, 1.0), 0.75)
        widths = [w * scale for w in widths]

    required = sum(widths) + gap * (len(widths) - 1)
    shift = (available - required) / 2.0
    cursor = x_min + max(shift, 0.0)

    pos_map = {}
    size_map = {}

    for idx, node in enumerate(nodes_in_order):
        w = widths[idx]
        h = heights[idx]
        cx = cursor + w / 2.0
        pos_map[node] = (cx, y_value)
        size_map[node] = (w, h)
        cursor += w
        if idx < len(nodes_in_order) - 1:
            cursor += gap

    return pos_map, size_map


def resolve_row_overlaps(pos, node_sizes, left_bound=0.00, right_bound=0.90, min_gap=0.022, passes=6):
    if not pos:
        return

    rows = {}
    for node, (_, y) in pos.items():
        rows.setdefault(y, []).append(node)

    for _ in range(passes):
        for y in sorted(rows.keys(), reverse=True):
            row_nodes = rows[y]
            if len(row_nodes) <= 1:
                continue

            ordered = sorted(row_nodes, key=lambda n: pos[n][0])
            x_centers = [pos[n][0] for n in ordered]
            widths = [node_sizes[n][0] for n in ordered]

            for i in range(1, len(ordered)):
                prev_x = x_centers[i - 1]
                prev_w = widths[i - 1]
                cur_x = x_centers[i]
                cur_w = widths[i]

                min_center_dist = (prev_w + cur_w) / 2.0 + min_gap
                if cur_x - prev_x < min_center_dist:
                    x_centers[i] = prev_x + min_center_dist

            row_left = min(x - w / 2.0 for x, w in zip(x_centers, widths))
            row_right = max(x + w / 2.0 for x, w in zip(x_centers, widths))
            row_center = (row_left + row_right) / 2.0
            target_center = (left_bound + right_bound) / 2.0
            offset = target_center - row_center

            offset = min(offset, right_bound - row_right)
            offset = max(offset, left_bound - row_left)

            x_centers = [x + offset for x in x_centers]

            for node, x in zip(ordered, x_centers):
                _, node_y = pos[node]
                pos[node] = (x, node_y)


def center_layer_nodes(pos, node_sizes, layers, layer_id, target_center=0.50, min_bound=0.04, max_bound=0.96):
    row_nodes = [node for node in pos if layers.get(node) == layer_id]
    if not row_nodes:
        return

    row_left = min(pos[node][0] - node_sizes[node][0] / 2.0 for node in row_nodes)
    row_right = max(pos[node][0] + node_sizes[node][0] / 2.0 for node in row_nodes)
    row_center = (row_left + row_right) / 2.0
    shift = target_center - row_center

    shift = min(shift, max_bound - row_right)
    shift = max(shift, min_bound - row_left)

    for node in row_nodes:
        x, y = pos[node]
        pos[node] = (x + shift, y)


def main():
    with open(INPUT_JSON, 'r') as f:
        dag_data = json.load(f)

    edges = [tuple(edge) for edge in dag_data.get('edges', [])]
    raw_layers = dag_data.get('layers', {})

    if not isinstance(raw_layers, dict):
        raise ValueError('Expected layers to be a dict in v2 JSON.')

    layers = {node: int(level) for node, level in raw_layers.items()}

    active_nodes = set(layers.keys())
    for u, v in edges:
        active_nodes.add(u)
        active_nodes.add(v)

    longest_path = dag_data.get('longest_path', [])

    fig, ax = plt.subplots(figsize=(3.45, 4.35), dpi=300)

    layer_y = {1: 4.0, 2: 3.25, 3: 2.5, 4: 1.75, 5: 1.0}
    layer_names = {
        1: "Physical/\nRoot",
        2: "Control/\nPhysical",
        3: "Device\nMetric",
        4: "Loss/\nCongestion",
        5: "Server/\nApplication",
    }

    layer_nodes = {}
    for node in active_nodes:
        layer = layers.get(node)
        if layer is not None:
            layer_nodes.setdefault(layer, []).append(node)

    node_labels = {
        node: DISPLAY_ALIASES.get(node, format_node_label(node))
        for node in active_nodes
    }

    ax.set_xlim(-0.36, 1.12)
    ax.set_ylim(0.55, 4.35)
    ax.axis('off')
    fig.canvas.draw()

    node_sizes = {
        node: measure_label_size_data(ax, node_labels[node], fontsize=5.8, linespacing=0.88)
        for node in active_nodes
    }

    pos = {}
    for layer, nodes_list in layer_nodes.items():
        ordered_nodes = sorted(nodes_list)
        n = len(ordered_nodes)

        if layer == 3:
            preferred_order = [
                'bgp_convergence_time',
                'ecmp_path_imbalance',
                'leaf_cpu_util',
                'leaf_queue_depth',
                'routing_loop_detected',
                'spine_cpu_util',
                'spine_queue_depth',
                'vxlan_decap_error_rate',
                'vxlan_encap_error_rate',
            ]
            ordered_nodes = [node for node in preferred_order if node in ordered_nodes] + [
                node for node in ordered_nodes if node not in preferred_order
            ]

            row_pos, row_sizes = place_layer_nodes(
                ordered_nodes,
                node_sizes,
                y_value=layer_y[layer],
                x_min=0.01,
                x_max=0.99,
                gap=0.014,
            )
            pos.update(row_pos)
            node_sizes.update(row_sizes)
            continue

        if n >= 8:
            upper = ordered_nodes[::2]
            lower = ordered_nodes[1::2]

            upper_pos, upper_sizes = place_layer_nodes(
                upper,
                node_sizes,
                y_value=layer_y[layer] + 0.16,
                x_min=0.02,
                x_max=0.98,
                gap=0.020,
            )
            lower_pos, lower_sizes = place_layer_nodes(
                lower,
                node_sizes,
                y_value=layer_y[layer] - 0.16,
                x_min=0.02,
                x_max=0.98,
                gap=0.020,
            )
            pos.update(upper_pos)
            pos.update(lower_pos)
            node_sizes.update(upper_sizes)
            node_sizes.update(lower_sizes)
        else:
            if layer == 1:
                row_gap = 0.050
                row_x_min = 0.10
                row_x_max = 0.90
            elif layer == 2:
                row_gap = 0.042
                row_x_min = 0.06
                row_x_max = 0.94
            elif layer == 4:
                row_gap = 0.054
                row_x_min = 0.06
                row_x_max = 0.94
            elif layer == 5:
                row_gap = 0.042
                row_x_min = 0.08
                row_x_max = 0.92
            else:
                row_gap = 0.022
                row_x_min = 0.03
                row_x_max = 0.97

            row_pos, row_sizes = place_layer_nodes(
                ordered_nodes,
                node_sizes,
                y_value=layer_y[layer],
                x_min=row_x_min,
                x_max=row_x_max,
                gap=row_gap,
            )
            pos.update(row_pos)
            node_sizes.update(row_sizes)

    for node in list(node_sizes.keys()):
        width, height = node_sizes[node]
        node_sizes[node] = (width * 1.08, height)

    for node in list(node_sizes.keys()):
        if layers.get(node) == 4:
            width, height = node_sizes[node]
            node_sizes[node] = (width * 1.06, height)

    # Increase ellipse width by 15% for Control/Physical row only
    for node in list(node_sizes.keys()):
        if layers.get(node) == 2:
            width, height = node_sizes[node]
            node_sizes[node] = (width * 1.15 * 1.10, height)

    # Add a small selective size boost for still-long labels
    selective_boost_nodes = {
        'leaf_packet_loss_rate',
        'spine_packet_loss_rate',
        'tor_congestion_index',
        'server_app_latency',
        'vxlan_encap_error_rate',
        'vxlan_decap_error_rate',
        'leaf_link_util_downlink',
    }
    for node in selective_boost_nodes:
        if node in node_sizes:
            width, height = node_sizes[node]
            node_sizes[node] = (width * 1.05, height * 1.04)

    # Additional +10% width for requested four nodes
    extra_width_nodes = {
        'leaf_link_util_downlink',
        'leaf_link_util_uplink',
        'server_conn_timeout_rate',
        'server_retransmit_rate',
    }
    for node in extra_width_nodes:
        if node in node_sizes:
            width, height = node_sizes[node]
            node_sizes[node] = (width * 1.10, height)

    # Additional +5% width for queue nodes
    queue_width_nodes = {
        'leaf_queue_depth',
        'spine_queue_depth',
    }
    for node in queue_width_nodes:
        if node in node_sizes:
            width, height = node_sizes[node]
            node_sizes[node] = (width * 1.05, height)

    resolve_row_overlaps(pos, node_sizes, left_bound=0.04, right_bound=0.96, min_gap=0.022, passes=6)

    if pos:
        min_left = min(pos[node][0] - node_sizes[node][0] / 2.0 for node in pos)
        max_right = max(pos[node][0] + node_sizes[node][0] / 2.0 for node in pos)
        current_center = (min_left + max_right) / 2.0
        offset = 0.50 - current_center

        left_bound = 0.04
        right_bound = 0.96
        offset = min(offset, right_bound - max_right)
        offset = max(offset, left_bound - min_left)

        for node in list(pos.keys()):
            x, y = pos[node]
            pos[node] = (x + offset, y)

    resolve_row_overlaps(pos, node_sizes, left_bound=0.04, right_bound=0.96, min_gap=0.022, passes=4)

    center_layer_nodes(pos, node_sizes, layers, layer_id=2, target_center=0.52, min_bound=0.06, max_bound=0.96)
    center_layer_nodes(pos, node_sizes, layers, layer_id=3, target_center=0.52, min_bound=0.04, max_bound=0.96)
    center_layer_nodes(pos, node_sizes, layers, layer_id=1, target_center=0.52, min_bound=0.06, max_bound=0.96)
    center_layer_nodes(pos, node_sizes, layers, layer_id=4, target_center=0.52, min_bound=0.06, max_bound=0.96)
    center_layer_nodes(pos, node_sizes, layers, layer_id=5, target_center=0.52, min_bound=0.08, max_bound=0.94)

    longest_path_nodes = set(longest_path)
    longest_path_edges = set()
    for i in range(len(longest_path) - 1):
        longest_path_edges.add((longest_path[i], longest_path[i + 1]))

    for u, v in edges:
        if u not in pos or v not in pos:
            continue

        x1, y1 = pos[u]
        x2, y2 = pos[v]

        if (u, v) in longest_path_edges:
            ax.plot([x1, x2], [y1, y2], color='#902020', linewidth=1.9, zorder=1, alpha=1.0, clip_on=False)
        else:
            ax.plot([x1, x2], [y1, y2], color='#808080', linewidth=0.9, zorder=1, alpha=0.72, clip_on=False)

    for node in active_nodes:
        if node not in pos:
            continue

        x, y = pos[node]

        if node in longest_path_nodes:
            node_color = 'white'
            border_color = '#902020'
            border_width = 1.7
        else:
            node_color = 'white'
            border_color = '#808080'
            border_width = 1.0

        node_width, node_height = node_sizes[node]
        node_height = node_height * 1.5 * 1.15

        ellipse = patches.Ellipse(
            (x, y),
            width=node_width,
            height=node_height,
            facecolor=node_color,
            edgecolor=border_color,
            linewidth=border_width,
            zorder=2,
            clip_on=False,
        )
        ax.add_patch(ellipse)

        ax.text(
            x,
            y,
            node_labels[node],
            ha='center',
            va='center',
            fontsize=5.8,
            color='black',
            weight='normal',
            zorder=3,
            linespacing=0.88,
        )

    for layer, y in layer_y.items():
        if layer in layer_names:
            ax.text(
                -0.31,
                y,
                layer_names[layer],
                ha='left',
                va='center',
                fontsize=7.4,
                color='black',
                weight='bold',
                alpha=1.0,
            )

    legend_elements = [
        plt.Line2D([0], [0], color='#902020', linewidth=1.9, label='Longest path edge'),
        plt.Line2D([0], [0], color='#808080', linewidth=0.8, label='Other edge'),
    ]
    ax.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.055),
        ncol=2,
        frameon=False,
        fontsize=7.2,
    )

    # Title removed for IEEE formatting (caption provides context)
    # ax.set_title('Datacenter Causal DAG', fontsize=9.4, pad=6)

    plt.tight_layout(pad=0.3)
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_PDF, dpi=300, bbox_inches='tight')

    print(f"✓ Figure 2 saved: {OUTPUT_PNG}")
    print(f"✓ Figure 2 saved: {OUTPUT_PDF}")
    print(f"  Active nodes: {len(active_nodes)}, Edges: {len(edges)}")
    print(f"  Longest path length: {len(longest_path)}")

    plt.close()


if __name__ == '__main__':
    main()
