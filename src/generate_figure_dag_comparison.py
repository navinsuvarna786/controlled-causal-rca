"""Generate side-by-side DAG comparison: experience-informed (ours) vs
data-driven (shrinkage discovery on AIOps).

Left panel: 28-node, 26-edge architecture-informed datacenter DAG, laid
out in 5 layers from physical/control roots to application leaves.

Right panel: 40-node, 1-edge data-driven AIOps DAG, recovered by
shrinkage-regularized partial correlation from healthy telemetry.

Output: outputs/figures/figure_dag_comparison.{png,pdf}
"""

import json
import pathlib

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

DATA_DIR = pathlib.Path('data')
OUT_DIR = pathlib.Path('outputs/figures')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_datacenter_dag():
    structure = json.loads((DATA_DIR / 'dag_structure_for_fig2.json').read_text())
    layers = structure['layers']
    G = nx.DiGraph()
    if isinstance(layers, dict):
        if all(isinstance(v, list) for v in layers.values()):
            for layer_name, nodes in layers.items():
                for node in nodes:
                    G.add_node(node, layer=layer_name)
        else:
            for node, layer in layers.items():
                G.add_node(node, layer=str(layer))
    edges = structure.get('edges', [])
    for edge in edges:
        if isinstance(edge, list) and len(edge) == 2:
            G.add_edge(edge[0], edge[1])
        elif isinstance(edge, dict):
            G.add_edge(edge['source'], edge['target'])
    return G


def load_aiops_dag():
    summary = json.loads((DATA_DIR / 'aiops_structural_summary.json').read_text())
    G = nx.DiGraph()
    nodes = summary.get('nodes', [])
    if not nodes:
        path_nodes = summary.get('longest_path_nodes', [])
        nodes = list(set(path_nodes))
        layers = json.loads((DATA_DIR / 'aiops_reduced_subgraph.json').read_text())
        if 'nodes' in layers:
            for n in layers['nodes']:
                if isinstance(n, str):
                    nodes.append(n)
                elif isinstance(n, dict) and 'id' in n:
                    nodes.append(n['id'])
        nodes = list(dict.fromkeys(nodes))
    for n in nodes:
        G.add_node(n)
    path_nodes = summary.get('longest_path_nodes', [])
    edges = summary.get('edges', [])
    if not edges and len(path_nodes) >= 2:
        for i in range(len(path_nodes) - 1):
            edges.append([path_nodes[i], path_nodes[i + 1]])
    for e in edges:
        if isinstance(e, list) and len(e) == 2:
            G.add_edge(e[0], e[1])
        elif isinstance(e, dict):
            G.add_edge(e['source'], e['target'])
    return G


def layered_layout(G, layer_order):
    pos = {}
    layers_to_nodes = {}
    for node, attrs in G.nodes(data=True):
        layer = attrs.get('layer', 'unknown')
        layers_to_nodes.setdefault(layer, []).append(node)
    for li, layer in enumerate(layer_order):
        nodes = layers_to_nodes.get(layer, [])
        for ni, n in enumerate(sorted(nodes)):
            x = (ni + 1) / (len(nodes) + 1)
            y = 1.0 - li / max(1, len(layer_order) - 1)
            pos[n] = (x, y)
    extras = [n for n in G.nodes() if n not in pos]
    for ni, n in enumerate(extras):
        pos[n] = ((ni + 1) / (len(extras) + 1), 0.5)
    return pos


def main():
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6),
                             gridspec_kw={'width_ratios': [1.0, 1.0]})

    # Left: datacenter experience DAG
    ax = axes[0]
    G_dc = load_datacenter_dag()
    layer_order = sorted(set(nx.get_node_attributes(G_dc, 'layer').values()))
    pos_dc = layered_layout(G_dc, layer_order)
    if not pos_dc or len(pos_dc) < len(G_dc.nodes):
        pos_dc = nx.kamada_kawai_layout(G_dc)
    nx.draw_networkx_edges(
        G_dc, pos_dc, ax=ax,
        edge_color='#404040', width=0.7, alpha=0.6,
        arrows=True, arrowsize=8, arrowstyle='-|>',
        connectionstyle='arc3,rad=0.05',
    )
    nx.draw_networkx_nodes(
        G_dc, pos_dc, ax=ax,
        node_color='#5A8FBF', node_size=140, edgecolors='black',
        linewidths=0.6,
    )
    ax.set_title(
        f'(a) Architecture-informed DAG\n'
        f'{G_dc.number_of_nodes()} nodes, {G_dc.number_of_edges()} edges, '
        f'longest path 5',
        fontsize=10,
    )
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.10, 1.10)
    ax.set_axis_off()

    # Right: AIOps data-driven DAG
    ax = axes[1]
    G_ai = load_aiops_dag()
    pos_ai = nx.kamada_kawai_layout(G_ai) if G_ai.number_of_nodes() > 1 else {n: (0.5, 0.5) for n in G_ai.nodes()}

    if G_ai.number_of_edges() > 0:
        edge_nodes = set()
        for u, v in G_ai.edges():
            edge_nodes.add(u)
            edge_nodes.add(v)
    else:
        edge_nodes = set()

    isolated = [n for n in G_ai.nodes() if n not in edge_nodes]
    if isolated and edge_nodes:
        cx, cy = 0.5, 0.5
        for u, v in G_ai.edges():
            pos_ai[u] = (0.4, 0.5)
            pos_ai[v] = (0.6, 0.5)
        n_iso = len(isolated)
        cols = max(4, int(n_iso ** 0.5) + 1)
        rows = (n_iso + cols - 1) // cols
        for idx, n in enumerate(sorted(isolated)):
            r = idx // cols
            c = idx % cols
            x = 0.05 + (c + 0.5) * (0.9 / cols)
            y = 0.92 - (r + 0.5) * (0.7 / max(1, rows))
            if 0.4 - 0.06 <= x <= 0.6 + 0.06 and 0.46 <= y <= 0.54:
                continue
            pos_ai[n] = (x, y)

    nx.draw_networkx_edges(
        G_ai, pos_ai, ax=ax,
        edge_color='#902020', width=2.0, alpha=0.9,
        arrows=True, arrowsize=14, arrowstyle='-|>',
    )
    nx.draw_networkx_nodes(
        G_ai, pos_ai, ax=ax,
        nodelist=list(edge_nodes), node_color='#C04444',
        node_size=200, edgecolors='black', linewidths=0.8,
    )
    nx.draw_networkx_nodes(
        G_ai, pos_ai, ax=ax,
        nodelist=isolated, node_color='#BBBBBB',
        node_size=60, edgecolors='#666666', linewidths=0.4,
    )
    ax.set_title(
        f'(b) Data-driven DAG (shrinkage discovery)\n'
        f'{G_ai.number_of_nodes()} nodes, {G_ai.number_of_edges()} edge, '
        f'longest path 1',
        fontsize=10,
    )
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.10, 1.10)
    ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'figure_dag_comparison.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT_DIR / 'figure_dag_comparison.pdf', bbox_inches='tight')
    print(f"Saved: {OUT_DIR}/figure_dag_comparison.{{png,pdf}}")


if __name__ == '__main__':
    main()
