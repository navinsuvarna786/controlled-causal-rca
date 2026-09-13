#!/usr/bin/env python3
"""
Generate Figure 3: Structural Comparison with split subpanels
Addresses y-axis compression from large root ratio (4.33).
Uses blue/gray palette: blue for Datacenter, gray + hatch for AIOps.
"""
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Read structural summaries
with open('datacenter_structural_summary.json', 'r') as f:
    dc_stats = json.load(f)

with open('aiops_structural_summary.json', 'r') as f:
    aiops_stats = json.load(f)

# Prepare comparison metrics
dc_values = {
    'Nodes': dc_stats['node_count'],
    'Edges': dc_stats['edge_count'],
    'Density': dc_stats['density'],
    'Roots': dc_stats['number_of_roots'],
    'Leaves': dc_stats['number_of_leaves'],
    'Longest Path': dc_stats['longest_path_length']
}

aiops_values = {
    'Nodes': aiops_stats['node_count'],
    'Edges': aiops_stats['edge_count'],
    'Density': aiops_stats['density'],
    'Roots': aiops_stats['number_of_roots'],
    'Leaves': aiops_stats['number_of_leaves'],
    'Longest Path': aiops_stats['longest_path_length']
}

# Split metrics into two groups
# Group 1: Nodes, Edges, Leaves, Longest Path (similar scales)
# Group 2: Roots, Density (different scales - roots has large ratio)

group1_metrics = ['Nodes', 'Edges', 'Leaves', 'Longest\nPath']
group2_metrics = ['Roots', 'Density']

group1_dc = [dc_values['Nodes'], dc_values['Edges'], dc_values['Leaves'], dc_values['Longest Path']]
group1_aiops = [aiops_values['Nodes'], aiops_values['Edges'], aiops_values['Leaves'], aiops_values['Longest Path']]

group2_dc = [dc_values['Roots'], dc_values['Density']]
group2_aiops = [aiops_values['Roots'], aiops_values['Density']]

# Create figure with two subpanels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3), dpi=300)

# Blue/gray colors
dc_color = '#2B6CB0'      # Blue
aiops_color = '#B0B7C3'   # Gray

width = 0.35

# ===== LEFT PANEL: Nodes, Edges, Leaves, Longest Path =====
x1 = np.arange(len(group1_metrics))

bars1_dc = ax1.bar(x1 - width/2, group1_dc, width, label='Datacenter',
                   color=dc_color, edgecolor='black', linewidth=0.8)
bars1_aiops = ax1.bar(x1 + width/2, group1_aiops, width, label='AIOps',
                      color=aiops_color, edgecolor='black', linewidth=0.8,
                      hatch='///')

# Add value labels on bars
def add_value_labels(ax, bars, values):
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.02,
               f'{int(val)}', ha='center', va='bottom', fontsize=8, fontweight='bold')

add_value_labels(ax1, bars1_dc, group1_dc)
add_value_labels(ax1, bars1_aiops, group1_aiops)

ax1.set_ylabel('Absolute Count', fontsize=10, fontweight='bold')
ax1.set_xticks(x1)
ax1.set_xticklabels(group1_metrics, fontsize=9)
ax1.set_ylim(0, max(max(group1_dc), max(group1_aiops)) * 1.15)
ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
ax1.set_axisbelow(True)
ax1.set_title('(a) Structural Properties', fontsize=10, fontweight='bold', pad=8)

# ===== RIGHT PANEL: Roots, Density =====
x2 = np.arange(len(group2_metrics))

bars2_dc = ax2.bar(x2 - width/2, group2_dc, width,
                   color=dc_color, edgecolor='black', linewidth=0.8)
bars2_aiops = ax2.bar(x2 + width/2, group2_aiops, width,
                      color=aiops_color, edgecolor='black', linewidth=0.8,
                      hatch='///')

# Add value labels with appropriate formatting
for bar, val in zip(bars2_dc, group2_dc):
    height = bar.get_height()
    if val < 1:
        label = f'{val:.3f}'
    else:
        label = f'{int(val)}'
    ax2.text(bar.get_x() + bar.get_width()/2., height + max(group2_dc)*0.02,
            label, ha='center', va='bottom', fontsize=8, fontweight='bold')

for bar, val in zip(bars2_aiops, group2_aiops):
    height = bar.get_height()
    if val < 1:
        label = f'{val:.4f}'
    else:
        label = f'{int(val)}'
    ax2.text(bar.get_x() + bar.get_width()/2., height + max(group2_aiops)*0.02,
            label, ha='center', va='bottom', fontsize=8, fontweight='bold')

ax2.set_ylabel('Absolute Value', fontsize=10, fontweight='bold')
ax2.set_xticks(x2)
ax2.set_xticklabels(group2_metrics, fontsize=9)
ax2.set_ylim(0, max(max(group2_dc), max(group2_aiops)) * 1.15)
ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
ax2.set_axisbelow(True)
ax2.set_title('(b) Roots & Density', fontsize=10, fontweight='bold', pad=8)

# Shared legend below both panels
handles = [
    mpatches.Patch(facecolor=dc_color, edgecolor='black', label='Datacenter (Synthetic)'),
    mpatches.Patch(facecolor=aiops_color, edgecolor='black', hatch='///', label='AIOps (Real)')
]
fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.05),
          ncol=2, frameon=False, fontsize=9)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig('figure3_structural_comparison.png', dpi=300,
           bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('figure3_structural_comparison.pdf', dpi=300,
           bbox_inches='tight', facecolor='white', edgecolor='none')
print("✓ Figure 3 saved: PNG and PDF")
plt.close()

# Print statistics for caption
density_ratio = aiops_stats['density'] / dc_stats['density']
print(f"\nStatistics for caption:")
print(f"  Datacenter: {dc_stats['node_count']} nodes, {dc_stats['edge_count']} edges, density {dc_stats['density']:.4f}")
print(f"  AIOps: {aiops_stats['node_count']} nodes, {aiops_stats['edge_count']} edges, density {aiops_stats['density']:.6f}")
print(f"  Density ratio: {density_ratio:.4f} (≈ 1/{1/density_ratio:.1f}, AIOps is ~{1/density_ratio:.0f}× sparser)")
