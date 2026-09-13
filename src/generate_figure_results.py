#!/usr/bin/env python3
"""
Main comparative figure: root-cause ranking accuracy of four scoring rules
over the same architecture-derived DAG.

Two panels make the paper's central point visible at a glance:
  (a) the full candidate set of Eq. (2), where the rules separate sharply;
  (b) the restricted candidate set used by much of the literature, where the
      same rules become statistically indistinguishable.

Colour is a redundant channel, never the only one: every bar also carries a
hatch pattern and a printed value, so the figure survives greyscale printing
and colour-vision deficiency. The four hues are categorical slots validated for
CVD separation against the chart surface. A rule keeps the same colour in both
panels so the reader can track it across them. Reads full_candidate_eval.json
and final_results.json.
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_STEM = "figure_results_comparison"

# Validated categorical slots; identity is carried by colour AND hatch AND label.
C_RESID, C_OURS, C_CF, C_ANOM = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

FULL = [("residual", "Residual (RR-RCA)", C_RESID, ""),
        ("counterfactual_z", "Counterfactual, corrected (ours)", C_OURS, "///"),
        ("counterfactual", "Counterfactual, uncorrected", C_CF, "\\\\\\"),
        ("anomaly", "Anomaly magnitude", C_ANOM, "...")]

REST = [("rr_rca", "Residual (RR-RCA)", C_RESID, ""),
        ("ours", "Counterfactual, uncorrected", C_CF, "\\\\\\"),
        ("ours_no_refine", "Anomaly magnitude", C_ANOM, "...")]


def main():
    with open("full_candidate_eval.json", encoding="utf-8") as f:
        full = json.load(f)["top1"]
    with open("final_results.json", encoding="utf-8") as f:
        rest = json.load(f)["aggregate"]

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.5))

    ax = axes[0]
    for i, (k, lab, c, h) in enumerate(FULL):
        ax.bar(i, full[k], width=0.68, color=c, hatch=h,
               edgecolor="black", linewidth=0.6)
        ax.text(i, full[k] + 0.02, f"{full[k]:.3f}", ha="center",
                fontsize=7, color="#2b2b2b")
    ax.set_xticks(range(len(FULL)))
    ax.set_xticklabels([lab for _, lab, _, _ in FULL], fontsize=6.6,
                       rotation=18, ha="right")
    ax.set_title("(a) Full candidate set (Eq. 2)\nmean 9.5 candidates, "
                 "860 incidents", fontsize=7.5)
    ax.set_ylabel("Top-1 accuracy", fontsize=7.5)

    ax = axes[1]
    for i, (k, lab, c, h) in enumerate(REST):
        m, s = rest[k]["top1_mean"], rest[k]["top1_std"]
        ax.bar(i, m, width=0.68, color=c, hatch=h, edgecolor="black",
               linewidth=0.6, yerr=s,
               error_kw=dict(ecolor="black", elinewidth=0.6, capsize=2))
        ax.text(i, m + s + 0.02, f"{m:.3f}", ha="center", fontsize=7,
                color="#2b2b2b")
    ax.set_xticks(range(len(REST)))
    ax.set_xticklabels([lab for _, lab, _, _ in REST], fontsize=6.6,
                       rotation=18, ha="right")
    ax.set_title("(b) Restricted to injected roots\n5 candidates, "
                 "all differences n.s.", fontsize=7.5)

    for ax in axes:
        ax.set_ylim(0, 0.92)
        ax.tick_params(axis="y", labelsize=7)
        ax.yaxis.grid(True, linewidth=0.4, color="#cccccc")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_linewidth(0.6)

    fig.tight_layout(pad=0.4)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT_STEM}.{ext}", dpi=400, bbox_inches="tight")
    print(f"Wrote {OUT_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
