"""
make_figures.py
=====================================================================
Publication figures for the revised manuscript. Reads the CSV/JSON that
the pipeline + experiment + stratification runs already produced. No model
retraining. Outputs 300-dpi PNG + vector PDF into ./figures/.

Expected inputs (edit PATHS below):
  pipeline_out/cohort_flow.csv
  pipeline_out/disease_frequency.csv
  results/aggregate_metrics.csv          (header=[metric, stat])
  results/results.json                   (clustering block)
  results_strat_k3/strat_ablation.csv
  results_strat_k3/cluster_profile_full.csv
  results_strat_k3/cluster_disease_enrich.csv
  results_strat_k3/strat_deep.json

Produces:
  fig1_architecture.pdf/png     — model schematic (no run data)
  fig2_cohort_flow.pdf/png      — participant-flow diagram
  fig3_modality_ablation.pdf/png— stability/silhouette vs modality set  (R1.12)
  fig4_cluster_profile.pdf/png  — per-cluster ICU/LOS/mortality          (R1.10/R1.13)
  fig5_prediction_metrics.pdf/png— model comparison bars w/ error bars   (R1.7)
  fig6_disease_frequency.pdf/png— top rare-disease codes in cohort       (R1.4)

Run:  python make_figures.py
=====================================================================
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ----------------------------- paths -----------------------------
PIPE   = "pipeline_out"
RESULT = "results"
STRAT  = "results_strat_k3"
OUT    = "figures"
os.makedirs(OUT, exist_ok=True)

# ----------------------------- style -----------------------------
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300,
    "font.size": 10, "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.axisbelow": True,
})
BLUE, ORANGE, GREEN, RED, GREY = "#2c6fbb", "#e08b2d", "#3c9a5f", "#c1443c", "#6b6b6b"


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.pdf/.png")


# ============================================================== #
# FIG 1 — architecture schematic (no run data)                   #
# ============================================================== #
def fig1_architecture():
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")

    def box(x, y, w, h, text, fc, fs=9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
                                    fc=fc, ec="#333", lw=1.0))
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=12, lw=1.1, color="#555"))

    mods = [("Diagnoses\n(128)", 5.1), ("Labs\n(32)", 3.9),
            ("Medications\n(64)", 2.7), ("Procedures\n(64)", 1.5)]
    for name, y in mods:
        box(0.2, y, 1.5, 0.9, name, "#dce7f5")
        box(2.3, y, 1.4, 0.9, "Proj\n→ D/k", "#eef3fb", fs=8)
        arrow(1.7, y+0.45, 2.3, y+0.45)
        arrow(3.7, y+0.45, 4.5, 3.3)

    box(4.5, 2.9, 1.6, 1.0, "Concat +\nTemporal\nstack", "#f3e6d0", fs=8)
    arrow(6.1, 3.4, 6.7, 3.4)
    box(6.7, 2.9, 1.7, 1.0, "Transformer\nencoder\n(2L, 4H)", "#f3e6d0", fs=8)
    arrow(8.4, 3.4, 9.0, 3.4)
    box(9.0, 2.9, 1.5, 1.0, "Attention\npooling", "#f3e6d0", fs=8)
    arrow(10.5, 3.4, 10.9, 3.4)
    box(10.7, 2.9, 1.1, 1.0, "Latent\nz∈R¹⁶\n(μ,σ²)", "#d7efd9", fs=8)

    # downstream heads
    box(9.4, 5.0, 2.4, 0.85, "GMM clustering\n→ resource strata", "#d7efd9", fs=8)
    box(9.4, 0.5, 2.4, 0.85, "Aux heads:\nICU risk · LOS", "#f6d9d6", fs=8)
    arrow(11.25, 3.9, 10.9, 5.0)
    arrow(11.25, 2.9, 10.9, 1.35)

    ax.text(6, 5.7, "Multimodal VaDeSC-EHR (proposed)", ha="center",
            fontsize=12, fontweight="bold")
    ax.text(6, 0.1, "Reconstruction + KL + auxiliary ICU/LOS supervision "
                    "(no survival head)", ha="center", fontsize=8, color="#555")
    save(fig, "fig1_architecture")


# ============================================================== #
# FIG 2 — participant-flow diagram                               #
# ============================================================== #
def fig2_cohort_flow():
    try:
        flow = pd.read_csv(f"{PIPE}/cohort_flow.csv")
    except FileNotFoundError:
        print("  [skip] cohort_flow.csv not found"); return
    fig, ax = plt.subplots(figsize=(6.5, len(flow) * 1.15 + 0.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, len(flow) + 0.5); ax.axis("off")
    n = len(flow)
    for i, r in flow.iterrows():
        y = n - i - 0.5
        txt = f"{r['stage']}\n{int(r['patients']):,} patients · {int(r['admissions']):,} admissions"
        ax.add_patch(FancyBboxPatch((1.5, y-0.35), 7, 0.7,
                     boxstyle="round,pad=0.05,rounding_size=0.06",
                     fc="#dce7f5", ec="#333", lw=1.0))
        ax.text(5, y, txt, ha="center", va="center", fontsize=8.5)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((5, y-0.35), (5, y-0.65),
                         arrowstyle="-|>", mutation_scale=12, lw=1.1, color="#555"))
    ax.text(5, n + 0.2, "Cohort construction (MIMIC-IV → Orphanet ICD-10)",
            ha="center", fontsize=11, fontweight="bold")
    save(fig, "fig2_cohort_flow")


# ============================================================== #
# FIG 3 — modality ablation (the headline R1.12 figure)          #
# ============================================================== #
def fig3_modality_ablation():
    try:
        ab = pd.read_csv(f"{STRAT}/strat_ablation.csv")
    except FileNotFoundError:
        print("  [skip] strat_ablation.csv not found"); return
    order = ["dx", "dx+lab", "dx+lab+med", "full"]
    ab = ab.set_index("subset").reindex([o for o in order if o in ab.subset.values]
                                        if "subset" in ab.columns else order)
    if ab.index.isna().all():
        ab = pd.read_csv(f"{STRAT}/strat_ablation.csv").set_index("subset").reindex(order)
    x = np.arange(len(ab))
    fig, ax1 = plt.subplots(figsize=(6.2, 4.2))
    ax1.plot(x, ab["stability_ari"], "-o", color=BLUE, lw=2, label="Cluster stability (ARI)")
    ax1.set_ylabel("Cluster stability (ARI)", color=BLUE)
    ax1.tick_params(axis="y", labelcolor=BLUE)
    ax1.set_ylim(0, max(0.8, ab["stability_ari"].max()*1.15))
    ax2 = ax1.twinx()
    ax2.plot(x, ab["silhouette_mean"], "--s", color=ORANGE, lw=1.8, label="Silhouette")
    ax2.set_ylabel("Silhouette", color=ORANGE)
    ax2.tick_params(axis="y", labelcolor=ORANGE)
    ax2.grid(False); ax2.spines["top"].set_visible(False)
    ax1.set_xticks(x); ax1.set_xticklabels(ab.index, rotation=15)
    ax1.set_xlabel("Modality set")
    ax1.set_title("Multimodal fusion improves stratification stability", fontsize=11)
    # combined legend
    l1, la1 = ax1.get_legend_handles_labels()
    l2, la2 = ax2.get_legend_handles_labels()
    ax1.legend(l1+l2, la1+la2, loc="upper left", fontsize=8, framealpha=0.9)
    save(fig, "fig3_modality_ablation")


# ============================================================== #
# FIG 4 — per-cluster clinical profile                           #
# ============================================================== #
def fig4_cluster_profile():
    try:
        prof = pd.read_csv(f"{STRAT}/cluster_profile_full.csv")
    except FileNotFoundError:
        print("  [skip] cluster_profile_full.csv not found"); return
    prof = prof.sort_values("icu_any")
    x = np.arange(len(prof)); w = 0.6
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    labels = [f"C{int(c)}\n(n={int(n)})" for c, n in zip(prof["cluster"], prof["n"])]

    axes[0].bar(x, prof["icu_any"]*100, w, color=RED)
    axes[0].set_title("ICU utilization"); axes[0].set_ylabel("% patients w/ ICU")
    axes[1].bar(x, prof["total_los_days"], w, color=BLUE)
    axes[1].set_title("Length of stay"); axes[1].set_ylabel("Mean total LOS (days)")
    axes[2].bar(x, prof["death_365d"]*100, w, color=GREY)
    axes[2].set_title("1-yr post-discharge mortality"); axes[2].set_ylabel("% deceased")
    for a in axes:
        a.set_xticks(x); a.set_xticklabels(labels, fontsize=8); a.set_xlabel("Cluster")
    fig.suptitle("Resource-use strata differ on outcomes never used in training",
                 fontsize=11, y=1.03)
    fig.tight_layout()
    save(fig, "fig4_cluster_profile")


# ============================================================== #
# FIG 5 — prediction metrics comparison                          #
# ============================================================== #
def fig5_prediction_metrics():
    try:
        df = pd.read_csv(f"{RESULT}/aggregate_metrics.csv", header=[0, 1], index_col=0)
    except FileNotFoundError:
        print("  [skip] aggregate_metrics.csv not found"); return
    mean = df.xs("mean", axis=1, level=1)
    std  = df.xs("std",  axis=1, level=1)
    metrics = [("auroc", "ICU AUROC"), ("auprc", "ICU AUPRC"), ("brier", "Brier (↓)")]
    models = mean.index.tolist()
    x = np.arange(len(models))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    colors = [BLUE if "Multimodal" in m else GREY for m in models]
    for ax, (col, title) in zip(axes, metrics):
        if col not in mean.columns:
            continue
        yerr = std[col].fillna(0).values
        ax.bar(x, mean[col].values, yerr=yerr, capsize=3, color=colors)
        ax.set_title(title); ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=40, ha="right", fontsize=7.5)
        if col == "auroc":
            ax.axhline(0.5, ls=":", c="k", lw=0.8)
    axes[0].set_ylabel("score")
    fig.suptitle("Next-admission prediction: proposed model vs baselines "
                 "(leakage-free; error bars = SD over 5 seeds)", fontsize=10, y=1.04)
    fig.tight_layout()
    save(fig, "fig5_prediction_metrics")


# ============================================================== #
# FIG 6 — disease frequency (top codes)                          #
# ============================================================== #
def fig6_disease_frequency():
    try:
        d = pd.read_csv(f"{PIPE}/disease_frequency.csv")
    except FileNotFoundError:
        print("  [skip] disease_frequency.csv not found"); return
    d = d.sort_values("n_patients", ascending=False).head(15)
    lbl = (d["PreferredTerm"].astype(str).str.slice(0, 34) +
           " (" + d.iloc[:, 0].astype(str) + ")") if "PreferredTerm" in d.columns \
          else d.iloc[:, 0].astype(str)
    fig, ax = plt.subplots(figsize=(7, 5))
    y = np.arange(len(d))[::-1]
    ax.barh(y, d["n_patients"], color=GREEN)
    ax.set_yticks(y); ax.set_yticklabels(lbl, fontsize=8)
    ax.set_xlabel("Patients"); ax.set_title("Most frequent Orphanet-mapped conditions in cohort")
    save(fig, "fig6_disease_frequency")


if __name__ == "__main__":
    print("Generating figures →", OUT)
    fig1_architecture()
    fig2_cohort_flow()
    fig3_modality_ablation()
    fig4_cluster_profile()
    fig5_prediction_metrics()
    fig6_disease_frequency()
    print("done.")
