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
  (Fig 1 is the static file figures/fig1_architecture.pdf)
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
    ax1.set_title("Partition stability as modalities are added (Dx \u2192 Lab \u2192 Med \u2192 Proc)", fontsize=10.5)
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
    fig.suptitle("Resource-use strata differ on outcomes not used as inputs or training targets",
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
ICD10CM = {
    "G20": "Parkinson's disease", "C61": "Malignant neoplasm of prostate",
    "I81": "Portal vein thrombosis", "G35": "Multiple sclerosis",
    "N10": "Acute pyelonephritis", "C541": "Malignant neoplasm of endometrium",
    "D65": "Disseminated intravascular coagulation",
    "J80": "Acute respiratory distress syndrome", "C20": "Malignant neoplasm of rectum",
    "C73": "Malignant neoplasm of thyroid gland",
    "C548": "Malignant neoplasm of overlapping sites of corpus uteri",
    "C543": "Malignant neoplasm of fundus uteri", "C540": "Malignant neoplasm of isthmus uteri",
    "C542": "Malignant neoplasm of myometrium",
}
ORPH_CSV = os.environ.get("ORPH_CSV", "icd10_diseases.csv")
D_ICD = os.environ.get("D_ICD", "")   # optional: MIMIC hosp/d_icd_diagnoses.csv


def fig6_disease_frequency():
    """Paper Fig 3: matched ICD-10 codes by patients, labelled with the ICD-10-CM
    description and the number of Orphanet terms mapped to each code."""
    try:
        # final timestamp-valid counts from cohort_flow.py (as in the paper);
        # pipeline_out/disease_frequency.csv counts 4 admissions dropped later
        src = ("cohort_reporting/disease_frequency_final.csv"
               if os.path.isfile("cohort_reporting/disease_frequency_final.csv")
               else f"{PIPE}/disease_frequency.csv")
        print(f"  fig6 frequencies from {src}")
        d = pd.read_csv(src)
        orph = pd.read_csv(ORPH_CSV)
    except FileNotFoundError as ex:
        print(f"  [skip] {ex}"); return
    norm = lambda s: s.astype(str).str.upper().str.replace(".", "", regex=False).str.strip()
    orph["icd_norm"] = norm(orph["ICDcodes"])
    n_terms = orph.groupby("icd_norm")["ORPHAcode"].nunique()
    desc = dict(ICD10CM)
    if D_ICD and os.path.isfile(D_ICD):
        di = pd.read_csv(D_ICD); di = di[di.icd_version == 10]
        desc.update(dict(zip(norm(di.icd_code), di.long_title)))
    code = d.iloc[:, 0].astype(str)
    d = d.assign(icd=code, desc=code.map(desc), n_terms=code.map(n_terms).astype(int))
    d = d.sort_values(["n_patients", "icd"], ascending=[False, True]).head(15)
    lbl = [f"{r.icd}  {r.desc}\n({r.n_terms} Orphanet term{'s' if r.n_terms != 1 else ''})"
           for r in d.itertuples()]
    fig, ax = plt.subplots(figsize=(8.2, 7.4))
    y = np.arange(len(d))[::-1]
    ax.barh(y, d["n_patients"], color="#3a8a6e", height=0.72)
    ax.set_yticks(y); ax.set_yticklabels(lbl, fontsize=7.6)
    for yi, n in zip(y, d["n_patients"]):
        ax.text(n + 8, yi, f"{int(n):,}", va="center", fontsize=7.6)
    ax.set_xlim(0, d["n_patients"].max() * 1.12)
    ax.set_xlabel("Patients"); ax.set_title("Most frequent matched ICD-10 codes in the cohort", fontsize=10)
    ax.grid(axis="y", visible=False)
    save(fig, "fig6_disease_frequency")


def fig9_umap_tsne():
    try:
        p = pd.read_csv(f"{STRAT}/test_projection.csv")
    except FileNotFoundError:
        print("  [skip] test_projection.csv not found"); return
    have = [(a, b, t) for a, b, t in
            [("tsne_x", "tsne_y", "t-SNE"), ("umap_x", "umap_y", "UMAP")]
            if a in p.columns and b in p.columns]
    if not have:
        print("  [skip] no projection columns"); return
    fig, axes = plt.subplots(1, len(have), figsize=(5.2*len(have), 4.6), squeeze=False)
    for ax, (xc, yc, title) in zip(axes[0], have):
        for cl in sorted(p["cluster"].unique()):
            m = p["cluster"] == cl
            ax.scatter(p.loc[m, xc], p.loc[m, yc], s=10, alpha=0.6, label=f"C{cl}")
        ax.set_title(f"{title} (held-out test latents)")
        ax.set_xticks([]); ax.set_yticks([]); ax.legend(fontsize=7, markerscale=1.4)
    fig.tight_layout(); save(fig, "fig9_umap_tsne")


if __name__ == "__main__":
    print("Generating figures →", OUT)
    # Fig 1 (architecture) is a static drawing: figures/fig1_architecture.pdf
    fig2_cohort_flow()
    fig3_modality_ablation()
    fig4_cluster_profile()
    fig6_disease_frequency()
    fig9_umap_tsne()
    print("done.")