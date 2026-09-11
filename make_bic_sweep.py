r"""
make_bic_sweep.py
=====================================================================
Generates Supplement S2 (reviewer item R1.11): the cluster-number
sweep the main text promises ("the full BIC sweep is reported in the
supplement" and "markedly higher stability at K=3").

It reuses the EXACT encoder, training loop, latent extraction, and GMM
settings from stratification_deep.py, so the numbers are consistent
with the locked ablation (full-model stability 0.671, etc.). Run it in
the SAME Kaggle environment (Python 3.11, T4, PyTorch) that produced
every other result; do not run it on other hardware, as retraining the
VAE encoder elsewhere yields different latents and therefore different
BIC/stability values.

For the FULL multimodal model it computes, for K in {2..8}:
  - BIC        : GMM on seed-0 training latents (identical to pick_K()).
  - stability  : mean pairwise adjusted Rand index of TEST assignments
                 across five seeds (identical to the ablation's stability).
  - silhouette / calinski / davies / gmm-kmeans ARI on seed-0 (context).

Outputs (under --out):
  bic_sweep.csv        - one row per K, all columns.
  s2_bic_table.tex     - paste-ready LaTeX table (\input into the supplement).
  bic_sweep.json       - same numbers + a deterministic fingerprint hash.

Run (from the folder holding stratification_deep.py, run_experiments.py,
and pipeline_out/):
  python make_bic_sweep.py --data pipeline_out --out results_bic --seeds 5
=====================================================================
"""
from __future__ import annotations
import argparse, json, hashlib, itertools, os
import numpy as np, pandas as pd
import torch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score, adjusted_rand_score)

# Reuse the canonical pieces so this stays faithful to the paper's pipeline.
from stratification_deep import (load, slices_for, train_encoder, latents)
from run_experiments import MultimodalVaDeSC, DEVICE

KRANGE = range(2, 9)          # K = 2..8, matching pick_K()
FULL_SUBSET = ["dx", "lab", "med", "proc"]


def bic_for(Ztr, K):
    """Identical to pick_K(): GMM BIC on training latents."""
    g = GaussianMixture(K, covariance_type="full", reg_covar=1e-4,
                        random_state=0, n_init=3).fit(Ztr)
    return float(g.bic(Ztr))


def geom_for(Ztr, Zte, K):
    """Identical to cluster_eval(): seed-0 geometric indices + gmm/kmeans ARI."""
    from sklearn.cluster import KMeans
    gmm = GaussianMixture(K, covariance_type="full", reg_covar=1e-4,
                          random_state=0, n_init=5).fit(Ztr)
    lab = gmm.predict(Zte)
    km = KMeans(K, n_init=10, random_state=0).fit(Ztr).predict(Zte)
    ok = 2 <= len(np.unique(lab)) <= len(Zte) - 1
    return dict(
        silhouette=float(silhouette_score(Zte, lab)) if ok else float("nan"),
        calinski=float(calinski_harabasz_score(Zte, lab)) if ok else float("nan"),
        davies=float(davies_bouldin_score(Zte, lab)) if ok else float("nan"),
        gmm_kmeans_ari=float(adjusted_rand_score(lab, km)),
    )


def test_labels(Ztr, Zte, K):
    """Test-set GMM assignment (same estimator as cluster_eval)."""
    gmm = GaussianMixture(K, covariance_type="full", reg_covar=1e-4,
                          random_state=0, n_init=5).fit(Ztr)
    return gmm.predict(Zte)


def main(data, out, seeds):
    os.makedirs(out, exist_ok=True)
    Xs, Ms, smeta, vocab = load(data)
    tr = smeta.partition.values == "train"
    te = smeta.partition.values == "test"
    Xtr, Mtr, Xte, Mte = Xs[tr], Ms[tr], Xs[te], Ms[te]
    sl, _ = slices_for(vocab, FULL_SUBSET)

    # Train the full-model encoder once per seed; cache latents. Latents are
    # K-independent, so we retrain only `seeds` times, not seeds x |KRANGE|.
    Ztr_by_seed, Zte_by_seed = [], []
    for s in range(seeds):
        torch.manual_seed(s); np.random.seed(s)
        m = MultimodalVaDeSC(sl, supervised=False).to(DEVICE)
        m = train_encoder(m, Xtr, Mtr)
        Ztr_by_seed.append(latents(m, Xtr, Mtr))
        Zte_by_seed.append(latents(m, Xte, Mte))
        print(f"[encoder] seed {s} trained")

    Ztr0, Zte0 = Ztr_by_seed[0], Zte_by_seed[0]   # seed-0 canonical (matches pick_K)

    rows = []
    for K in KRANGE:
        # stability: pairwise ARI of test assignments across seeds
        labs = [test_labels(Ztr_by_seed[s], Zte_by_seed[s], K) for s in range(seeds)]
        stab = float(np.mean([adjusted_rand_score(a, b)
                              for a, b in itertools.combinations(labs, 2)]))
        geom = geom_for(Ztr0, Zte0, K)
        rows.append(dict(K=K, bic=bic_for(Ztr0, K), stability_ari=stab, **geom))
        print(f"[K={K}] BIC={rows[-1]['bic']:.1f} stab_ARI={stab:.3f} "
              f"sil={geom['silhouette']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{out}/bic_sweep.csv", index=False)

    k_bic = int(df.loc[df.bic.idxmin(), "K"])
    k_stab = int(df.loc[df.stability_ari.idxmax(), "K"])
    fp = hashlib.sha256(
        np.round(df[["bic", "stability_ari", "silhouette",
                     "calinski", "davies", "gmm_kmeans_ari"]].values, 6)
        .tobytes()).hexdigest()[:16]

    with open(f"{out}/bic_sweep.json", "w") as f:
        json.dump(dict(rows=rows, argmin_bic_K=k_bic, argmax_stability_K=k_stab,
                       fingerprint=fp, seeds=seeds), f, indent=2)

    # paste-ready LaTeX fragment (bold the K=3 row)
    def fmt(r):
        cells = [f"{int(r.K)}", f"{r.bic:.0f}", f"{r.stability_ari:.3f}",
                 f"{r.silhouette:.3f}", f"{r.calinski:.1f}",
                 f"{r.davies:.3f}", f"{r.gmm_kmeans_ari:.3f}"]
        if int(r.K) == 3:
            cells = [f"\\textbf{{{c}}}" for c in cells]
        return " & ".join(cells) + r" \\"
    body = "\n".join(fmt(r) for _, r in df.iterrows())
    tex = (
        "% Auto-generated by make_bic_sweep.py -- do not edit by hand.\n"
        "\\begin{table}[!h]\n\\centering\n"
        "\\caption{BIC sweep and stability for the full multimodal model "
        "($K\\in\\{2,\\dots,8\\}$). BIC on seed-0 training latents; stability "
        "is the mean pairwise adjusted Rand index of test-set assignments "
        "across five seeds. Lower BIC and Davies--Bouldin are better; higher "
        "stability, Silhouette, and Calinski--Harabasz are better.}\n"
        "\\label{tab:bic-sweep}\n"
        "\\begin{tabular}{ccccccc}\n\\toprule\n"
        "$K$ & BIC & Stability (ARI) & Silhouette & Calinski--Harabasz & "
        "Davies--Bouldin & GMM/$k$-means ARI \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
    )
    with open(f"{out}/s2_bic_table.tex", "w") as f:
        f.write(tex)

    print(f"\n[done] argmin BIC at K={k_bic}; max stability at K={k_stab}.")
    print(f"[fingerprint] {fp}")
    print(f"[wrote] {out}/bic_sweep.csv, {out}/s2_bic_table.tex, {out}/bic_sweep.json")
    if k_stab != 3:
        print("[NOTE] stability peak is not at K=3 in this run; check the "
              "main-text wording before submitting.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="pipeline_out")
    ap.add_argument("--out", default="results_bic")
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    main(a.data, a.out, a.seeds)
