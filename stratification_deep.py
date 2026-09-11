"""
stratification_deep.py
=====================================================================
Deepens the PRIMARY (stratification) contribution for a Q1 target.
Runs standalone against pipeline_out/ — no retraining of the prediction
harness required; it re-fits lightweight encoders per modality subset
and evaluates cluster quality, stability, cross-algorithm agreement,
independent-outcome separation, and per-cluster clinical profiles.

Produces (under --out):
  strat_ablation.csv       — cluster quality per modality subset (R1.12 title claim)
  cluster_profile_full.csv — per-cluster clinical fingerprint (R1.13)
  cluster_disease_enrich.csv — top rare-disease codes per cluster
  strat_deep.json          — stability, kmeans/GMM ARI agreement, outcome-separation tests

Run:
  python stratification_deep.py --data pipeline_out --out results_strat --seeds 5
=====================================================================
"""
from __future__ import annotations
import argparse, json, itertools, os
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score, adjusted_rand_score)
from scipy.stats import kruskal, chi2_contingency

# reuse the proposed encoder from the harness
from run_experiments import MultimodalVaDeSC, DEVICE


def load(data):
    Xs = np.load(f"{data}/strat_X.npy"); Ms = np.load(f"{data}/strat_M.npy")
    smeta = pd.read_csv(f"{data}/strat_meta.csv")
    vocab = json.load(open(f"{data}/vocab.json"))
    return Xs, Ms, smeta, vocab


def slices_for(vocab, subset):
    n = dict(dx=len(vocab["dx"]), lab=len(vocab["lab"]),
             med=len(vocab["med"]), proc=len(vocab["proc"]))
    order = ["dx", "lab", "med", "proc"]
    bounds, c = {}, 0
    for k in order:
        bounds[k] = (c, c + n[k]); c += n[k]
    return {k: bounds[k] for k in subset}, c


def train_encoder(model, Xtr, Mtr, epochs=60, lr=1e-3, patience=12):
    """Unsupervised only (recon + KL) — no outcome heads, so clustering is
    NOT circular (R1.10). supervised=False."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ds = TensorDataset(torch.as_tensor(Xtr), torch.as_tensor(Mtr))
    dl = DataLoader(ds, batch_size=256, shuffle=True)
    best, bad, state = np.inf, 0, None
    for ep in range(epochs):
        model.train()
        for x, m in dl:
            opt.zero_grad()
            loss, _ = model.loss(x.float().to(DEVICE), m.float().to(DEVICE))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad():
            vl = model.loss(torch.as_tensor(Xtr).float().to(DEVICE),
                            torch.as_tensor(Mtr).float().to(DEVICE))[0].item()
        if vl < best - 1e-5: best, bad, state = vl, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    if state: model.load_state_dict(state)
    return model


def latents(model, X, M):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            x = torch.as_tensor(X[i:i+256]).float().to(DEVICE)
            m = torch.as_tensor(M[i:i+256]).float().to(DEVICE)
            _, mu, _ = model.encode(x, m); out.append(mu.cpu().numpy())
    return np.concatenate(out)


def cluster_eval(Ztr, Zte, K):
    gmm = GaussianMixture(K, covariance_type="full", reg_covar=1e-4,
                          random_state=0, n_init=5).fit(Ztr)
    lab = gmm.predict(Zte)
    km = KMeans(K, n_init=10, random_state=0).fit(Ztr).predict(Zte)
    nuniq = len(np.unique(lab))
    ok = 2 <= nuniq <= len(Zte) - 1
    return dict(
        silhouette=float(silhouette_score(Zte, lab)) if ok else float("nan"),
        calinski=float(calinski_harabasz_score(Zte, lab)) if ok else float("nan"),
        davies=float(davies_bouldin_score(Zte, lab)) if ok else float("nan"),
        gmm_kmeans_ari=float(adjusted_rand_score(lab, km)),
    ), lab, gmm


def pick_K(Ztr):
    b = {k: GaussianMixture(k, covariance_type="full", reg_covar=1e-4,
                            random_state=0, n_init=3).fit(Ztr).bic(Ztr) for k in range(2, 9)}
    return min(b, key=b.get), b


def outcome_separation(prof):
    """Independent-outcome tests across clusters (R1.10). Never-trained outcomes."""
    res = {}
    # continuous: total_los via Kruskal-Wallis
    groups = [g["total_los_days"].values for _, g in prof.groupby("cluster")]
    res["los_kruskal_p"] = float(kruskal(*groups).pvalue)
    # binary: ICU and 365d mortality via chi-square
    for col in ["icu_any", "death_within_365d"]:
        ct = pd.crosstab(prof["cluster"], prof[col])
        res[f"{col}_chi2_p"] = float(chi2_contingency(ct).pvalue)
    return res


def main(data, out, seeds, fixed_k=None):
    os.makedirs(out, exist_ok=True)
    Xs, Ms, smeta, vocab = load(data)
    tr = smeta.partition.values == "train"
    te = smeta.partition.values == "test"
    Xtr, Mtr, Xte, Mte = Xs[tr], Ms[tr], Xs[te], Ms[te]

    ablation_rows, deep = [], {}
    if fixed_k:
        print(f"[config] K pinned to {fixed_k} for all subsets")

    # ---------- 1. Modality ablation on CLUSTERING (R1.12; title claim) ----------
    subsets = {
        "dx": ["dx"],
        "dx+lab": ["dx", "lab"],
        "dx+lab+med": ["dx", "lab", "med"],
        "full": ["dx", "lab", "med", "proc"],
    }
    ref_labels = {}
    for name, sub in subsets.items():
        sl, _ = slices_for(vocab, sub)
        per_seed = []
        seed_labels = []
        for s in range(seeds):
            torch.manual_seed(s); np.random.seed(s)
            m = MultimodalVaDeSC(sl, supervised=False).to(DEVICE)
            m = train_encoder(m, Xtr, Mtr)
            Ztr, Zte = latents(m, Xtr, Mtr), latents(m, Xte, Mte)
            K = fixed_k if fixed_k else pick_K(Ztr)[0]
            metrics, lab, _ = cluster_eval(Ztr, Zte, K)
            metrics.update(subset=name, seed=s, K=K)
            per_seed.append(metrics); seed_labels.append(lab)
        # stability across seeds for this subset
        ari = np.mean([adjusted_rand_score(a, b)
                       for a, b in itertools.combinations(seed_labels, 2)])
        dfm = pd.DataFrame(per_seed)
        row = dict(subset=name,
                   K_mode=int(dfm.K.mode().iloc[0]),
                   silhouette_mean=float(dfm.silhouette.mean()),
                   silhouette_std=float(dfm.silhouette.std()),
                   calinski_mean=float(dfm.calinski.mean()),
                   davies_mean=float(dfm.davies.mean()),
                   gmm_kmeans_ari=float(dfm.gmm_kmeans_ari.mean()),
                   stability_ari=float(ari))
        ablation_rows.append(row)
        ref_labels[name] = seed_labels[0]
        print(f"[ablation] {name:<12s} K={row['K_mode']} sil={row['silhouette_mean']:.3f} "
              f"CH={row['calinski_mean']:.1f} stab_ARI={row['stability_ari']:.3f} "
              f"gmm/kmeans_ARI={row['gmm_kmeans_ari']:.3f}")
    pd.DataFrame(ablation_rows).to_csv(f"{out}/strat_ablation.csv", index=False)

    # does full beat dx-only on independent-outcome separation? (the honest
    # "multimodal helps" test — on the PRIMARY objective, not prediction)
    prof_te = smeta.iloc[np.where(te)[0]].copy()
    for name in ["dx", "full"]:
        prof_te["cluster"] = ref_labels[name]
        deep[f"outcome_separation_{name}"] = outcome_separation(prof_te)

    # ---------- 2. Full-model per-cluster clinical fingerprint (R1.13) ----------
    sl, _ = slices_for(vocab, ["dx", "lab", "med", "proc"])
    torch.manual_seed(0); np.random.seed(0)
    mm = MultimodalVaDeSC(sl, supervised=False).to(DEVICE)
    mm = train_encoder(mm, Xtr, Mtr)
    Ztr, Zte = latents(mm, Xtr, Mtr), latents(mm, Xte, Mte)
    K, bic = (fixed_k, {}) if fixed_k else pick_K(Ztr)
    metrics, lab_te, _ = cluster_eval(Ztr, Zte, K)
    deep.update(K=K, bic=bic, **metrics)

    prof = smeta.iloc[np.where(te)[0]].copy(); prof["cluster"] = lab_te
    profile = (prof.groupby("cluster")
               .agg(n=("subject_id", "size"),
                    icu_any=("icu_any", "mean"),
                    total_los_days=("total_los_days", "mean"),
                    death_365d=("death_within_365d", "mean"),
                    n_admissions=("n_admissions", "mean"))
               .round(3).reset_index())
    profile.to_csv(f"{out}/cluster_profile_full.csv", index=False)
    deep["outcome_separation_full_final"] = outcome_separation(prof)

    # per-cluster mean of the standardized feature blocks → top signals
    # (uses TEST admission-level features averaged per patient via the tensor)
    feat_names = ([f"dx_{c}" for c in vocab["dx"]] +
                  [f"lab_{c}" for c in vocab["lab"]] +
                  [f"med_{c}" for c in vocab["med"]] +
                  [f"proc_{c}" for c in vocab["proc"]])
    Xte_mean = (Xte * Mte[..., None]).sum(1) / (Mte.sum(1, keepdims=True) + 1e-6)
    fp = pd.DataFrame(Xte_mean, columns=feat_names)
    fp["cluster"] = lab_te
    enrich_rows = []
    for cl, g in fp.groupby("cluster"):
        means = g[feat_names].mean()
        top = means.reindex(means.abs().sort_values(ascending=False).index).head(10)
        for rank, (feat, val) in enumerate(top.items(), 1):
            enrich_rows.append(dict(cluster=cl, rank=rank, feature=feat,
                                    mean_z=round(float(val), 3)))
    pd.DataFrame(enrich_rows).to_csv(f"{out}/cluster_disease_enrich.csv", index=False)

    with open(f"{out}/strat_deep.json", "w") as f:
        json.dump(deep, f, indent=2, default=str)
    print("\n[done] wrote strat_ablation.csv, cluster_profile_full.csv, "
          "cluster_disease_enrich.csv, strat_deep.json")
    print(json.dumps({k: deep[k] for k in
                      ["K", "silhouette", "gmm_kmeans_ari",
                       "outcome_separation_full_final"] if k in deep},
                     indent=2, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="pipeline_out")
    ap.add_argument("--out", default="results_strat")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--fixed_k", type=int, default=None,
                    help="pin K for all subsets (e.g. 3); omit for BIC selection")
    a = ap.parse_args()
    main(a.data, a.out, a.seeds, a.fixed_k)
