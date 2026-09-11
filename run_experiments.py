"""
run_experiments.py
=====================================================================
Training + evaluation harness for the revision. All models consume the
SAME tensors from rare_ehr_pipeline.py (R1.6: same patients, same split,
same T_MAX, same leakage-free predictors, same outcome definitions).

Models
  1. MultimodalVaDeSC   — proposed. VAE + Transformer + attention pooling.
                          NO survival head (R1.3: survival language removed).
                          Auxiliary ICU/LOS heads train ONLY on the
                          prospective (next-admission) targets.
  2. UnimodalVaDeSC     — dx-block-only ablation of the same architecture.
  3. MambaFixed         — TinyMamba masked-reconstruction baseline on the
                          SAME multimodal features (R1.2: outcome triple
                          [LOS, ICU, expired] removed from inputs).
  4. LogisticRegression / Ridge — simple baselines (R1.6).
  5. GRU                — recurrent baseline (R1.6).

Evaluation (R1.7 / R1.8 / R1.16)
  ICU:   AUROC, AUPRC, Brier, calibration slope/intercept, sens/spec/F1
         at Youden threshold — on the untouched TEST partition.
  LOS:   MAE, RMSE, R^2 on the ORIGINAL day scale.
  Clust: Silhouette / Calinski-Harabasz / Davies-Bouldin on TEST latents;
         stability = mean pairwise ARI across N_SEEDS retrainings.
  CIs:   patient-level bootstrap (B=1000).
  Tests: paired bootstrap deltas + Wilcoxon signed-rank across seeds.
  Cluster validity (R1.9 / R1.10): clusters fit on TRAIN latents, assigned
         to TEST patients, profiled on outcomes NEVER used in training —
         90-day readmission, 365-day post-discharge mortality, next-
         admission ICU/LOS.

Run:  python run_experiments.py --data ./pipeline_out --out ./results
=====================================================================
"""
from __future__ import annotations
import argparse, json, os, itertools
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.mixture import GaussianMixture
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, mean_absolute_error,
                             mean_squared_error, r2_score, silhouette_score,
                             calinski_harabasz_score, davies_bouldin_score,
                             adjusted_rand_score, f1_score)
from sklearn.linear_model import LogisticRegression, Ridge
from scipy.stats import wilcoxon

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------ #
# Models                                                             #
# ------------------------------------------------------------------ #
class AttentionPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d))

    def forward(self, H, m):
        s = (H * self.q).sum(-1).masked_fill(m == 0, -1e9)
        w = torch.softmax(s, -1).unsqueeze(-1)
        return (w * H).sum(1)


class MultimodalVaDeSC(nn.Module):
    """VAE + Transformer encoder + attention pooling. slices maps modality
    name -> (start, end) column range; pass a dx-only slice for the
    unimodal ablation. No survival head (R1.3)."""
    PROJ_FRAC = dict(dx=0.5, lab=0.25, med=0.25, proc=0.25)

    def __init__(self, slices, d_model=128, d_lat=16, nhead=4, nlayers=2,
                 dropout=0.1, supervised=True):
        super().__init__()
        self.slices, self.supervised = slices, supervised
        self.projs = nn.ModuleDict()
        fused = 0
        for k, (a, b) in slices.items():
            dim = max(int(d_model * self.PROJ_FRAC.get(k, 0.25)), 16)
            self.projs[k] = nn.Sequential(nn.Linear(b - a, dim), nn.ReLU(),
                                          nn.Dropout(dropout))
            fused += dim
        enc = nn.TransformerEncoderLayer(fused, nhead, fused * 2, dropout,
                                         batch_first=True)
        self.tr = nn.TransformerEncoder(enc, nlayers)
        self.pool = AttentionPool(fused)
        self.mu, self.logvar = nn.Linear(fused, d_lat), nn.Linear(fused, d_lat)
        # reconstruct ONLY the columns this model consumes (so the dx-only
        # ablation is well-posed): register the used-column index buffer
        used = torch.cat([torch.arange(a, b) for a, b in slices.values()])
        self.register_buffer("used_cols", used)
        self.dec = nn.Sequential(nn.Linear(d_lat, fused), nn.ReLU(),
                                 nn.Linear(fused, len(used)))
        self.h_icu, self.h_los = nn.Linear(d_lat, 1), nn.Linear(d_lat, 1)

    def encode(self, x, m):
        H = torch.cat([self.projs[k](x[..., a:b])
                       for k, (a, b) in self.slices.items()], -1)
        H = self.tr(H, src_key_padding_mask=(m == 0))
        c = self.pool(H, m)
        mu, lv = self.mu(c), self.logvar(c)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        return z, mu, lv

    def forward(self, x, m):
        z, mu, lv = self.encode(x, m)
        return self.dec(z), mu, lv, z, \
               self.h_icu(z).squeeze(-1), self.h_los(z).squeeze(-1)

    def loss(self, x, m, y_icu=None, y_los=None, beta=1e-3):
        rec, mu, lv, z, li, ll = self(x, m)
        xu = x[..., self.used_cols]
        xm = (xu * m.unsqueeze(-1)).sum(1) / (m.sum(1, keepdim=True) + 1e-6)
        L = F.mse_loss(rec, xm)
        L = L + beta * (-0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp()))
        parts = {"rec": L.item()}
        if self.supervised and y_icu is not None:
            bce = F.binary_cross_entropy_with_logits(li, y_icu)
            mse = F.mse_loss(ll, y_los)
            L = L + bce + mse
            parts.update(bce=bce.item(), mse=mse.item())
        return L, parts


class TinyMambaBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.f1, self.f2 = nn.Linear(d, d), nn.Linear(d, d)
        self.norm, self.drop = nn.LayerNorm(d), nn.Dropout(0.1)

    def forward(self, x):
        return self.norm(x + self.drop(torch.sigmoid(self.f1(x)) * self.f2(x)))


class MambaFixed(nn.Module):
    """Masked-reconstruction self-supervised baseline on the SAME
    multimodal features (outcome triple removed — R1.2)."""
    def __init__(self, d_in, d_model=256, n_layers=4, t_max=8):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.pos = nn.Embedding(t_max, d_model)
        self.blocks = nn.ModuleList(TinyMambaBlock(d_model)
                                    for _ in range(n_layers))
        self.norm, self.dec = nn.LayerNorm(d_model), nn.Linear(d_model, d_in)

    def forward(self, x):
        B, T, _ = x.shape
        h = self.proj(x) + self.pos(torch.arange(T, device=x.device))[None]
        for b in self.blocks:
            h = b(h)
        h = self.norm(h)
        return self.dec(h), h.mean(1)


class GRUBaseline(nn.Module):
    def __init__(self, d_in, d_h=128):
        super().__init__()
        self.gru = nn.GRU(d_in, d_h, batch_first=True)
        self.h_icu, self.h_los = nn.Linear(d_h, 1), nn.Linear(d_h, 1)

    def forward(self, x, m):
        out, _ = self.gru(x)
        idx = (m.sum(1).long() - 1).clamp(min=0)
        h = out[torch.arange(len(x)), idx]
        return self.h_icu(h).squeeze(-1), self.h_los(h).squeeze(-1), h


# ------------------------------------------------------------------ #
# Metrics                                                            #
# ------------------------------------------------------------------ #
def calibration(y, p):
    from sklearn.linear_model import LogisticRegression as LR
    eps = 1e-6
    logit = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps))
    lr = LR(penalty=None).fit(logit.reshape(-1, 1), y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def icu_metrics(y, p):
    fpr_grid = np.linspace(0, 1, 101)
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, p)
    youden = thr[np.argmax(tpr - fpr)]
    yhat = (p >= youden).astype(int)
    slope, intercept = calibration(y, p)
    tp = ((yhat == 1) & (y == 1)).sum(); tn = ((yhat == 0) & (y == 0)).sum()
    fp = ((yhat == 1) & (y == 0)).sum(); fn = ((yhat == 0) & (y == 1)).sum()
    return dict(auroc=roc_auc_score(y, p),
                auprc=average_precision_score(y, p),
                brier=brier_score_loss(y, p),
                cal_slope=slope, cal_intercept=intercept,
                sensitivity=tp / max(tp + fn, 1),
                specificity=tn / max(tn + fp, 1),
                ppv=tp / max(tp + fp, 1), npv=tn / max(tn + fn, 1),
                f1=f1_score(y, yhat),
                bal_acc=0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)))


def los_metrics(y_log, p_log):
    y, p = np.expm1(y_log), np.expm1(p_log)          # original day scale (R1.7)
    return dict(mae_days=mean_absolute_error(y, p),
                rmse_days=float(np.sqrt(mean_squared_error(y, p))),
                r2=r2_score(y, p))


def bootstrap_ci(y, p, fn, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        try:
            vals.append(fn(y[i], p[i]))
        except ValueError:
            continue
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


# ------------------------------------------------------------------ #
# Training loops                                                     #
# ------------------------------------------------------------------ #
def train_torch(model, tr_loader, va_loader, step_fn, epochs=100, lr=1e-3,
                patience=15, tag=""):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, bad, best_state = np.inf, 0, None
    for ep in range(epochs):
        model.train()
        for batch in tr_loader:
            opt.zero_grad()
            loss = step_fn(model, [b.to(DEVICE) for b in batch])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = np.mean([step_fn(model, [b.to(DEVICE) for b in batch]).item()
                             for batch in va_loader])
        if vloss < best - 1e-5:
            best, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    print(f"[train:{tag}] best val loss {best:.4f} @ early stop epoch {ep - bad}")
    return model


def loaders(*tensors, part, meta, batch=256, shuffle=False):
    idx = np.where(meta.partition.values == part)[0]
    ds = TensorDataset(*[torch.as_tensor(t[idx]) for t in tensors])
    return DataLoader(ds, batch_size=batch, shuffle=shuffle), idx


# ------------------------------------------------------------------ #
# Main                                                               #
# ------------------------------------------------------------------ #
def main(data_dir, out_dir, n_seeds=5, epochs=100):
    os.makedirs(out_dir, exist_ok=True)
    Xs = np.load(f"{data_dir}/strat_X.npy"); Ms = np.load(f"{data_dir}/strat_M.npy")
    smeta = pd.read_csv(f"{data_dir}/strat_meta.csv")
    Xp = np.load(f"{data_dir}/prosp_X.npy"); Mp = np.load(f"{data_dir}/prosp_M.npy")
    yI = np.load(f"{data_dir}/prosp_y_icu.npy"); yL = np.load(f"{data_dir}/prosp_y_los.npy")
    pmeta = pd.read_csv(f"{data_dir}/prosp_meta.csv")
    vocab = json.load(open(f"{data_dir}/vocab.json"))

    # modality slices (must match pipeline column order)
    n_dx, n_lab = len(vocab["dx"]), len(vocab["lab"])
    n_med, n_proc = len(vocab["med"]), len(vocab["proc"])
    c = np.cumsum([0, n_dx, n_lab, n_med, n_proc])
    slices_mm = dict(dx=(c[0], c[1]), lab=(c[1], c[2]),
                     med=(c[2], c[3]), proc=(c[3], c[4]))
    slices_uni = dict(dx=(c[0], c[1]))
    D = int(c[-1])

    results, seed_records = {}, []

    for seed in range(n_seeds):
        torch.manual_seed(seed); np.random.seed(seed)

        # ---------- prospective task loaders ----------
        tr_l, tr_i = loaders(Xp, Mp, yI, yL, part="train", meta=pmeta, shuffle=True)
        va_l, va_i = loaders(Xp, Mp, yI, yL, part="val", meta=pmeta)
        te_l, te_i = loaders(Xp, Mp, yI, yL, part="test", meta=pmeta)

        def vae_step(mdl, b):
            x, m, yi, yl = b
            return mdl.loss(x.float(), m.float(), yi.float(), yl.float())[0]

        def eval_heads(mdl, loader):
            mdl.eval(); P_i, P_l, Y_i, Y_l, Z = [], [], [], [], []
            with torch.no_grad():
                for x, m, yi, yl in loader:
                    _, _, _, z, li, ll = mdl(x.float().to(DEVICE), m.float().to(DEVICE))
                    P_i.append(torch.sigmoid(li).cpu().numpy()); P_l.append(ll.cpu().numpy())
                    Y_i.append(yi.numpy()); Y_l.append(yl.numpy()); Z.append(z.cpu().numpy())
            return map(np.concatenate, (P_i, P_l, Y_i, Y_l, Z))

        # ===== 1. Multimodal (proposed) =====
        mm = MultimodalVaDeSC(slices_mm).to(DEVICE)
        mm = train_torch(mm, tr_l, va_l, vae_step, epochs, tag=f"mm_s{seed}")
        pi, pl, yi_t, yl_t, _ = eval_heads(mm, te_l)
        rec = dict(model="Multimodal-VaDeSC", seed=seed,
                   **icu_metrics(yi_t, pi), **los_metrics(yl_t, pl))
        seed_records.append(rec)

        # ===== 2. Unimodal ablation =====
        uni = MultimodalVaDeSC(slices_uni).to(DEVICE)
        uni = train_torch(uni, tr_l, va_l, vae_step, epochs, tag=f"uni_s{seed}")
        pi, pl, yi_t, yl_t, _ = eval_heads(uni, te_l)
        seed_records.append(dict(model="Unimodal-VaDeSC(dx)", seed=seed,
                                 **icu_metrics(yi_t, pi), **los_metrics(yl_t, pl)))

        # ===== 3. GRU =====
        gru = GRUBaseline(D).to(DEVICE)

        def gru_step(mdl, b):
            x, m, yi, yl = b
            li, ll, _ = mdl(x.float(), m.float())
            return F.binary_cross_entropy_with_logits(li, yi.float()) + \
                   F.mse_loss(ll, yl.float())
        gru = train_torch(gru, tr_l, va_l, gru_step, epochs, tag=f"gru_s{seed}")
        gru.eval(); Pi, Pl = [], []
        with torch.no_grad():
            for x, m, yi_, yl_ in te_l:
                li, ll, _ = gru(x.float().to(DEVICE), m.float().to(DEVICE))
                Pi.append(torch.sigmoid(li).cpu().numpy()); Pl.append(ll.cpu().numpy())
        seed_records.append(dict(model="GRU", seed=seed,
                                 **icu_metrics(yi_t, np.concatenate(Pi)),
                                 **los_metrics(yl_t, np.concatenate(Pl))))

        # ===== 4. MambaFixed (self-supervised; heads = LR/Ridge probe) =====
        mam = MambaFixed(D).to(DEVICE)

        def mam_step(mdl, b):
            x, m = b[0].float(), b[1].float()
            keep = (torch.rand_like(m) > 0.15).float() * m
            rec_, _ = mdl(x * keep.unsqueeze(-1))
            msk = (m.unsqueeze(-1) > 0)
            return (F.mse_loss(rec_, x, reduction="none") * msk).sum() / msk.sum()
        mam = train_torch(mam, tr_l, va_l, mam_step, epochs, tag=f"mamba_s{seed}")

        def embed(loader):
            mam.eval(); E = []
            with torch.no_grad():
                for x, m, *_ in loader:
                    _, e = mam(x.float().to(DEVICE)); E.append(e.cpu().numpy())
            return np.concatenate(E)
        Etr, Ete = embed(tr_l), embed(te_l)
        yi_tr, yl_tr = yI[tr_i], yL[tr_i]
        pi = LogisticRegression(max_iter=2000).fit(Etr, yi_tr).predict_proba(Ete)[:, 1]
        pl = Ridge().fit(Etr, yl_tr).predict(Ete)
        seed_records.append(dict(model="EHR-Mamba(fixed)", seed=seed,
                                 **icu_metrics(yi_t, pi), **los_metrics(yl_t, pl)))

        # ===== 5. Flat LR / Ridge on mean-pooled features =====
        if seed == 0:
            pool = lambda X_, M_: (X_ * M_[..., None]).sum(1) / \
                                  (M_.sum(1, keepdims=True) + 1e-6)
            Ftr, Fte = pool(Xp[tr_i], Mp[tr_i]), pool(Xp[te_i], Mp[te_i])
            pi = LogisticRegression(max_iter=2000, C=0.1).fit(Ftr, yi_tr)\
                     .predict_proba(Fte)[:, 1]
            pl = Ridge(alpha=1.0).fit(Ftr, yl_tr).predict(Fte)
            seed_records.append(dict(model="LogReg/Ridge", seed=seed,
                                     **icu_metrics(yi_t, pi), **los_metrics(yl_t, pl)))

        # ===== Stratification: latents, clustering, validity (seed 0) =====
        if seed == 0:
            s_tr, s_tr_i = loaders(Xs, Ms, part="train", meta=smeta)
            s_te, s_te_i = loaders(Xs, Ms, part="test", meta=smeta)

            def latents(mdl, loader):
                mdl.eval(); Z = []
                with torch.no_grad():
                    for x, m in loader:
                        _, mu, _ = mdl.encode(x.float().to(DEVICE),
                                              m.float().to(DEVICE))
                        Z.append(mu.cpu().numpy())
                return np.concatenate(Z)
            Ztr, Zte = latents(mm, s_tr), latents(mm, s_te)

            # K by BIC on TRAIN latents only; evaluate on TEST (R1.5/R1.14)
            bics = {k: GaussianMixture(k, covariance_type="full", reg_covar=1e-4,
                                       random_state=0, n_init=3).fit(Ztr).bic(Ztr)
                    for k in range(2, 9)}
            K = min(bics, key=bics.get)
            gmm = GaussianMixture(K, covariance_type="full", reg_covar=1e-4,
                                  random_state=0, n_init=5).fit(Ztr)
            lab_te = gmm.predict(Zte)
            clus = dict(K=K, bic=bics,
                        silhouette=float(silhouette_score(Zte, lab_te)),
                        calinski=float(calinski_harabasz_score(Zte, lab_te)),
                        davies=float(davies_bouldin_score(Zte, lab_te)))

            # stability across seeds: refit GMM with 5 inits (R1.9/R1.14)
            labs = [GaussianMixture(K, covariance_type="full", reg_covar=1e-4,
                                    random_state=s, n_init=3).fit(Ztr).predict(Zte)
                    for s in range(5)]
            clus["stability_ari"] = float(np.mean(
                [adjusted_rand_score(a, b) for a, b in itertools.combinations(labs, 2)]))

            # independent validity profile (R1.10): outcomes never in a loss
            prof = smeta.iloc[s_te_i].copy()
            prof["cluster"] = lab_te
            clus["test_cluster_profile"] = (
                prof.groupby("cluster")
                    .agg(n=("subject_id", "size"),
                         icu_any=("icu_any", "mean"),
                         total_los=("total_los_days", "mean"),
                         death_365d=("death_within_365d", "mean"))
                    .round(3).to_dict())
            results["clustering"] = clus

    # ---------- aggregate across seeds + stats (R1.8) ----------
    df = pd.DataFrame(seed_records)
    df.to_csv(f"{out_dir}/per_seed_metrics.csv", index=False)
    agg = df.groupby("model").agg(["mean", "std"]).round(4)
    agg.to_csv(f"{out_dir}/aggregate_metrics.csv")
    stats = {}
    piv = df.pivot_table(index="seed", columns="model", values="auroc")
    for a, b in itertools.combinations([c_ for c_ in piv.columns
                                        if piv[c_].notna().all()], 2):
        try:
            w = wilcoxon(piv[a], piv[b])
            stats[f"{a} vs {b}"] = dict(stat=float(w.statistic), p=float(w.pvalue))
        except ValueError:
            pass
    results["wilcoxon_auroc"] = stats
    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(agg)
    print(json.dumps(results.get("clustering", {}), indent=2, default=str)[:2000])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./pipeline_out")
    ap.add_argument("--out", default="./results")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=100)
    a = ap.parse_args()
    main(a.data, a.out, a.seeds, a.epochs)
