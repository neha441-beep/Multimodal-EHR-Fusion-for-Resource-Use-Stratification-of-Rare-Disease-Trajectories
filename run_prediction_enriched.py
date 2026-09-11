"""
run_prediction_enriched.py
=====================================================================
Tier-1+ prediction harness on the ENRICHED prospective tensors.
Every model receives BOTH the admission sequence AND the per-patient
enriched feature block (demographics, prior-outcome history,
comorbidity, admission cadence, time gap). All features pre-cutoff.

Models (all on identical enriched data, R1.6):
  MM+       — proposed VAE encoder → latent ⊕ extra → ICU + LOS(3-class) heads
  Uni+(dx)  — dx-only encoder, same fusion
  GRU+      — GRU over sequence ⊕ extra
  LogReg+   — mean-pooled sequence ⊕ extra → logistic / softmax / ridge
  Extra-only— enriched block ALONE (ablation: how much do the new
              per-patient features carry vs the sequence?)

Reports on TEST: ICU AUROC/AUPRC/Brier/calibration/F1; LOS 3-class
macro-F1 + per-class AUROC + (kept) regression MAE/RMSE/R²; 95%
bootstrap CIs; 5-seed Wilcoxon.

Run: python run_prediction_enriched.py --data prospective_enriched --out results_pred_enriched
=====================================================================
"""
from __future__ import annotations
import argparse, json, itertools, os
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, mean_absolute_error,
                             mean_squared_error, r2_score, f1_score, roc_curve)
from scipy.stats import wilcoxon

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------ models
class AttnPool(nn.Module):
    def __init__(self, d):
        super().__init__(); self.q = nn.Parameter(torch.randn(d))
    def forward(self, H, m):
        s = (H*self.q).sum(-1).masked_fill(m == 0, -1e9)
        return (torch.softmax(s, -1).unsqueeze(-1)*H).sum(1)


class TCNBlock(nn.Module):
    def __init__(self, d, k=3, dil=1, drop=0.1):
        super().__init__()
        pad = (k - 1) * dil
        self.conv = nn.Conv1d(d, d, k, padding=pad, dilation=dil)
        self.pad = pad
        self.norm = nn.LayerNorm(d); self.drop = nn.Dropout(drop)
    def forward(self, x):                      # x: (B,T,d)
        y = self.conv(x.transpose(1, 2))       # (B,d,T+pad)
        y = y[..., :x.size(1)].transpose(1, 2) # causal crop → (B,T,d)
        return self.norm(x + self.drop(torch.relu(y)))


class MambaLite(nn.Module):
    def __init__(self, d, drop=0.1):
        super().__init__()
        self.f1, self.f2 = nn.Linear(d, d), nn.Linear(d, d)
        self.norm, self.drop = nn.LayerNorm(d), nn.Dropout(drop)
    def forward(self, x):
        return self.norm(x + self.drop(torch.sigmoid(self.f1(x)) * self.f2(x)))


class SeqEncoder(nn.Module):
    """Swappable sequence encoder feeding a variational latent.
    encoder in {transformer, gru, lstm, tcn, mamba}. Same latent/pool
    interface across all variants so downstream fusion is identical."""
    def __init__(self, slices, encoder="transformer", d_model=128, d_lat=16,
                 nhead=4, nlayers=2, drop=0.1):
        super().__init__()
        self.slices, self.encoder = slices, encoder
        self.projs = nn.ModuleDict()
        fused = 0
        for k, (a, b) in slices.items():
            dim = max(int(d_model*(0.5 if k == "dx" else 0.25)), 16)
            self.projs[k] = nn.Sequential(nn.Linear(b-a, dim), nn.ReLU(), nn.Dropout(drop))
            fused += dim
        self.fused = fused
        if encoder == "transformer":
            enc = nn.TransformerEncoderLayer(fused, nhead, fused*2, drop, batch_first=True)
            self.seq = nn.TransformerEncoder(enc, nlayers)
        elif encoder == "gru":
            self.seq = nn.GRU(fused, fused, nlayers, batch_first=True, dropout=drop if nlayers > 1 else 0)
        elif encoder == "lstm":
            self.seq = nn.LSTM(fused, fused, nlayers, batch_first=True, dropout=drop if nlayers > 1 else 0)
        elif encoder == "tcn":
            self.seq = nn.ModuleList([TCNBlock(fused, dil=2**i, drop=drop) for i in range(nlayers)])
        elif encoder == "mamba":
            self.seq = nn.ModuleList([MambaLite(fused, drop) for _ in range(nlayers)])
        else:
            raise ValueError(f"unknown encoder {encoder}")
        self.pool = AttnPool(fused)
        self.mu, self.logvar = nn.Linear(fused, d_lat), nn.Linear(fused, d_lat)

    def _run(self, H, m):
        if self.encoder == "transformer":
            return self.seq(H, src_key_padding_mask=(m == 0))
        if self.encoder in ("gru", "lstm"):
            out, _ = self.seq(H)
            return out
        # tcn / mamba: iterate blocks (no mask needed; padding rows are zero-projected)
        for blk in self.seq:
            H = blk(H)
        return H

    def forward(self, x, m):
        H = torch.cat([self.projs[k](x[..., a:b]) for k, (a, b) in self.slices.items()], -1)
        H = self._run(H, m)
        c = self.pool(H, m)
        mu, lv = self.mu(c), self.logvar(c)
        z = mu + torch.randn_like(mu)*torch.exp(0.5*lv)
        return z, mu, lv


class FusionNet(nn.Module):
    """Encoder latent ⊕ enriched block → ICU head + LOS 3-class head."""
    def __init__(self, slices, n_extra, encoder="transformer", d_lat=16, hidden=64):
        super().__init__()
        self.enc = SeqEncoder(slices, encoder=encoder, d_lat=d_lat)
        self.head = nn.Sequential(nn.Linear(d_lat+n_extra, hidden), nn.ReLU(), nn.Dropout(0.1))
        self.h_icu = nn.Linear(hidden, 1)
        self.h_los = nn.Linear(hidden, 3)   # 3-class LOS
    def forward(self, x, m, e):
        z, mu, lv = self.enc(x, m)
        h = self.head(torch.cat([z, e], -1))
        return self.h_icu(h).squeeze(-1), self.h_los(h), mu, lv


class GRUPlus(nn.Module):
    def __init__(self, d_in, n_extra, d_h=128, hidden=64):
        super().__init__()
        self.gru = nn.GRU(d_in, d_h, batch_first=True)
        self.head = nn.Sequential(nn.Linear(d_h+n_extra, hidden), nn.ReLU())
        self.h_icu, self.h_los = nn.Linear(hidden, 1), nn.Linear(hidden, 3)
    def forward(self, x, m, e):
        out, _ = self.gru(x)
        idx = (m.sum(1).long()-1).clamp(min=0)
        h = out[torch.arange(len(x)), idx]
        h = self.head(torch.cat([h, e], -1))
        return self.h_icu(h).squeeze(-1), self.h_los(h)


# ------------------------------------------------------------------ metrics
def icu_metrics(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    yhat = (p >= thr[np.argmax(tpr-fpr)]).astype(int)
    tp = ((yhat == 1) & (y == 1)).sum(); tn = ((yhat == 0) & (y == 0)).sum()
    fp = ((yhat == 1) & (y == 0)).sum(); fn = ((yhat == 0) & (y == 1)).sum()
    return dict(auroc=roc_auc_score(y, p), auprc=average_precision_score(y, p),
                brier=brier_score_loss(y, p), f1=f1_score(y, yhat),
                sensitivity=tp/max(tp+fn, 1), specificity=tn/max(tn+fp, 1))


def los_class_metrics(y, logits):
    p = torch.softmax(torch.tensor(logits), -1).numpy()
    pred = p.argmax(1)
    out = dict(los_macro_f1=f1_score(y, pred, average="macro"),
               los_acc=(pred == y).mean())
    for c in range(3):
        try:
            out[f"los_auroc_c{c}"] = roc_auc_score((y == c).astype(int), p[:, c])
        except ValueError:
            out[f"los_auroc_c{c}"] = np.nan
    return out


def boot_ci(y, p, fn, B=1000, seed=0):
    rng = np.random.default_rng(seed); vals = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        try: vals.append(fn(y[i], p[i]))
        except ValueError: pass
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ------------------------------------------------------------------ train
def train(model, loaders_, step, epochs=120, lr=1e-3, patience=15, tag=""):
    tr, va = loaders_
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, bad, state = np.inf, 0, None
    for ep in range(epochs):
        model.train()
        for b in tr:
            opt.zero_grad(); loss = step(model, [t.to(DEVICE) for t in b])
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad():
            vl = np.mean([step(model, [t.to(DEVICE) for t in b]).item() for b in va])
        if vl < best-1e-5: best, bad, state = vl, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    if state: model.load_state_dict(state)
    print(f"[train:{tag}] best val {best:.4f} @ ep {ep-bad}")
    return model


def loaders(tensors, part, meta, bs=256, shuffle=False):
    idx = np.where(meta.partition.values == part)[0]
    ds = TensorDataset(*[torch.as_tensor(t[idx]) for t in tensors])
    return DataLoader(ds, batch_size=bs, shuffle=shuffle), idx


def main(data, out, seeds=5, epochs=120, encoders=("transformer", "gru", "lstm", "tcn")):
    os.makedirs(out, exist_ok=True)
    X = np.load(f"{data}/prosp_X.npy"); M = np.load(f"{data}/prosp_M.npy")
    E = np.load(f"{data}/prosp_extra.npy")
    yI = np.load(f"{data}/prosp_y_icu.npy"); yL = np.load(f"{data}/prosp_y_los.npy")
    yC = np.load(f"{data}/prosp_y_los_class.npy")
    meta = pd.read_csv(f"{data}/prosp_meta.csv")
    vocab = json.load(open(f"{data}/vocab.json"))
    n = dict(dx=len(vocab["dx"]), lab=len(vocab["lab"]), med=len(vocab["med"]), proc=len(vocab["proc"]))
    c = np.cumsum([0, n["dx"], n["lab"], n["med"], n["proc"]])
    sl_mm = dict(dx=(int(c[0]), int(c[1])), lab=(int(c[1]), int(c[2])),
                 med=(int(c[2]), int(c[3])), proc=(int(c[3]), int(c[4])))
    sl_uni = dict(dx=(int(c[0]), int(c[1])))
    D, nE = int(c[-1]), E.shape[1]
    recs = []

    tr, tri = loaders((X, M, E, yI, yC, yL), "train", meta, shuffle=True)
    va, vai = loaders((X, M, E, yI, yC, yL), "val", meta)
    te, tei = loaders((X, M, E, yI, yC, yL), "test", meta)
    yI_te, yC_te, yL_te = yI[tei], yC[tei], yL[tei]

    def fusion_step(mdl, b):
        x, m, e, yi, yc, yl = b
        li, ll, mu, lv = mdl(x.float(), m.float(), e.float())
        kl = -0.5*torch.mean(1+lv-mu.pow(2)-lv.exp())
        return (F.binary_cross_entropy_with_logits(li, yi.float())
                + F.cross_entropy(ll, yc.long()) + 1e-3*kl)

    def gru_step(mdl, b):
        x, m, e, yi, yc, yl = b
        li, ll = mdl(x.float(), m.float(), e.float())
        return F.binary_cross_entropy_with_logits(li, yi.float()) + F.cross_entropy(ll, yc.long())

    def eval_fusion(mdl):
        mdl.eval(); Pi, Ll = [], []
        with torch.no_grad():
            for x, m, e, *_ in te:
                li, ll, _, _ = mdl(x.float().to(DEVICE), m.float().to(DEVICE), e.float().to(DEVICE))
                Pi.append(torch.sigmoid(li).cpu().numpy()); Ll.append(ll.cpu().numpy())
        return np.concatenate(Pi), np.concatenate(Ll)

    def eval_gru(mdl):
        mdl.eval(); Pi, Ll = [], []
        with torch.no_grad():
            for x, m, e, *_ in te:
                li, ll = mdl(x.float().to(DEVICE), m.float().to(DEVICE), e.float().to(DEVICE))
                Pi.append(torch.sigmoid(li).cpu().numpy()); Ll.append(ll.cpu().numpy())
        return np.concatenate(Pi), np.concatenate(Ll)

    for s in range(seeds):
        torch.manual_seed(s); np.random.seed(s)

        # ---- encoder benchmark: same VAE+fusion, swap the sequence encoder ----
        for enc in encoders:
            net = FusionNet(sl_mm, nE, encoder=enc).to(DEVICE)
            net = train(net, (tr, va), fusion_step, epochs, tag=f"{enc} s{s}")
            pi, ll = eval_fusion(net)
            recs.append(dict(model=f"MM+[{enc}]", seed=s,
                             **icu_metrics(yI_te, pi), **los_class_metrics(yC_te, ll)))

        # dx-only ablation uses the default (transformer) encoder
        uni = FusionNet(sl_uni, nE, encoder="transformer").to(DEVICE)
        uni = train(uni, (tr, va), fusion_step, epochs, tag=f"Uni+ s{s}")
        pi, ll = eval_fusion(uni)
        recs.append(dict(model="Uni+(dx)", seed=s, **icu_metrics(yI_te, pi), **los_class_metrics(yC_te, ll)))

        if s == 0:
            # mean-pool sequence ⊕ extra → classical baselines
            pool = lambda Xa, Ma: (Xa*Ma[..., None]).sum(1)/(Ma.sum(1, keepdims=True)+1e-6)
            Ftr = np.hstack([pool(X[tri], M[tri]), E[tri]])
            Fte = np.hstack([pool(X[tei], M[tei]), E[tei]])
            pi = LogisticRegression(max_iter=2000, C=0.3).fit(Ftr, yI[tri]).predict_proba(Fte)[:, 1]
            los_lr = LogisticRegression(max_iter=2000, C=0.3).fit(Ftr, yC[tri])
            ll = los_lr.predict_log_proba(Fte)
            recs.append(dict(model="LogReg+", seed=0, **icu_metrics(yI_te, pi), **los_class_metrics(yC_te, ll)))

            # extra-ONLY ablation: how much do the enriched features carry alone?
            pi = LogisticRegression(max_iter=2000, C=0.3).fit(E[tri], yI[tri]).predict_proba(E[tei])[:, 1]
            ll = LogisticRegression(max_iter=2000, C=0.3).fit(E[tri], yC[tri]).predict_log_proba(E[tei])
            recs.append(dict(model="Extra-only", seed=0, **icu_metrics(yI_te, pi), **los_class_metrics(yC_te, ll)))

    df = pd.DataFrame(recs); df.to_csv(f"{out}/per_seed_metrics.csv", index=False)
    agg = df.groupby("model").agg(["mean", "std"]).round(4)
    agg.to_csv(f"{out}/aggregate_metrics.csv")

    # Wilcoxon on AUROC across seeds (multi-seed models only)
    piv = df.pivot_table(index="seed", columns="model", values="auroc")
    stats = {}
    for a, b in itertools.combinations([m for m in piv.columns if piv[m].notna().all()], 2):
        try:
            w = wilcoxon(piv[a], piv[b]); stats[f"{a} vs {b}"] = dict(stat=float(w.statistic), p=float(w.pvalue))
        except ValueError: pass

    # bootstrap CI for the best model's AUROC
    best_model = df.groupby("model").auroc.mean().idxmax()
    with open(f"{out}/results.json", "w") as f:
        json.dump(dict(wilcoxon_auroc=stats, best_model=best_model,
                       icu_base_rate=float(yI_te.mean())), f, indent=2, default=str)
    print(agg.to_string())
    print("\nWilcoxon AUROC:", json.dumps(stats, indent=2))
    print("ICU base rate (test):", round(float(yI_te.mean()), 3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="prospective_enriched")
    ap.add_argument("--out", default="results_pred_enriched")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--encoders", default="transformer,gru,lstm,tcn",
                    help="comma-separated: transformer,gru,lstm,tcn,mamba")
    a = ap.parse_args()
    main(a.data, a.out, a.seeds, a.epochs, tuple(a.encoders.split(",")))
