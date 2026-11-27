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

Round-2 changes (reviewer minor points; no change to model training):
  * Operating threshold (Youden) is selected on the VALIDATION predictions
    and applied to the test predictions (previously chosen on test).
  * Bootstrap CIs and paired ΔAUROC are computed on SEED-AVERAGED test
    predictions (mean predicted probability over the five seeds), with point
    estimates from the same averaged predictions, so estimate and interval
    refer to the same quantity. Seed-0 CIs are kept under bootstrap_ci_seed0.
  * Constant-predictor Brier score and Brier skill score are reported.
  * The lightweight gated residual comparator is named "gated_residual"
    (alias "mamba" kept for old commands). It implements no state-space model.
  * Validation predictions never touch the global random state: the latent is
    sampled with a private torch.Generator and batches are sliced directly
    (no DataLoader iterator, which would draw from the CPU generator). The
    global CPU/CUDA streams, and the cuDNN RNN dropout state, are therefore
    exactly as in the previous run, so every trained model and every
    five-seed-mean metric in Table 4 is unchanged. (An earlier version used
    torch.random.fork_rng; restoring the CUDA state re-seeds cuDNN's RNN
    dropout state, which changed every model trained after the GRU.)

Run: python run_prediction_enriched.py --data prospective_enriched_nogap --out results_pred_nogap
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
from scipy.stats import wilcoxon, friedmanchisquare

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


class GatedResidual(nn.Module):
    """Lightweight gated residual block: LayerNorm(x + Dropout(sigmoid(W1 x) * W2 x)).
    Not a state-space (Mamba) layer."""
    def __init__(self, d, drop=0.1):
        super().__init__()
        self.f1, self.f2 = nn.Linear(d, d), nn.Linear(d, d)
        self.norm, self.drop = nn.LayerNorm(d), nn.Dropout(drop)
    def forward(self, x):
        return self.norm(x + self.drop(torch.sigmoid(self.f1(x)) * self.f2(x)))


class SeqEncoder(nn.Module):
    """Swappable sequence encoder feeding a variational latent.
    encoder in {transformer, gru, lstm, tcn, gated_residual}. Same latent/pool
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
        elif encoder in ("gated_residual", "mamba"):   # "mamba" = legacy alias
            self.seq = nn.ModuleList([GatedResidual(fused, drop) for _ in range(nlayers)])
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
        # tcn / gated_residual: iterate blocks (no mask needed; padding rows are zero-projected)
        for blk in self.seq:
            H = blk(H)
        return H

    def forward(self, x, m, gen=None):
        H = torch.cat([self.projs[k](x[..., a:b]) for k, (a, b) in self.slices.items()], -1)
        H = self._run(H, m)
        c = self.pool(H, m)
        mu, lv = self.mu(c), self.logvar(c)
        if gen is None:          # training / test path: unchanged
            eps = torch.randn_like(mu)
        else:                    # validation path: private generator only
            eps = torch.randn(mu.shape, device=mu.device, dtype=mu.dtype, generator=gen)
        z = mu + eps*torch.exp(0.5*lv)
        return z, mu, lv


class FusionNet(nn.Module):
    """Encoder latent ⊕ enriched block → ICU head + LOS 3-class head."""
    def __init__(self, slices, n_extra, encoder="transformer", d_lat=16, hidden=64):
        super().__init__()
        self.enc = SeqEncoder(slices, encoder=encoder, d_lat=d_lat)
        self.head = nn.Sequential(nn.Linear(d_lat+n_extra, hidden), nn.ReLU(), nn.Dropout(0.1))
        self.h_icu = nn.Linear(hidden, 1)
        self.h_los = nn.Linear(hidden, 3)   # 3-class LOS
    def forward(self, x, m, e, gen=None):
        z, mu, lv = self.enc(x, m, gen=gen)
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
def youden_threshold(y, p):
    """Youden-optimal threshold. Call on VALIDATION predictions only."""
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr-fpr)])


def icu_metrics(y, p, thr):
    """Threshold-free metrics on (y, p); operating metrics at the supplied
    threshold, which must be chosen on the validation partition."""
    yhat = (p >= thr).astype(int)
    tp = ((yhat == 1) & (y == 1)).sum(); tn = ((yhat == 0) & (y == 0)).sum()
    fp = ((yhat == 1) & (y == 0)).sum(); fn = ((yhat == 0) & (y == 1)).sum()
    sens = tp/max(tp+fn, 1); spec = tn/max(tn+fp, 1)
    ppv = tp/max(tp+fp, 1); npv = tn/max(tn+fn, 1)
    return dict(auroc=roc_auc_score(y, p), auprc=average_precision_score(y, p),
                brier=brier_score_loss(y, p), f1=f1_score(y, yhat),
                sensitivity=sens, specificity=spec,
                ppv=ppv, npv=npv, balanced_acc=0.5*(sens+spec), threshold=float(thr))


def brier_reference(y_te, p, base_rate):
    """Brier of a constant predictor at `base_rate`, and the Brier skill score."""
    b_const = float(np.mean((y_te - base_rate) ** 2))
    b = float(brier_score_loss(y_te, p))
    return dict(brier_constant=b_const, brier_skill=1.0 - b / b_const)


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


def calib_metrics(y, p, n_bins=10):
    p = np.clip(np.asarray(p, float), 1e-6, 1-1e-6); y = np.asarray(y, float)
    edges = np.linspace(0, 1, n_bins+1); ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.sum() > 0:
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    logit = np.log(p/(1-p)).reshape(-1, 1)
    try:
        lr = LogisticRegression(fit_intercept=True, C=1e6, solver="lbfgs").fit(logit, y)
        slope = float(lr.coef_[0][0]); intercept = float(lr.intercept_[0])
    except Exception:
        slope, intercept = float("nan"), float("nan")
    return dict(ece=float(ece), calib_slope=slope, calib_intercept=intercept)


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
    preds0, los_regression = {}, {}
    allpreds = {}          # model -> list over seeds of (p_test, los_logits_test, p_val)

    tr, tri = loaders((X, M, E, yI, yC, yL), "train", meta, shuffle=True)
    va, vai = loaders((X, M, E, yI, yC, yL), "val", meta)
    te, tei = loaders((X, M, E, yI, yC, yL), "test", meta)
    yI_te, yC_te, yL_te = yI[tei], yC[tei], yL[tei]
    yI_va = yI[vai]
    base_rate_te = float(yI_te.mean())

    def fusion_step(mdl, b):
        x, m, e, yi, yc, yl = b
        li, ll, mu, lv = mdl(x.float(), m.float(), e.float())
        kl = -0.5*torch.mean(1+lv-mu.pow(2)-lv.exp())
        return (F.binary_cross_entropy_with_logits(li, yi.float())
                + F.cross_entropy(ll, yc.long()) + 1e-3*kl)

    def fusion_step_nokl(mdl, b):
        x, m, e, yi, yc, yl = b
        li, ll, mu, lv = mdl(x.float(), m.float(), e.float())
        return (F.binary_cross_entropy_with_logits(li, yi.float())
                + F.cross_entropy(ll, yc.long()))

    def gru_step(mdl, b):
        x, m, e, yi, yc, yl = b
        li, ll = mdl(x.float(), m.float(), e.float())
        return F.binary_cross_entropy_with_logits(li, yi.float()) + F.cross_entropy(ll, yc.long())

    def eval_fusion(mdl, loader=None):
        loader = te if loader is None else loader
        mdl.eval(); Pi, Ll = [], []
        with torch.no_grad():
            for x, m, e, *_ in loader:
                li, ll, _, _ = mdl(x.float().to(DEVICE), m.float().to(DEVICE), e.float().to(DEVICE))
                Pi.append(torch.sigmoid(li).cpu().numpy()); Ll.append(ll.cpu().numpy())
        return np.concatenate(Pi), np.concatenate(Ll)

    Xv, Mv, Ev = (torch.as_tensor(X[vai]).float(), torch.as_tensor(M[vai]).float(),
                  torch.as_tensor(E[vai]).float())

    def eval_val(mdl, seed):
        """Validation probabilities without touching the global CPU/CUDA random
        state or cuDNN's RNN dropout state: z is sampled from a private
        generator and batches are sliced directly (no DataLoader)."""
        gen = torch.Generator(device=DEVICE); gen.manual_seed(10_000 + seed)
        mdl.eval(); P = []
        with torch.no_grad():
            for i in range(0, len(Xv), 256):
                li, _, _, _ = mdl(Xv[i:i+256].to(DEVICE), Mv[i:i+256].to(DEVICE),
                                  Ev[i:i+256].to(DEVICE), gen=gen)
                P.append(torch.sigmoid(li).cpu().numpy())
        return np.concatenate(P)

    def record(name, s, pi, ll, pv):
        thr = youden_threshold(yI_va, pv)
        allpreds.setdefault(name, []).append((pi, ll, pv))
        if s == 0: preds0[name] = (pi, ll)
        recs.append(dict(model=name, seed=s, **icu_metrics(yI_te, pi, thr),
                         **brier_reference(yI_te, pi, base_rate_te),
                         **calib_metrics(yI_te, pi), **los_class_metrics(yC_te, ll)))

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
            pi, ll = eval_fusion(net); pv = eval_val(net, s)
            record(f"MM+[{enc}]", s, pi, ll, pv)

        # dx-only ablation uses the default (transformer) encoder
        uni = FusionNet(sl_uni, nE, encoder="transformer").to(DEVICE)
        uni = train(uni, (tr, va), fusion_step, epochs, tag=f"Uni+ s{s}")
        pi, ll = eval_fusion(uni); pv = eval_val(uni, s)
        record("Uni+(dx)", s, pi, ll, pv)

               # multimodal WITHOUT variational regularisation (KL weight = 0) -- Comment 6 baseline
        nokl = FusionNet(sl_mm, nE, encoder="transformer").to(DEVICE)
        nokl = train(nokl, (tr, va), fusion_step_nokl, epochs, tag=f"noVAE s{s}")
        pi, ll = eval_fusion(nokl); pv = eval_val(nokl, s)
        record("MM+[transformer,noVAE]", s, pi, ll, pv)

        # lightweight gated residual comparator (no state-space mechanism); placed
        # AFTER uni/noVAE so the other models' RNG stream is unchanged
        gres = FusionNet(sl_mm, nE, encoder="gated_residual").to(DEVICE)
        gres = train(gres, (tr, va), fusion_step, epochs, tag=f"gated_residual s{s}")
        pi, ll = eval_fusion(gres); pv = eval_val(gres, s)
        record("MM+[gated_residual]", s, pi, ll, pv)

        if s == 0:
            # mean-pool sequence ⊕ extra → classical baselines
            pool = lambda Xa, Ma: (Xa*Ma[..., None]).sum(1)/(Ma.sum(1, keepdims=True)+1e-6)
            Ftr = np.hstack([pool(X[tri], M[tri]), E[tri]])
            Fte = np.hstack([pool(X[tei], M[tei]), E[tei]])
            Fva = np.hstack([pool(X[vai], M[vai]), E[vai]])
            icu_lr = LogisticRegression(max_iter=2000, C=0.3).fit(Ftr, yI[tri])
            pi, pv = icu_lr.predict_proba(Fte)[:, 1], icu_lr.predict_proba(Fva)[:, 1]
            los_lr = LogisticRegression(max_iter=2000, C=0.3).fit(Ftr, yC[tri])
            ll = los_lr.predict_log_proba(Fte)
            record("LogReg+", 0, pi, ll, pv)

            # extra-ONLY ablation: how much do the enriched features carry alone?
            icu_ex = LogisticRegression(max_iter=2000, C=0.3).fit(E[tri], yI[tri])
            pi, pv = icu_ex.predict_proba(E[tei])[:, 1], icu_ex.predict_proba(E[vai])[:, 1]
            ll = LogisticRegression(max_iter=2000, C=0.3).fit(E[tri], yC[tri]).predict_log_proba(E[tei])
            pi_extra, ll_extra, pv_extra = pi, ll, pv
            # continuous LOS regression (day scale) -- justifies the 3-class reframing (Comment 7)
            los_reg = Ridge(alpha=1.0).fit(Ftr, yL[tri])
            yl_pred_day = np.expm1(los_reg.predict(Fte)); yl_true_day = np.expm1(yL_te)
            los_regression = dict(
                mae_days=float(mean_absolute_error(yl_true_day, yl_pred_day)),
                rmse_days=float(np.sqrt(mean_squared_error(yl_true_day, yl_pred_day))),
                r2_days=float(r2_score(yl_true_day, yl_pred_day)))
            record("Extra-only", 0, pi_extra, ll_extra, pv_extra)

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

    # Friedman omnibus across multi-seed encoders + Holm-corrected Wilcoxon (Comment 8)
    multiseed = [m for m in piv.columns if piv[m].notna().all()]
    friedman = None
    if len(multiseed) >= 3:
        fr = friedmanchisquare(*[piv[m].values for m in multiseed])
        friedman = dict(models=multiseed, stat=float(fr.statistic), p=float(fr.pvalue))
    running = 0.0
    for rank, (k, v) in enumerate(sorted(stats.items(), key=lambda kv: kv[1]["p"])):
        running = max(running, min(1.0, (len(stats) - rank) * v["p"])); v["p_holm"] = float(running)

    # ---- seed-averaged predictions: one probability per test patient per model ----
    avg = {}
    for mdl, runs in allpreds.items():
        p_te = np.mean([r[0] for r in runs], axis=0)
        pcls_te = np.mean([torch.softmax(torch.tensor(r[1]), -1).numpy() for r in runs], axis=0)
        p_va = np.mean([r[2] for r in runs], axis=0)
        avg[mdl] = (p_te, pcls_te, p_va)

    def los_f1(yy, pp): return f1_score(yy, pp.argmax(1), average="macro")
    def bss(yy, pp):    return 1.0 - brier_score_loss(yy, pp) / np.mean((yy - base_rate_te) ** 2)

    seed_avg, boot = {}, {}
    for mdl, (p_te, pcls_te, p_va) in avg.items():
        thr = youden_threshold(yI_va, p_va)
        seed_avg[mdl] = dict(n_seeds=len(allpreds[mdl]),
                             **icu_metrics(yI_te, p_te, thr),
                             **brier_reference(yI_te, p_te, base_rate_te),
                             **calib_metrics(yI_te, p_te),
                             los_macro_f1=float(los_f1(yC_te, pcls_te)))
        boot[mdl] = dict(
            auroc_ci=boot_ci(yI_te, p_te, roc_auc_score),
            auprc_ci=boot_ci(yI_te, p_te, average_precision_score),
            brier_ci=boot_ci(yI_te, p_te, brier_score_loss),
            brier_skill_ci=boot_ci(yI_te, p_te, bss),
            los_macro_f1_ci=boot_ci(yC_te, pcls_te, los_f1))
    pd.DataFrame(seed_avg).T.round(4).to_csv(f"{out}/seed_averaged_metrics.csv")
    np.savez(f"{out}/seed_averaged_predictions.npz", y_test=yI_te, y_val=yI_va,
             **{f"test__{k}": v[0] for k, v in avg.items()},
             **{f"val__{k}": v[2] for k, v in avg.items()})

    # seed-0 CIs kept for comparison with the previous submission only
    boot0 = {}
    for mdl, (pi, ll) in preds0.items():
        pcls = torch.softmax(torch.tensor(ll), -1).numpy()
        boot0[mdl] = dict(
            auroc_ci=boot_ci(yI_te, pi, roc_auc_score),
            auprc_ci=boot_ci(yI_te, pi, average_precision_score),
            brier_ci=boot_ci(yI_te, pi, brier_score_loss),
            los_macro_f1_ci=boot_ci(yC_te, pcls, los_f1))

    # paired patient-level bootstrap on SEED-AVERAGED predictions: ΔAUROC vs proposed
    def paired_delta_ci(mA, mB, B=1000, seed=0):
        pA, pB = avg[mA][0], avg[mB][0]; rng = np.random.default_rng(seed); d = []
        for _ in range(B):
            i = rng.integers(0, len(yI_te), len(yI_te))
            try: d.append(roc_auc_score(yI_te[i], pA[i]) - roc_auc_score(yI_te[i], pB[i]))
            except ValueError: pass
        return dict(delta_auroc=float(roc_auc_score(yI_te, pA) - roc_auc_score(yI_te, pB)),
                    bootstrap_mean=float(np.mean(d)),
                    ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])
    ref = "MM+[transformer]"
    paired = ({f"{ref} vs {m}": paired_delta_ci(ref, m) for m in avg if m != ref}
              if ref in avg else {})

    best_model = df.groupby("model").auroc.mean().idxmax()
    with open(f"{out}/results.json", "w") as f:
        json.dump(dict(wilcoxon_auroc=stats, friedman=friedman,
                       seed_averaged_metrics=seed_avg,
                       bootstrap_ci=boot, bootstrap_ci_seed0=boot0,
                       paired_bootstrap_delta_auroc=paired,
                       constant_predictor=dict(base_rate_test=base_rate_te,
                           brier=float(np.mean((yI_te - base_rate_te) ** 2)),
                           base_rate_train=float(yI[tri].mean())),
                       threshold_selection="Youden on validation predictions",
                       los_regression=los_regression, best_model=best_model,
                       icu_base_rate=float(yI_te.mean())), f, indent=2, default=str)
    print(agg.to_string())
    print("\nWilcoxon AUROC:", json.dumps(stats, indent=2))
    print("ICU base rate (test):", round(base_rate_te, 4),
          "| constant-predictor Brier:", round(float(np.mean((yI_te - base_rate_te) ** 2)), 4))
    print("\nSeed-averaged metrics with bootstrap 95% CI:")
    for mdl in seed_avg:
        sa, b = seed_avg[mdl], boot[mdl]
        print(f"  {mdl:<26s} AUROC {sa['auroc']:.3f} [{b['auroc_ci'][0]:.3f}, {b['auroc_ci'][1]:.3f}]"
              f"  AUPRC {sa['auprc']:.3f} [{b['auprc_ci'][0]:.3f}, {b['auprc_ci'][1]:.3f}]"
              f"  Brier {sa['brier']:.3f} [{b['brier_ci'][0]:.3f}, {b['brier_ci'][1]:.3f}]"
              f"  BSS {sa['brier_skill']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="prospective_enriched")
    ap.add_argument("--out", default="results_pred_enriched")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--encoders", default="transformer,gru,lstm,tcn",
                    help="comma-separated: transformer,gru,lstm,tcn (the gated residual "
                         "comparator always runs after these)")
    a = ap.parse_args()
    main(a.data, a.out, a.seeds, a.epochs, tuple(a.encoders.split(",")))