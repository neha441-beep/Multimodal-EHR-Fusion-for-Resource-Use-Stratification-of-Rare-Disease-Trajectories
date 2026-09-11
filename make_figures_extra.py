"""
make_figures_extra.py  (revised)
=====================================================================
Generates the two remaining reviewer-requested figures:
  fig7_calibration.pdf/png   — ICU reliability curve + Brier (R1.16)
  fig8_tsne_test.pdf/png      — t-SNE of TEST-set latents, GMM K=3 (R1.14)

CHANGE vs the previous version: fig7 now aggregates over the SAME five
seeds used for Table 3 instead of a single seed. It pools the test-set
predicted probabilities across seeds and computes the reliability curve
and Brier on the pooled set. Pooling equal-size per-seed prediction sets
makes the pooled Brier identical to the mean of the per-seed Briers,
i.e. the value reported in Table 3 (0.139) — so the figure legend and the
manuscript now agree by construction. The script prints each per-seed
Brier and the pooled Brier; if the pooled value is not 0.139, set the
manuscript to the printed value (it is the single source of truth for
Sec. 5.3, Sec. 5.4, the Fig. 7 caption, and the Table 3 ICU-Brier cell).

Both figures are written as vector PDF (matplotlib PDF backend) and PNG.

Run AFTER pipeline_out/ and prospective_enriched/ exist:
  python make_figures_extra.py
=====================================================================
"""
import os, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from run_prediction_enriched import FusionNet, DEVICE, train, loaders
from run_experiments import MultimodalVaDeSC
from stratification_deep import train_encoder, latents, slices_for

PIPE  = "pipeline_out"
PROSP = "prospective_enriched"
OUT   = "figures"
SEEDS = 5          # match Table 3 (mean over five seeds)
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})
BLUE, ORANGE = "#2c6fbb", "#e08b2d"


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")   # vector
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight")
    plt.close(fig); print(f"  wrote {name}")


# ---------------- FIG 7: calibration (proposed model, prospective ICU) ----
# 5-seed pooled so the Brier equals the Table 3 mean (0.139).
def fig7_calibration(seeds=SEEDS, epochs=120):
    import torch.nn.functional as F
    X = np.load(f"{PROSP}/prosp_X.npy"); M = np.load(f"{PROSP}/prosp_M.npy")
    E = np.load(f"{PROSP}/prosp_extra.npy")
    yI = np.load(f"{PROSP}/prosp_y_icu.npy"); yC = np.load(f"{PROSP}/prosp_y_los_class.npy")
    yL = np.load(f"{PROSP}/prosp_y_los.npy")
    meta = pd.read_csv(f"{PROSP}/prosp_meta.csv")
    vocab = json.load(open(f"{PROSP}/vocab.json"))
    n = [len(vocab[k]) for k in ["dx", "lab", "med", "proc"]]
    c = np.cumsum([0]+n)
    sl = dict(dx=(int(c[0]), int(c[1])), lab=(int(c[1]), int(c[2])),
              med=(int(c[2]), int(c[3])), proc=(int(c[3]), int(c[4])))

    def step(mdl, b):
        x, m, e, yi, yc, yl = b
        li, ll, mu, lv = mdl(x.float(), m.float(), e.float())
        kl = -0.5*torch.mean(1+lv-mu.pow(2)-lv.exp())
        return (F.binary_cross_entropy_with_logits(li, yi.float())
                + F.cross_entropy(ll, yc.long()) + 1e-3*kl)

    per_seed_brier, pooled_p, pooled_y = [], [], []
    for s in range(seeds):
        tr, _   = loaders((X, M, E, yI, yC, yL), "train", meta, shuffle=True)
        va, _   = loaders((X, M, E, yI, yC, yL), "val",  meta)
        te, tei = loaders((X, M, E, yI, yC, yL), "test", meta)
        torch.manual_seed(s); np.random.seed(s)
        net = FusionNet(sl, E.shape[1], encoder="transformer").to(DEVICE)
        net = train(net, (tr, va), step, epochs, tag=f"calib_s{s}")
        net.eval(); P = []
        with torch.no_grad():
            for x, m, e, *_ in te:
                li, _, _, _ = net(x.float().to(DEVICE), m.float().to(DEVICE), e.float().to(DEVICE))
                P.append(torch.sigmoid(li).cpu().numpy())
        p = np.concatenate(P); y = yI[tei]
        b = brier_score_loss(y, p); per_seed_brier.append(b)
        pooled_p.append(p); pooled_y.append(y)
        print(f"  [seed {s}] Brier={b:.4f}")

    p = np.concatenate(pooled_p); y = np.concatenate(pooled_y)
    brier = brier_score_loss(y, p)               # == mean of per-seed Briers
    print(f"  per-seed mean Brier={np.mean(per_seed_brier):.4f}  "
          f"pooled Brier={brier:.4f}  -> legend shows {brier:.3f}")
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=8, strategy="quantile")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, "-o", color=BLUE, lw=2,
            label=f"Multimodal VaDeSC-EHR (Brier={brier:.3f})")
    ax.set_xlabel("Mean predicted ICU probability")
    ax.set_ylabel("Observed ICU frequency")
    ax.set_title("ICU prediction calibration (test set)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    save(fig, "fig7_calibration")


# ---------------- FIG 8: t-SNE of TEST latents (unsupervised, K=3) --------
def fig8_tsne(seed=0):
    Xs = np.load(f"{PIPE}/strat_X.npy"); Ms = np.load(f"{PIPE}/strat_M.npy")
    smeta = pd.read_csv(f"{PIPE}/strat_meta.csv")
    vocab = json.load(open(f"{PIPE}/vocab.json"))
    tr = smeta.partition.values == "train"; te = smeta.partition.values == "test"
    sl, _ = slices_for(vocab, ["dx", "lab", "med", "proc"])

    torch.manual_seed(seed); np.random.seed(seed)
    enc = MultimodalVaDeSC(sl, supervised=False).to(DEVICE)
    enc = train_encoder(enc, Xs[tr], Ms[tr])
    Ztr, Zte = latents(enc, Xs[tr], Ms[tr]), latents(enc, Xs[te], Ms[te])
    gmm = GaussianMixture(3, covariance_type="full", reg_covar=1e-4,
                          random_state=0, n_init=5).fit(Ztr)
    lab = gmm.predict(Zte)

    Zp = PCA(n_components=min(30, Zte.shape[1]), random_state=0).fit_transform(Zte)
    var = PCA(n_components=min(30, Zte.shape[1]), random_state=0).fit(Zte).explained_variance_ratio_.sum()
    Z2 = TSNE(n_components=2, perplexity=30, learning_rate="auto",
              init="pca", random_state=seed).fit_transform(Zp)

    fig, ax = plt.subplots(figsize=(6, 5.2))
    for k in sorted(np.unique(lab)):
        s = lab == k
        ax.scatter(Z2[s, 0], Z2[s, 1], s=12, alpha=0.75, label=f"Cluster {k} (n={s.sum()})")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.set_title("Test-set patient latents (K=3, unsupervised)")
    ax.legend(fontsize=8, loc="best")
    ax.text(0.01, -0.13, f"PCA\u2192{Zp.shape[1]}d ({var:.0%} var) then t-SNE "
            f"(perplexity 30, PCA init, seed {seed})",
            transform=ax.transAxes, fontsize=7, color="#666")
    save(fig, "fig8_tsne_test")


if __name__ == "__main__":
    print("Generating extra figures \u2192", OUT)
    fig7_calibration()
    fig8_tsne()
    print("done.")
