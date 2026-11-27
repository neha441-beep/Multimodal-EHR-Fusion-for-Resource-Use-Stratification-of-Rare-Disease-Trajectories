"""
stratification_sensitivity.py
=====================================================================
Round-2 sensitivity analysis (reviewer: "a sensitivity analysis excluding
the most frequent common codes would show whether the strata depend on
them").

Re-runs the PRIMARY stratification analysis on the exact-match cohort after
removing patients whose cohort membership rests on common codes
(default: G20, C61, G35, I81, N10), using the same frozen features, the same
patient-level split, the same encoder, K = 3, and the same evaluation as
stratification_deep.py. Nothing in pipeline_out/ is modified.

Two exclusion rules are reported:
  only : drop patients whose ONLY matched Orphanet-associated codes are in the
         excluded set (patients who also carry another matched code stay)
  any  : drop every patient carrying ANY excluded code

For each rule it reports:
  1. cohort size by partition;
  2. modality ablation (validation partition, five seeds, fixed K = 3):
     stability ARI, Silhouette, Calinski-Harabasz, Davies-Bouldin,
     GMM/k-means ARI -- directly comparable with Table 2;
  3. seed-0 full model applied once to the retained TEST patients: per-stratum
     n, ICU use, mean total LOS, one-year mortality (comparable with Table 3)
     and outcome-separation tests (chi-square / Kruskal-Wallis);
  4. agreement with the main analysis: the main seed-0 full model (trained on
     the full cohort, as in stratification_deep.py) assigns the same retained
     test patients to strata; adjusted Rand index between the two assignments
     and the cross-tabulation.

Features, vocabularies, and the scaler stay those fitted on the full-cohort
training partition, so differences reflect cohort composition only.

Run (from the folder holding stratification_deep.py and run_experiments.py,
after rare_ehr_pipeline.py has written pipeline_out/):
  python stratification_sensitivity.py --data pipeline_out \
      --mimic /path/to/mimic-iv-2.1 --orph /path/to/icd10_diseases.csv \
      --out results_strat_sensitivity --seeds 5
=====================================================================
"""
from __future__ import annotations
import argparse, itertools, json, os
import numpy as np, pandas as pd, torch
from sklearn.metrics import adjusted_rand_score

from run_experiments import MultimodalVaDeSC, DEVICE
from stratification_deep import (load, slices_for, train_encoder, latents,
                                 cluster_eval, outcome_separation)

SUBSETS = {"dx": ["dx"], "dx+lab": ["dx", "lab"],
           "dx+lab+med": ["dx", "lab", "med"], "full": ["dx", "lab", "med", "proc"]}


def norm_icd(s):
    return s.astype(str).str.upper().str.replace(".", "", regex=False).str.strip()


def matched_codes(mimic, orph_csv, subjects):
    """Per-patient set of exact-matched Orphanet-associated ICD-10 codes, using
    the same rule as rare_ehr_pipeline.build_cohort (ICD-10 rows only, dots
    removed, exact match, admissions with valid admit/discharge times)."""
    adm = pd.read_csv(f"{mimic}/hosp/admissions.csv",
                      usecols=["subject_id", "hadm_id", "admittime", "dischtime"],
                      parse_dates=["admittime", "dischtime"])
    adm = adm.dropna(subset=["admittime", "dischtime"])
    adm = adm[adm.dischtime > adm.admittime]
    dx = pd.read_csv(f"{mimic}/hosp/diagnoses_icd.csv",
                     usecols=["subject_id", "hadm_id", "icd_code", "icd_version"])
    dx = dx[(dx.icd_version == 10) & dx.subject_id.isin(subjects)
            & dx.hadm_id.isin(set(adm.hadm_id))].copy()
    dx["icd_norm"] = norm_icd(dx.icd_code)
    orph = pd.read_csv(orph_csv)
    rare = set(norm_icd(orph["ICDcodes"].dropna()))
    dx = dx[dx.icd_norm.isin(rare)]
    return dx.groupby("subject_id").icd_norm.agg(lambda s: frozenset(s))


def fit_full(Xs, Ms, tr_idx, seed=0):
    sl, _ = slices_for(VOCAB, SUBSETS["full"])
    torch.manual_seed(seed); np.random.seed(seed)
    m = MultimodalVaDeSC(sl, supervised=False).to(DEVICE)
    return train_encoder(m, Xs[tr_idx], Ms[tr_idx])


def profile(meta, labels):
    p = meta.copy(); p["cluster"] = labels
    prof = (p.groupby("cluster")
            .agg(n=("subject_id", "size"), icu_any=("icu_any", "mean"),
                 total_los_days=("total_los_days", "mean"),
                 death_365d=("death_within_365d", "mean"),
                 n_admissions=("n_admissions", "mean"))
            .sort_values("icu_any").round(3).reset_index())
    prof.insert(1, "acuity_rank", ["low", "intermediate", "high"][:len(prof)]
                if len(prof) == 3 else range(len(prof)))
    return prof, outcome_separation(p)


def run_rule(rule, keep, Xs, Ms, smeta, main_enc, K, seeds, out):
    tr = np.where(keep & (smeta.partition.values == "train"))[0]
    va = np.where(keep & (smeta.partition.values == "val"))[0]
    te = np.where(keep & (smeta.partition.values == "test"))[0]
    res = dict(rule=rule, n_train=len(tr), n_val=len(va), n_test=len(te))
    print(f"\n[{rule}] retained patients: train={len(tr)} val={len(va)} test={len(te)}")

    rows = []
    for name, sub in SUBSETS.items():
        sl, _ = slices_for(VOCAB, sub)
        per_seed, labs = [], []
        for s in range(seeds):
            torch.manual_seed(s); np.random.seed(s)
            m = MultimodalVaDeSC(sl, supervised=False).to(DEVICE)
            m = train_encoder(m, Xs[tr], Ms[tr])
            met, lab, _ = cluster_eval(latents(m, Xs[tr], Ms[tr]),
                                       latents(m, Xs[va], Ms[va]), K)
            per_seed.append(met); labs.append(lab)
        d = pd.DataFrame(per_seed)
        row = dict(rule=rule, subset=name,
                   stability_ari=float(np.mean([adjusted_rand_score(a, b)
                                                for a, b in itertools.combinations(labs, 2)])),
                   silhouette_mean=float(d.silhouette.mean()),
                   calinski_mean=float(d.calinski.mean()),
                   davies_mean=float(d.davies.mean()),
                   gmm_kmeans_ari=float(d.gmm_kmeans_ari.mean()))
        rows.append(row)
        print(f"[{rule}] {name:<11s} stab_ARI={row['stability_ari']:.3f} "
              f"sil={row['silhouette_mean']:.3f} CH={row['calinski_mean']:.1f} "
              f"DB={row['davies_mean']:.3f} gmm/km_ARI={row['gmm_kmeans_ari']:.3f}")
    abl = pd.DataFrame(rows)

    # seed-0 full model on the retained cohort, applied once to retained test
    enc = fit_full(Xs, Ms, tr, seed=0)
    Ztr, Zte = latents(enc, Xs[tr], Ms[tr]), latents(enc, Xs[te], Ms[te])
    met_te, lab_te, _ = cluster_eval(Ztr, Zte, K)
    prof, sep = profile(smeta.iloc[te], lab_te)
    res.update(test_indices={k: float(v) for k, v in met_te.items()},
               outcome_separation_test=sep)
    print(f"[{rule}] test strata profile:\n{prof.to_string(index=False)}")
    print(f"[{rule}] outcome separation (test): {json.dumps(sep)}")

    # agreement with the main analysis on the same retained test patients
    tr_full = np.where(smeta.partition.values == "train")[0]
    Zm_tr = latents(main_enc, Xs[tr_full], Ms[tr_full])
    _, lab_main, _ = cluster_eval(Zm_tr, latents(main_enc, Xs[te], Ms[te]), K)
    ari = float(adjusted_rand_score(lab_main, lab_te))
    xt = pd.crosstab(pd.Series(lab_main, name="main_stratum"),
                     pd.Series(lab_te, name="sensitivity_stratum"))
    res.update(ari_vs_main_on_retained_test=ari,
               crosstab_main_vs_sensitivity=xt.to_dict())
    print(f"[{rule}] ARI vs main strata on the same {len(te)} test patients: {ari:.3f}")
    print(xt.to_string())

    abl.to_csv(f"{out}/ablation_{rule}.csv", index=False)
    prof.to_csv(f"{out}/test_profile_{rule}.csv", index=False)
    xt.to_csv(f"{out}/crosstab_main_vs_{rule}.csv")
    return res


def main(a):
    global VOCAB
    os.makedirs(a.out, exist_ok=True)
    Xs, Ms, smeta, VOCAB = load(a.data)
    excl = set(norm_icd(pd.Series(a.exclude.split(","))))
    codes = matched_codes(a.mimic, a.orph, set(smeta.subject_id))
    pc = smeta.subject_id.map(codes)
    if pc.isna().any():
        raise SystemExit(f"{int(pc.isna().sum())} cohort patients have no matched code; "
                         "check --mimic/--orph match the pipeline run")
    keep_only = ~pc.map(lambda s: s <= excl).values      # has >=1 non-excluded code
    keep_any = ~pc.map(lambda s: bool(s & excl)).values  # carries no excluded code
    summary = dict(excluded_codes=sorted(excl), n_cohort=int(len(smeta)),
                   n_retained_only=int(keep_only.sum()), n_retained_any=int(keep_any.sum()))
    print("[cohort]", summary)

    # main analysis seed-0 full model (same as stratification_deep.py)
    main_enc = fit_full(Xs, Ms, np.where(smeta.partition.values == "train")[0], seed=0)
    tr_full = np.where(smeta.partition.values == "train")[0]
    te_full = np.where(smeta.partition.values == "test")[0]
    _, lab_full, _ = cluster_eval(latents(main_enc, Xs[tr_full], Ms[tr_full]),
                                  latents(main_enc, Xs[te_full], Ms[te_full]), a.k)
    sizes = sorted(np.bincount(lab_full).tolist())
    summary["main_model_test_stratum_sizes"] = sizes
    print(f"[check] main seed-0 test strata sizes {sizes} (paper Table 3: [55, 225, 325])")

    results = [run_rule(r, k, Xs, Ms, smeta, main_enc, a.k, a.seeds, a.out)
               for r, k in [("only", keep_only), ("any", keep_any)]]
    with open(f"{a.out}/sensitivity.json", "w") as f:
        json.dump(dict(summary=summary, results=results), f, indent=2, default=str)
    print(f"\n[done] wrote {a.out}/sensitivity.json, ablation_*.csv, "
          "test_profile_*.csv, crosstab_main_vs_*.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="pipeline_out")
    ap.add_argument("--mimic", required=True, help="MIMIC-IV v2.1 folder containing hosp/")
    ap.add_argument("--orph", required=True, help="Orphanet mapping CSV used by the pipeline")
    ap.add_argument("--out", default="results_strat_sensitivity")
    ap.add_argument("--exclude", default="G20,C61,G35,I81,N10")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--k", type=int, default=3)
    main(ap.parse_args())
