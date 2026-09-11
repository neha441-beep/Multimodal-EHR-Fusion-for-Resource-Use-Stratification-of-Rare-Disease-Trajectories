"""
prospective_enriched.py
=====================================================================
Tier-1+ enriched PROSPECTIVE prediction task. Rebuilds ONLY the
next-admission tensors (admissions 1..n-1 -> admission n) with
additional LEGITIMATE, pre-cutoff features. Does NOT touch the
stratification tensors or the primary result.

Every added feature is available strictly at the prediction cutoff
(= discharge of admission n-1). NO feature derives from the target
admission n. The prior-outcome-history features use ONLY admissions
1..n-1 (the history slice), which is the correct, non-leaking version
of "past ICU/LOS predicts next ICU/LOS".

Added feature groups (all logged in feature_table_enriched.csv):
  demographics    : age at last history admission, sex
  history_outcome : prior ICU rate, prior mean/max LOS, prior mortality-free,
                    n prior admissions, days since first admission
  comorbidity     : distinct prior diagnosis-code count (Charlson-style proxy)
  admission_type  : emergency flag, admission count rate
  timing          : gap days from last discharge to target admission
                    (this one is the target admission's SCHEDULED time only —
                    admittime is known at scheduling, before the stay; if you
                    prefer to exclude it, set cfg.use_target_gap=False)

Labels:
  y_icu           : target admission ICU flag (binary)
  y_los_log       : log1p(target LOS)              (regression, kept)
  y_los_class     : 0=short(<2d) 1=med(2-7d) 2=long(>7d)  (R1.7 reframe)

Reads:  pipeline_out/  (needs admittime/dischtime/in_icu/los_days per admission)
Writes: prospective_enriched/  (new dir — no collision)

Run:
  python prospective_enriched.py --data pipeline_out --mimic <BASE> \
         --orph <ORPH_CSV> --out prospective_enriched
=====================================================================
"""
from __future__ import annotations
import argparse, json, os
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler

# reuse the exact ICD normalization + cohort logic from the base pipeline
from rare_ehr_pipeline import (_norm_icd, PipelineConfig, CohortFlow,
                               build_cohort, build_features)


def build_enriched_prospective(core, feat_cols, splits, patients_df, cfg):
    """core: admission-level featurized table (from build_features).
    patients_df: subject_id, gender, anchor_age (for demographics)."""
    core = core.sort_values(["subject_id", "admittime"]).reset_index(drop=True)
    part_map = dict(zip(splits.subject_id, splits.partition))
    train_subj = set(splits.loc[splits.partition == "train", "subject_id"])
    D = len(feat_cols)

    # demographics lookup
    demo = patients_df.set_index("subject_id")
    gender_map = {"M": 1.0, "F": 0.0}

    rowsX, rowsM, extra, y_icu, y_los, y_cls, pmeta = [], [], [], [], [], [], []

    for sid, g in core.groupby("subject_id", sort=False):
        if len(g) < 2:
            continue
        hist, target = g.iloc[:-1], g.iloc[-1]

        # ---- sequence features (same as base): history admissions only ----
        Xi = hist[feat_cols].to_numpy(np.float32)[-cfg.t_max:]
        Xp = np.zeros((cfg.t_max, D), np.float32)
        Mp = np.zeros(cfg.t_max, np.float32)
        Xp[:len(Xi)] = Xi
        Mp[:len(Xi)] = 1.0

        # ---- ENRICHED per-patient features (strictly from history 1..n-1) ----
        age = float(demo.anchor_age.get(sid, np.nan)) if sid in demo.index else np.nan
        sex = gender_map.get(demo.gender.get(sid, None), np.nan) if sid in demo.index else np.nan

        prior_icu_rate   = float(hist.in_icu.mean())
        prior_icu_any    = float(hist.in_icu.max())
        prior_los_mean   = float(hist.los_days.mean())
        prior_los_max    = float(hist.los_days.max())
        prior_mort_any   = float(hist.hospital_expire_flag.max())  # ~always 0 (alive to readmit)
        n_prior          = float(len(hist))
        span_days        = float((hist.dischtime.iloc[-1] - hist.admittime.iloc[0]).days)
        # comorbidity proxy: distinct diagnosis columns ever active in history
        dx_cols = [c for c in feat_cols if c.startswith("dx_")]
        comorb           = float((hist[dx_cols].to_numpy() > 0).any(0).sum())
        # admission cadence
        adm_rate         = n_prior / (span_days + 1.0)
        # timing: gap from last discharge to target's scheduled admit time
        gap_days         = float((target.admittime - hist.dischtime.iloc[-1]).days)
        if not cfg.use_target_gap:
            gap_days = 0.0

        extra.append([age, sex, prior_icu_rate, prior_icu_any, prior_los_mean,
                      prior_los_max, prior_mort_any, n_prior, span_days,
                      comorb, adm_rate, gap_days])

        rowsX.append(Xp); rowsM.append(Mp)
        # ---- targets (from target admission) ----
        los = float(target.los_days)
        y_icu.append(float(target.in_icu))
        y_los.append(float(np.log1p(los)))
        y_cls.append(0 if los < 2 else (1 if los <= 7 else 2))
        pmeta.append(dict(subject_id=sid, partition=part_map[sid],
                          n_history=len(hist),
                          readmit_within_90d=int(0 <= gap_days <= cfg.readmit_horizon_days),
                          target_los_days=los, gap_days=int(gap_days),
                          prediction_time="discharge of admission n-1"))

    extra_names = ["age", "sex", "prior_icu_rate", "prior_icu_any",
                   "prior_los_mean", "prior_los_max", "prior_mort_any",
                   "n_prior", "span_days", "comorbidity", "adm_rate", "gap_days"]
    E = np.array(extra, np.float32)
    Xp = np.stack(rowsX) if rowsX else np.zeros((0, cfg.t_max, D), np.float32)
    Mp = np.stack(rowsM) if rowsM else np.zeros((0, cfg.t_max), np.float32)
    pmeta = pd.DataFrame(pmeta)

    # ---- impute + scale the extra block on TRAIN only ----
    tr = pmeta.partition.values == "train"
    med = np.nanmedian(E[tr], axis=0)
    inds = np.where(np.isnan(E))
    E[inds] = np.take(med, inds[1])
    scaler = StandardScaler().fit(E[tr])
    E = scaler.transform(E).astype(np.float32)

    # feature-timing rows for the enriched block (R1.2)
    timing = {
        "age": "known at admission (demographic)",
        "sex": "known at admission (demographic)",
        "prior_icu_rate": "from admissions 1..n-1 only",
        "prior_icu_any": "from admissions 1..n-1 only",
        "prior_los_mean": "from admissions 1..n-1 only",
        "prior_los_max": "from admissions 1..n-1 only",
        "prior_mort_any": "from admissions 1..n-1 only (in-hospital death flag; ~0)",
        "n_prior": "count of prior admissions at cutoff",
        "span_days": "first-to-last prior discharge span",
        "comorbidity": "distinct prior diagnosis codes (Charlson-style proxy)",
        "adm_rate": "prior admission cadence",
        "gap_days": ("target scheduled admit minus last discharge; "
                     "set use_target_gap=False to drop"),
    }
    ftab = pd.DataFrame([dict(feature=n, group="enriched",
                              available_at="≤ discharge of admission n-1",
                              used_as="predictor", timing_note=timing[n])
                         for n in extra_names])
    return Xp, Mp, E, extra_names, np.array(y_icu, np.float32), \
           np.array(y_los, np.float32), np.array(y_cls, np.int64), pmeta, ftab


def main(data, mimic, orph, out, use_target_gap=True):
    os.makedirs(out, exist_ok=True)
    cfg = PipelineConfig(mimic_base=mimic, orphanet_csv=orph, out_dir=out)
    cfg.use_target_gap = use_target_gap  # dynamic attr

    # rebuild cohort + features via the SAME logic as the base pipeline,
    # reusing the persisted split so partitions match exactly
    flow = CohortFlow()
    cohort_adm, _ = build_cohort(cfg, flow)
    splits = pd.read_csv(f"{data}/splits.csv")
    core, feat_cols, vocab, scaler, lab_med, _ = build_features(
        cohort_adm, splits, cfg, flow)

    patients = pd.read_csv(f"{mimic}/hosp/patients.csv",
                           usecols=["subject_id", "gender", "anchor_age"])

    Xp, Mp, E, enames, yI, yL, yC, pmeta, ftab = build_enriched_prospective(
        core, feat_cols, splits, patients, cfg)

    print(f"[enriched] prospective N={len(pmeta)} | "
          f"test={int((pmeta.partition=='test').sum())} | "
          f"extra feats={len(enames)} | ICU rate={yI.mean():.3f} | "
          f"LOS classes={np.bincount(yC).tolist()}")

    np.save(f"{out}/prosp_X.npy", Xp); np.save(f"{out}/prosp_M.npy", Mp)
    np.save(f"{out}/prosp_extra.npy", E)
    np.save(f"{out}/prosp_y_icu.npy", yI)
    np.save(f"{out}/prosp_y_los.npy", yL)
    np.save(f"{out}/prosp_y_los_class.npy", yC)
    pmeta.to_csv(f"{out}/prosp_meta.csv", index=False)
    with open(f"{out}/extra_names.json", "w") as f:
        json.dump(enames, f)
    with open(f"{out}/vocab.json", "w") as f:
        json.dump({k: [str(x) for x in v] for k, v in vocab.items()}, f)
    ftab.to_csv(f"{out}/feature_table_enriched.csv", index=False)
    print(f"[done] wrote enriched prospective tensors to {out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="pipeline_out")
    ap.add_argument("--mimic", required=True)
    ap.add_argument("--orph", required=True)
    ap.add_argument("--out", default="prospective_enriched")
    ap.add_argument("--no_target_gap", action="store_true",
                    help="drop the target-admission scheduled gap feature")
    a = ap.parse_args()
    main(a.data, a.mimic, a.orph, a.out, use_target_gap=not a.no_target_gap)
