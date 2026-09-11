"""
rare_ehr_pipeline.py
=====================================================================
Unified, leakage-free cohort + feature pipeline for the revised
"Multimodal EHR Fusion for Resource-Use Stratification of Rare-Disease
Patient Trajectories" manuscript (MIMIC-IV v2.1, Orphanet cohort).

This module is the SINGLE SOURCE OF TRUTH for all models (Multimodal
VaDeSC-EHR, Unimodal VaDeSC-EHR, EHR-Mamba, LR/GRU baselines).
Every model imports its tensors, splits, and preprocessing from here.

Design decisions (mapped to Reviewer 1 points):
  R1.2  (leakage)      : NO outcome-derived predictors. icu_los_hours,
                         transfer_events, note counts, LOS, ICU flag,
                         mortality are NEVER inputs. Scaler, imputation
                         medians, and top-K vocabularies fit on TRAIN
                         patients only.
  R1.4  (cohort)       : ICD-10-only matching against Orphanet (dots
                         stripped, uppercased on both sides); stage-by-
                         stage exclusion counts written to cohort_flow.csv;
                         disease frequency table written to
                         disease_frequency.csv; per-feature metadata to
                         feature_table.csv (the reviewer's predictor-
                         timing table).
  R1.5  (validation)   : patient-level 70/15/15 train/val/test split,
                         all admissions of a patient in one partition,
                         split IDs persisted to splits.csv.
  R1.1  (objective)    : two task exports —
                         (A) STRATIFICATION: full-trajectory tensors for
                             unsupervised phenotyping (primary objective).
                         (B) PROSPECTIVE: for patients with >=2 admissions,
                             features from admissions 1..(n-1) predict
                             admission n's ICU flag and LOS (secondary
                             objective; prediction time = discharge of
                             admission n-1, horizon = next admission).
  R1.10 (circularity)  : independent cluster-validation outcomes that are
                         NEVER in any training loss: 90-day readmission,
                         1-year post-discharge mortality (from patients.dod),
                         and future bed-days.

Usage (on Lightning AI / Kaggle with MIMIC-IV mounted):
    from rare_ehr_pipeline import PipelineConfig, run_pipeline
    cfg = PipelineConfig(mimic_base="/path/to/mimic-iv-2.1",
                         orphanet_csv="/path/to/icd10_diseases.csv",
                         out_dir="./pipeline_out")
    data = run_pipeline(cfg)

Outputs (all under cfg.out_dir):
    cohort_flow.csv, disease_frequency.csv, feature_table.csv, splits.csv,
    strat_{X,M}.npy + strat_meta.csv,
    prosp_{X,M,y_icu,y_los}.npy + prosp_meta.csv,
    scaler.joblib, vocab.json, config.json
=====================================================================
"""

from __future__ import annotations
import os, json, gc, hashlib
from dataclasses import dataclass, asdict, field
from collections import Counter

import numpy as np
import pandas as pd


# --------------------------------------------------------------------
# Config
# --------------------------------------------------------------------
@dataclass
class PipelineConfig:
    mimic_base: str = "/kaggle/input/mimic-iv-2-1/mimic-iv-2.1"
    orphanet_csv: str = "/kaggle/input/rare-diseases/icd10_diseases.csv"
    out_dir: str = "./pipeline_out"

    # Feature vocabulary sizes (selected on TRAIN only)
    k_dx: int = 128
    k_lab: int = 32
    k_med: int = 64
    k_proc: int = 64

    # Sequence cap — SINGLE value for ALL models (R1.6 fairness)
    t_max: int = 8

    # Split fractions (patient-level) and seed
    train_frac: float = 0.70
    val_frac: float = 0.15   # test = 1 - train - val
    seed: int = 42

    # Orphanet ICD matching: "exact" (code string equality, comparable to the
    # original submission) or "prefix" (a MIMIC ICD-10-CM code matches if it
    # starts with an Orphanet code, so C649 matches C64). Run both as a
    # sensitivity analysis for R1.4/R2.3.
    icd_match: str = "exact"

    # LOS handling
    los_clip_days: float = 120.0

    # Independent validation horizon (days)
    readmit_horizon_days: int = 90
    mortality_horizon_days: int = 365


# --------------------------------------------------------------------
# Cohort flow logger (feeds the participant-flow diagram, R1.4)
# --------------------------------------------------------------------
class CohortFlow:
    def __init__(self):
        self.rows = []

    def log(self, stage: str, patients: int, admissions: int, note: str = ""):
        self.rows.append(dict(stage=stage, patients=patients,
                              admissions=admissions, note=note))
        print(f"[flow] {stage:<45s} pts={patients:>7,} adms={admissions:>7,}  {note}")

    def to_csv(self, path):
        pd.DataFrame(self.rows).to_csv(path, index=False)


def _norm_icd(s: pd.Series) -> pd.Series:
    """Canonical ICD normalization: string, strip, uppercase, remove dots.
    Applied IDENTICALLY to MIMIC codes and Orphanet codes (fixes the
    3,781-vs-3,778 discrepancy caused by divergent normalization)."""
    return (s.astype(str).str.strip().str.upper()
             .str.replace(".", "", regex=False))


# --------------------------------------------------------------------
# Step 1 — cohort construction
# --------------------------------------------------------------------
def build_cohort(cfg: PipelineConfig, flow: CohortFlow):
    B = cfg.mimic_base

    adm = pd.read_csv(f"{B}/hosp/admissions.csv",
                      usecols=["subject_id", "hadm_id", "admittime",
                               "dischtime", "hospital_expire_flag"],
                      parse_dates=["admittime", "dischtime"])
    flow.log("Raw admissions", adm.subject_id.nunique(), len(adm))

    adm = adm.dropna(subset=["admittime", "dischtime"])
    adm = adm[adm.dischtime > adm.admittime].copy()
    adm["los_days"] = ((adm.dischtime - adm.admittime)
                       .dt.total_seconds() / 86400.0).clip(0.01, cfg.los_clip_days)
    flow.log("Valid admit/discharge times", adm.subject_id.nunique(), len(adm))

    dx = pd.read_csv(f"{B}/hosp/diagnoses_icd.csv",
                     usecols=["subject_id", "hadm_id", "icd_code", "icd_version"])
    dx["icd_norm"] = _norm_icd(dx.icd_code)

    # --- Orphanet matching: ICD-10 rows ONLY (R1.4: no cross-version
    # string collisions between ICD-9 and ICD-10 code spaces) ---
    orph = pd.read_csv(cfg.orphanet_csv)
    orph["icd_norm"] = _norm_icd(orph["ICDcodes"])
    rare_set = set(orph.icd_norm.unique())
    print(f"[orphanet] {len(rare_set)} unique normalized ICD-10 codes")

    dx10 = dx[dx.icd_version == 10]
    if cfg.icd_match == "prefix":
        # MIMIC ICD-10-CM codes are specific (e.g. C649); match if the code
        # starts with any Orphanet code. Assign the LONGEST matching Orphanet
        # code so each dx maps to at most one mapping row.
        prefixes = sorted(rare_set, key=len, reverse=True)
        import re
        pat = re.compile("^(" + "|".join(map(re.escape, prefixes)) + ")")
        m = dx10.icd_norm.str.extract(pat, expand=False)
        rare_dx = dx10[m.notna()].copy()
        rare_dx["icd_match_code"] = m[m.notna()]
    else:
        rare_dx = dx10[dx10.icd_norm.isin(rare_set)].copy()
        rare_dx["icd_match_code"] = rare_dx.icd_norm
    flow.log(f"Admissions w/ >=1 Orphanet ICD-10 dx ({cfg.icd_match})",
             rare_dx.subject_id.nunique(), rare_dx.hadm_id.nunique(),
             "ICD-10 rows only; dots stripped both sides")

    # report Orphanet codes with zero matches (R1.4 transparency)
    matched = set(rare_dx.icd_match_code.unique())
    zero = sorted(rare_set - matched)
    print(f"[orphanet] codes with ZERO MIMIC matches ({len(zero)}): {zero}")

    rare_hadm = set(rare_dx.hadm_id.unique())
    cohort_adm = adm[adm.hadm_id.isin(rare_hadm)].copy()
    flow.log("Cohort after admission-quality filters",
             cohort_adm.subject_id.nunique(), len(cohort_adm))

    # Disease frequency table (R1.4) — joined on the matched Orphanet code
    freq = (rare_dx.merge(orph[["icd_norm", "PreferredTerm", "ORPHAcode"]]
                          .drop_duplicates("icd_norm"),
                          left_on="icd_match_code", right_on="icd_norm",
                          how="left")
            .groupby(["icd_match_code", "PreferredTerm", "ORPHAcode"])
            .agg(n_patients=("subject_id", "nunique"),
                 n_admissions=("hadm_id", "nunique"))
            .reset_index().sort_values("n_patients", ascending=False))

    # Patients with multiple distinct rare diseases (reviewer asked)
    multi = (rare_dx.groupby("subject_id").icd_norm.nunique() > 1).sum()
    print(f"[cohort] patients with >1 distinct rare-disease code: {multi}")

    # ICU flag per admission (TARGET/descriptor only — never a feature)
    icu = pd.read_csv(f"{B}/icu/icustays.csv",
                      usecols=["subject_id", "hadm_id"]).drop_duplicates()
    cohort_adm["in_icu"] = cohort_adm.hadm_id.isin(set(icu.hadm_id)).astype(int)

    # Date of death for independent mortality validation (R1.10)
    pats = pd.read_csv(f"{B}/hosp/patients.csv",
                       usecols=["subject_id", "dod"], parse_dates=["dod"])
    cohort_adm = cohort_adm.merge(pats, on="subject_id", how="left")

    return cohort_adm.sort_values(["subject_id", "admittime"]).reset_index(drop=True), freq


# --------------------------------------------------------------------
# Step 2 — patient-level split (R1.5)
# --------------------------------------------------------------------
def make_splits(cohort_adm: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    subjects = np.sort(cohort_adm.subject_id.unique())
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(len(subjects))
    n_tr = int(cfg.train_frac * len(subjects))
    n_va = int(cfg.val_frac * len(subjects))
    part = np.full(len(subjects), "test", dtype=object)
    part[perm[:n_tr]] = "train"
    part[perm[n_tr:n_tr + n_va]] = "val"
    splits = pd.DataFrame({"subject_id": subjects, "partition": part})
    print("[split]", splits.partition.value_counts().to_dict())
    return splits


# --------------------------------------------------------------------
# Step 3 — leakage-free feature construction
#   Vocabularies, imputation medians, scaler: TRAIN ONLY.
#   Predictors: dx / lab / med / proc recorded during the admission —
#   all available at that admission's discharge (see feature_table.csv).
#   EXCLUDED as predictors: LOS, ICU flag/hours, mortality, transfers,
#   note counts/spans (R1.2).
# --------------------------------------------------------------------
def build_features(cohort_adm, splits, cfg: PipelineConfig, flow: CohortFlow):
    B = cfg.mimic_base
    cohort_hadm = set(cohort_adm.hadm_id)
    train_subj = set(splits.loc[splits.partition == "train", "subject_id"])

    dx = pd.read_csv(f"{B}/hosp/diagnoses_icd.csv",
                     usecols=["subject_id", "hadm_id", "icd_code"])
    dx = dx[dx.hadm_id.isin(cohort_hadm)].copy()
    dx["icd_norm"] = _norm_icd(dx.icd_code)

    procs = pd.read_csv(f"{B}/hosp/procedures_icd.csv",
                        usecols=["subject_id", "hadm_id", "icd_code"])
    procs = procs[procs.hadm_id.isin(cohort_hadm)].copy()
    procs["icd_norm"] = _norm_icd(procs.icd_code)

    meds = pd.concat([c[c.hadm_id.isin(cohort_hadm)] for c in
                      pd.read_csv(f"{B}/hosp/prescriptions.csv",
                                  usecols=["subject_id", "hadm_id", "drug"],
                                  chunksize=1_000_000)], ignore_index=True)
    meds["drug"] = meds.drug.astype(str).str.lower().str.strip()

    labs = pd.concat([c[c.hadm_id.isin(cohort_hadm)] for c in
                      pd.read_csv(f"{B}/hosp/labevents.csv",
                                  usecols=["subject_id", "hadm_id",
                                           "itemid", "valuenum"],
                                  chunksize=1_000_000)], ignore_index=True)
    labs = labs.dropna(subset=["hadm_id", "valuenum"])
    labs = labs[labs.valuenum.between(-1e4, 1e6)]
    gc.collect()

    # ---- vocabularies from TRAIN patients only (R1.2 / R1.4) ----
    def top_k(df, col, k, subj_col="subject_id"):
        return (df[df[subj_col].isin(train_subj)][col]
                .value_counts().index[:k].tolist())

    vocab = dict(
        dx=top_k(dx, "icd_norm", cfg.k_dx),
        lab=top_k(labs, "itemid", cfg.k_lab),
        med=top_k(meds, "drug", cfg.k_med),
        proc=top_k(procs, "icd_norm", cfg.k_proc),
    )
    print({k: len(v) for k, v in vocab.items()})

    def pivot_counts(df, col, keep, prefix):
        d = df[df[col].isin(keep)]
        p = (d.groupby(["subject_id", "hadm_id", col]).size()
             .rename("v").reset_index()
             .pivot_table(index=["subject_id", "hadm_id"],
                          columns=col, values="v", fill_value=0)
             .reset_index())
        cols = [f"{prefix}_{c}" for c in p.columns[2:]]
        p.columns = ["subject_id", "hadm_id"] + cols
        # guarantee full vocab columns, stable order
        for c0 in keep:
            cn = f"{prefix}_{c0}"
            if cn not in p.columns:
                p[cn] = 0
        return p[["subject_id", "hadm_id"] + [f"{prefix}_{c}" for c in keep]]

    dx_p = pivot_counts(dx, "icd_norm", vocab["dx"], "dx")
    med_p = pivot_counts(meds, "drug", vocab["med"], "med")
    proc_p = pivot_counts(procs, "icd_norm", vocab["proc"], "proc")

    lab_a = (labs[labs.itemid.isin(vocab["lab"])]
             .groupby(["subject_id", "hadm_id", "itemid"])
             .valuenum.mean().rename("v").reset_index()
             .pivot_table(index=["subject_id", "hadm_id"],
                          columns="itemid", values="v")
             .reset_index())
    lab_cols = [f"lab_{i}" for i in vocab["lab"]]
    lab_a.columns = ["subject_id", "hadm_id"] + [f"lab_{c}" for c in lab_a.columns[2:]]
    for c in lab_cols:
        if c not in lab_a.columns:
            lab_a[c] = np.nan
    lab_a = lab_a[["subject_id", "hadm_id"] + lab_cols]

    core = (cohort_adm.merge(dx_p, on=["subject_id", "hadm_id"], how="left")
                       .merge(lab_a, on=["subject_id", "hadm_id"], how="left")
                       .merge(med_p, on=["subject_id", "hadm_id"], how="left")
                       .merge(proc_p, on=["subject_id", "hadm_id"], how="left"))

    feat_cols = ([f"dx_{c}" for c in vocab["dx"]] + lab_cols +
                 [f"med_{c}" for c in vocab["med"]] +
                 [f"proc_{c}" for c in vocab["proc"]])
    count_cols = [c for c in feat_cols if not c.startswith("lab_")]
    core[count_cols] = core[count_cols].fillna(0.0)

    # ---- imputation medians + scaler: TRAIN ONLY (R1.2) ----
    tr_mask = core.subject_id.isin(train_subj)
    lab_medians = core.loc[tr_mask, lab_cols].median()
    core[lab_cols] = core[lab_cols].fillna(lab_medians)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(
        core.loc[tr_mask, feat_cols].to_numpy(np.float32))
    core[feat_cols] = scaler.transform(core[feat_cols].to_numpy(np.float32))

    # ---- feature-timing table (R1.2 / R1.4) ----
    def _rows(cols, modality, meaning, timing):
        return [dict(feature=c, modality=modality, clinical_meaning=meaning,
                     encoding="z-scored count" if not c.startswith("lab_")
                              else "z-scored mean value (train-median imputed)",
                     available_at="discharge of the same admission",
                     used_as="predictor", timing_note=timing) for c in cols]

    ft = []
    ft += _rows([f"dx_{c}" for c in vocab["dx"]], "diagnosis",
                "ICD-10 code count in admission", "coded at discharge")
    ft += _rows(lab_cols, "laboratory",
                "mean lab value during admission", "measured during stay")
    ft += _rows([f"med_{c}" for c in vocab["med"]], "medication",
                "prescription count during admission", "ordered during stay")
    ft += _rows([f"proc_{c}" for c in vocab["proc"]], "procedure",
                "ICD procedure count in admission", "coded at discharge")
    for nm, why in [("los_days", "outcome (LOS)"), ("in_icu", "outcome (ICU)"),
                    ("hospital_expire_flag", "outcome (mortality)")]:
        ft.append(dict(feature=nm, modality="outcome", clinical_meaning=why,
                       encoding="-", available_at="after discharge",
                       used_as="TARGET / descriptor only — NEVER a predictor",
                       timing_note="excluded from all model inputs"))
    feature_table = pd.DataFrame(ft)

    flow.log("Featurized admissions", core.subject_id.nunique(), len(core))
    return core, feat_cols, vocab, scaler, lab_medians, feature_table


# --------------------------------------------------------------------
# Step 4 — tensor exports for the two tasks
# --------------------------------------------------------------------
def build_tensors(core, feat_cols, splits, cfg: PipelineConfig):
    core = core.sort_values(["subject_id", "admittime"]).reset_index(drop=True)
    part_map = dict(zip(splits.subject_id, splits.partition))
    D = len(feat_cols)

    # ---------- Task A: STRATIFICATION (full trajectory) ----------
    subj = core.subject_id.unique()
    N = len(subj)
    Xs = np.zeros((N, cfg.t_max, D), np.float32)
    Ms = np.zeros((N, cfg.t_max), np.float32)
    meta = []
    for i, (sid, g) in enumerate(core.groupby("subject_id", sort=False)):
        Xi = g[feat_cols].to_numpy(np.float32)[:cfg.t_max]
        Xs[i, :len(Xi)] = Xi
        Ms[i, :len(Xi)] = 1.0
        last_disch = g.dischtime.max()
        dod = g.dod.iloc[0]
        meta.append(dict(
            subject_id=sid, partition=part_map[sid],
            n_admissions=len(g),
            # descriptors for post-hoc cluster profiling ONLY:
            icu_any=int(g.in_icu.max()),
            total_los_days=float(g.los_days.sum()),
            died_in_hosp=int(g.hospital_expire_flag.max()),
            # independent validation outcomes (never in any loss, R1.10):
            death_within_365d=int(pd.notna(dod) and
                                  (dod - last_disch).days <= cfg.mortality_horizon_days
                                  and (dod - last_disch).days >= 0),
        ))
    strat_meta = pd.DataFrame(meta)

    # ---------- Task B: PROSPECTIVE (adm 1..n-1 -> adm n) ----------
    rowsX, rowsM, y_icu, y_los, pmeta = [], [], [], [], []
    for sid, g in core.groupby("subject_id", sort=False):
        if len(g) < 2:
            continue
        hist, target = g.iloc[:-1], g.iloc[-1]
        Xi = hist[feat_cols].to_numpy(np.float32)[-cfg.t_max:]
        Xp = np.zeros((cfg.t_max, D), np.float32)
        Mp = np.zeros(cfg.t_max, np.float32)
        Xp[:len(Xi)] = Xi
        Mp[:len(Xi)] = 1.0
        rowsX.append(Xp); rowsM.append(Mp)
        y_icu.append(float(target.in_icu))
        y_los.append(float(np.log1p(target.los_days)))
        gap = (target.admittime - hist.dischtime.iloc[-1]).days
        pmeta.append(dict(subject_id=sid, partition=part_map[sid],
                          n_history=len(hist),
                          readmit_within_90d=int(0 <= gap <= cfg.readmit_horizon_days),
                          target_los_days=float(target.los_days),
                          prediction_time="discharge of admission n-1",
                          gap_days=int(gap)))
    Xp = np.stack(rowsX) if rowsX else np.zeros((0, cfg.t_max, D), np.float32)
    Mp = np.stack(rowsM) if rowsM else np.zeros((0, cfg.t_max), np.float32)
    prosp_meta = pd.DataFrame(pmeta)
    print(f"[tensors] stratification N={N} | prospective N={len(prosp_meta)} "
          f"({(prosp_meta.partition=='test').sum() if len(prosp_meta) else 0} test)")
    return (Xs, Ms, strat_meta), (Xp, Mp,
            np.array(y_icu, np.float32), np.array(y_los, np.float32), prosp_meta)


# --------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------
def run_pipeline(cfg: PipelineConfig):
    os.makedirs(cfg.out_dir, exist_ok=True)
    flow = CohortFlow()

    cohort_adm, disease_freq = build_cohort(cfg, flow)
    splits = make_splits(cohort_adm, cfg)
    core, feat_cols, vocab, scaler, lab_medians, feature_table = \
        build_features(cohort_adm, splits, cfg, flow)
    (Xs, Ms, strat_meta), (Xp, Mp, yI, yL, prosp_meta) = \
        build_tensors(core, feat_cols, splits, cfg)

    # ---- persist everything (R1.19 reproducibility) ----
    O = cfg.out_dir
    flow.to_csv(f"{O}/cohort_flow.csv")
    disease_freq.to_csv(f"{O}/disease_frequency.csv", index=False)
    feature_table.to_csv(f"{O}/feature_table.csv", index=False)
    splits.to_csv(f"{O}/splits.csv", index=False)
    np.save(f"{O}/strat_X.npy", Xs); np.save(f"{O}/strat_M.npy", Ms)
    strat_meta.to_csv(f"{O}/strat_meta.csv", index=False)
    np.save(f"{O}/prosp_X.npy", Xp); np.save(f"{O}/prosp_M.npy", Mp)
    np.save(f"{O}/prosp_y_icu.npy", yI); np.save(f"{O}/prosp_y_los.npy", yL)
    prosp_meta.to_csv(f"{O}/prosp_meta.csv", index=False)
    import joblib
    joblib.dump(scaler, f"{O}/scaler.joblib")
    with open(f"{O}/vocab.json", "w") as f:
        json.dump({k: [str(x) for x in v] for k, v in vocab.items()}, f)
    with open(f"{O}/config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    # data fingerprint so every result can be traced to one pipeline run
    h = hashlib.sha256(Xs.tobytes()).hexdigest()[:16]
    print(f"[done] pipeline fingerprint: {h}  -> report this in the paper")
    return dict(strat=(Xs, Ms, strat_meta),
                prosp=(Xp, Mp, yI, yL, prosp_meta),
                feat_cols=feat_cols, splits=splits, fingerprint=h)


if __name__ == "__main__":
    run_pipeline(PipelineConfig())
