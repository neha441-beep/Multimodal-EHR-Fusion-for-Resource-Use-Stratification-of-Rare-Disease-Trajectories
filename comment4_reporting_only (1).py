#!/usr/bin/env python3
"""Generate Comment 4 reporting artifacts without changing frozen model data.

This script is deliberately read-only with respect to ``pipeline_out``. It does
not rebuild tensors, vocabularies, splits, clusters, models, or predictions.

It produces:
  * cohort_flow_sequential.csv
  * disease_frequency_final.csv
  * orphanet_mapping_full.csv
  * feature_dictionary_complete.csv
  * excluded_variables.csv
  * comment4_report.json
  * comment4_summary.txt

Example:
  python comment4_reporting_only.py \
    --data pipeline_out \
    --mimic /kaggle/input/.../mimic-iv-2.1 \
    --orph /kaggle/input/.../icd10_diseases.csv \
    --out comment4_reporting
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_ARCHIVE_URL = (
    "https://web.archive.org/web/20250918214727/"
    "https://www.orpha.net/en/disease"
)


def table_path(base: Path, relative: str) -> Path:
    """Resolve a MIMIC CSV stored either uncompressed or gzip-compressed."""
    p = base / relative
    if p.exists():
        return p
    gz = Path(str(p) + ".gz")
    if gz.exists():
        return gz
    raise FileNotFoundError(f"Missing MIMIC table: {p} or {gz}")


def norm_icd(values: pd.Series) -> pd.Series:
    """Normalize ICD strings while preserving null values."""
    out = (
        values.astype("string")
        .str.upper()
        .str.replace(".", "", regex=False)
        .str.strip()
    )
    return out.mask(out.eq(""))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def find_column(frame: pd.DataFrame, candidates: Iterable[str], purpose: str) -> str:
    by_lower = {str(c).lower(): c for c in frame.columns}
    for candidate in candidates:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    raise KeyError(
        f"Could not find {purpose}. Expected one of {list(candidates)}; "
        f"found {list(frame.columns)}"
    )


def scalar(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def load_code_dictionary(path: Path, code_col: str, title_col: str) -> dict[str, str]:
    frame = pd.read_csv(path, usecols=lambda c: c in {code_col, title_col, "icd_version"})
    frame["code_norm"] = norm_icd(frame[code_col])
    frame = frame.dropna(subset=["code_norm", title_col]).copy()
    if "icd_version" in frame.columns:
        frame["description"] = (
            "ICD-" + frame["icd_version"].astype("Int64").astype("string")
            + ": " + frame[title_col].astype("string")
        )
    else:
        frame["description"] = frame[title_col].astype("string")
    return (
        frame.groupby("code_norm", sort=False)["description"]
        .agg(lambda x: " | ".join(dict.fromkeys(x.astype(str))))
        .to_dict()
    )


def observed_lab_units(
    labevents_path: Path,
    final_hadm: set[int],
    lab_items: set[int],
) -> dict[int, str]:
    units: dict[int, set[str]] = {item: set() for item in lab_items}
    usecols = ["hadm_id", "itemid", "valueuom"]
    for chunk in pd.read_csv(labevents_path, usecols=usecols, chunksize=1_000_000):
        chunk = chunk[
            chunk["hadm_id"].isin(final_hadm) & chunk["itemid"].isin(lab_items)
        ]
        for item, values in chunk.groupby("itemid")["valueuom"]:
            units[int(item)].update(
                str(v).strip() for v in values.dropna().unique() if str(v).strip()
            )
    return {
        item: "; ".join(sorted(values)) if values else "Unit not recorded"
        for item, values in units.items()
    }


def add_sequence_features(
    vocab: dict,
    dx_desc: dict[str, str],
    proc_desc: dict[str, str],
    lab_meta: dict[int, dict],
    lab_units: dict[int, str],
) -> list[dict]:
    rows: list[dict] = []

    for code in vocab.get("dx", []):
        code = str(code)
        rows.append({
            "feature": f"dx_{code}",
            "original_code_or_name": code,
            "clinical_meaning": dx_desc.get(
                code, "Diagnosis code; description unavailable in MIMIC dictionary"
            ),
            "unit_or_encoding": "Admission-level count, then train-fitted z-score",
            "modality": "diagnosis",
            "missingness_handling": "Absent code represented as zero before standardization",
            "preprocessing": "Top 128 by frequency in training patients; train-fitted StandardScaler",
            "availability_stratification": "Available by discharge of each included admission",
            "availability_prospective": "Admissions 1 through n-1 only; available at prediction cutoff",
            "status": "model input",
        })

    for raw_item in vocab.get("lab", []):
        item = int(raw_item)
        meta = lab_meta.get(item, {})
        label = meta.get("label", "Laboratory item; label unavailable")
        context = ", ".join(
            str(meta[k]) for k in ("fluid", "category") if meta.get(k)
        )
        meaning = f"{label} ({context})" if context else label
        rows.append({
            "feature": f"lab_{item}",
            "original_code_or_name": str(item),
            "clinical_meaning": meaning,
            "unit_or_encoding": (
                f"Observed unit(s): {lab_units.get(item, 'Unit not recorded')}; "
                "admission-level mean, then train-fitted z-score"
            ),
            "modality": "laboratory",
            "missingness_handling": "Training-partition median imputation before standardization",
            "preprocessing": "Top 32 by frequency in training patients; train-median imputation; train-fitted StandardScaler",
            "availability_stratification": "Available by discharge of each included admission",
            "availability_prospective": "Admissions 1 through n-1 only; available at prediction cutoff",
            "status": "model input",
        })

    for drug in vocab.get("med", []):
        drug = str(drug)
        rows.append({
            "feature": f"med_{drug}",
            "original_code_or_name": drug,
            "clinical_meaning": f"Prescription drug name: {drug}",
            "unit_or_encoding": "Admission-level prescription count, then train-fitted z-score",
            "modality": "medication",
            "missingness_handling": "Absent medication represented as zero before standardization",
            "preprocessing": "Lowercased and stripped; top 64 by frequency in training patients; train-fitted StandardScaler",
            "availability_stratification": "Available by discharge of each included admission",
            "availability_prospective": "Admissions 1 through n-1 only; available at prediction cutoff",
            "status": "model input",
        })

    for code in vocab.get("proc", []):
        code = str(code)
        rows.append({
            "feature": f"proc_{code}",
            "original_code_or_name": code,
            "clinical_meaning": proc_desc.get(
                code, "Procedure code; description unavailable in MIMIC dictionary"
            ),
            "unit_or_encoding": "Admission-level count, then train-fitted z-score",
            "modality": "procedure",
            "missingness_handling": "Absent procedure represented as zero before standardization",
            "preprocessing": "Top 64 by frequency in training patients; train-fitted StandardScaler",
            "availability_stratification": "Available by discharge of each included admission",
            "availability_prospective": "Admissions 1 through n-1 only; available at prediction cutoff",
            "status": "model input",
        })

    return rows


def patient_level_features() -> list[dict]:
    specs = [
        ("age", "MIMIC-IV anchor age", "continuous", "static demographic"),
        ("sex", "Patient sex", "binary encoding", "static demographic"),
        ("prior_icu_rate", "Fraction of admissions 1 through n-1 with ICU use", "continuous proportion", "history through discharge of admission n-1"),
        ("prior_icu_any", "Any ICU use in admissions 1 through n-1", "binary indicator", "history through discharge of admission n-1"),
        ("prior_los_mean", "Mean LOS over admissions 1 through n-1", "continuous days before standardization", "history through discharge of admission n-1"),
        ("prior_los_max", "Maximum LOS over admissions 1 through n-1", "continuous days before standardization", "history through discharge of admission n-1"),
        ("prior_mort_any", "Any in-hospital mortality flag in admissions 1 through n-1", "binary indicator; constant zero in eligible cohort", "history through discharge of admission n-1"),
        ("n_prior", "Number of admissions before target admission n", "integer count", "history through discharge of admission n-1"),
        ("span_days", "Time span across the prior-admission history", "continuous days", "history through discharge of admission n-1"),
        ("comorbidity", "Number of distinct prior diagnosis codes", "integer count", "history through discharge of admission n-1"),
        ("adm_rate", "Prior admission cadence", "continuous rate", "history through discharge of admission n-1"),
    ]
    rows = []
    for name, meaning, encoding, timing in specs:
        note = (
            "Retained as a pre-cutoff column but constant at zero and therefore carries no predictive signal"
            if name == "prior_mort_any" else "Used in secondary prospective model"
        )
        rows.append({
            "feature": name,
            "original_code_or_name": name,
            "clinical_meaning": meaning,
            "unit_or_encoding": encoding + "; standardized using training partition",
            "modality": "patient-level pre-cutoff predictor",
            "missingness_handling": "Constructed value; no missing value supplied to model",
            "preprocessing": "Calculated before target admission; train-fitted standardization",
            "availability_stratification": "Not used in primary stratification encoder",
            "availability_prospective": timing,
            "status": note,
        })
    rows.append({
        "feature": "gap_days",
        "original_code_or_name": "gap_days",
        "clinical_meaning": "Interval from discharge of admission n-1 to admission n",
        "unit_or_encoding": "Days; fixed to zero in primary prospective analysis",
        "modality": "disabled target-row placeholder",
        "missingness_handling": "Not imputed; deliberately fixed to zero",
        "preprocessing": "Disabled by --no_target_gap",
        "availability_stratification": "Not used",
        "availability_prospective": "References admission n and is not assumed knowable at cutoff",
        "status": "disabled; carries no target-admission information",
    })
    return rows


def excluded_variables() -> pd.DataFrame:
    rows = [
        ("ICU length-of-stay hours", "resource proxy", "Removed from every model input"),
        ("Same-admission transfer events", "resource proxy", "Removed from every model input"),
        ("Clinical-note counts and spans", "text-derived proxy", "No clinical-notes pathway"),
        ("Total LOS of admission n", "prediction target", "Target only; never an input"),
        ("ICU status of admission n", "prediction target", "Target only; never an input"),
        ("In-hospital mortality of admission n", "target-admission outcome", "Never an input"),
        ("One-year post-discharge mortality", "cluster-validation outcome", "Never an encoder input or training target"),
    ]
    return pd.DataFrame(rows, columns=["variable", "type", "status"])


def main(args: argparse.Namespace) -> None:
    data = Path(args.data)
    mimic = Path(args.mimic)
    orph_path = Path(args.orph)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    required = [data / "splits.csv", data / "cohort_flow.csv", data / "vocab.json"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing frozen pipeline artifacts: {missing}")

    # Exact mapping file and implemented, order-dependent display-label rule.
    orph = pd.read_csv(orph_path)
    code_col = find_column(orph, ["ICDcodes", "icd_code", "ICD10"], "Orphanet ICD-10 column")
    orpha_col = find_column(orph, ["ORPHAcode", "orpha_code"], "ORPHAcode column")
    term_col = find_column(orph, ["PreferredTerm", "preferred_term", "term"], "preferred-term column")
    orph = orph[orph[code_col].notna()].copy()
    orph["source_row"] = orph.index
    orph["icd_norm"] = norm_icd(orph[code_col])
    orph = orph[orph["icd_norm"].notna()].copy()
    full_map = (
        orph[["source_row", "icd_norm", orpha_col, term_col]]
        .rename(columns={orpha_col: "ORPHAcode", term_col: "PreferredTerm"})
        .drop_duplicates()
    )
    full_map.to_csv(out / "orphanet_mapping_full.csv", index=False)
    representative = full_map.drop_duplicates("icd_norm", keep="first").copy()
    rare_set = set(representative["icd_norm"])

    admissions_path = table_path(mimic, "hosp/admissions.csv")
    diagnoses_path = table_path(mimic, "hosp/diagnoses_icd.csv")
    adm = pd.read_csv(
        admissions_path,
        usecols=["subject_id", "hadm_id", "admittime", "dischtime"],
        parse_dates=["admittime", "dischtime"],
    )
    dx = pd.read_csv(
        diagnoses_path,
        usecols=["subject_id", "hadm_id", "icd_code", "icd_version"],
    )

    raw_patients, raw_admissions = adm["subject_id"].nunique(), adm["hadm_id"].nunique()
    valid = adm[
        adm["admittime"].notna()
        & adm["dischtime"].notna()
        & (adm["dischtime"] > adm["admittime"])
    ].copy()
    valid_patients, valid_admissions = valid["subject_id"].nunique(), valid["hadm_id"].nunique()

    dx10 = dx[
        dx["icd_version"].eq(10) & dx["hadm_id"].isin(set(valid["hadm_id"]))
    ].copy()
    dx10["icd_norm"] = norm_icd(dx10["icd_code"])
    rare_final = dx10[dx10["icd_norm"].isin(rare_set)].copy()
    final_pairs = rare_final[["subject_id", "hadm_id"]].drop_duplicates()
    final_subjects = set(final_pairs["subject_id"])
    final_hadm = set(final_pairs["hadm_id"])
    final_patients = len(final_subjects)
    final_admissions = len(final_hadm)

    # Fail closed if this reporting reconstruction differs from frozen artifacts.
    splits = pd.read_csv(data / "splits.csv")
    pipeline_subjects = set(splits["subject_id"])
    if final_subjects != pipeline_subjects:
        only_rebuilt = sorted(final_subjects - pipeline_subjects)[:10]
        only_pipeline = sorted(pipeline_subjects - final_subjects)[:10]
        raise RuntimeError(
            "Final patient membership does not match frozen splits.csv. "
            f"Examples only in reconstruction={only_rebuilt}; "
            f"only in frozen pipeline={only_pipeline}"
        )

    old_flow = pd.read_csv(data / "cohort_flow.csv")
    expected_patients = int(old_flow.iloc[-1]["patients"])
    expected_admissions = int(old_flow.iloc[-1]["admissions"])
    if (final_patients, final_admissions) != (expected_patients, expected_admissions):
        raise RuntimeError(
            "Final reconstructed counts do not match frozen cohort_flow.csv: "
            f"reconstructed={(final_patients, final_admissions)}, "
            f"frozen={(expected_patients, expected_admissions)}"
        )

    flow = pd.DataFrame([
        {
            "stage": "Raw MIMIC-IV admissions",
            "patients": raw_patients,
            "admissions": raw_admissions,
            "patients_removed_from_previous": 0,
            "admissions_removed_from_previous": 0,
        },
        {
            "stage": "Admissions with valid admit/discharge timestamps",
            "patients": valid_patients,
            "admissions": valid_admissions,
            "patients_removed_from_previous": raw_patients - valid_patients,
            "admissions_removed_from_previous": raw_admissions - valid_admissions,
        },
        {
            "stage": "Timestamp-valid admissions with >=1 exact Orphanet-associated ICD-10 match",
            "patients": final_patients,
            "admissions": final_admissions,
            "patients_removed_from_previous": valid_patients - final_patients,
            "admissions_removed_from_previous": valid_admissions - final_admissions,
        },
        {
            "stage": "Final featurized cohort (no further exclusions)",
            "patients": final_patients,
            "admissions": final_admissions,
            "patients_removed_from_previous": 0,
            "admissions_removed_from_previous": 0,
        },
    ])
    flow.to_csv(out / "cohort_flow_sequential.csv", index=False)

    disease = rare_final.merge(
        representative[["icd_norm", "ORPHAcode", "PreferredTerm", "source_row"]],
        on="icd_norm",
        how="left",
    )
    disease_frequency = (
        disease.groupby(
            ["icd_norm", "ORPHAcode", "PreferredTerm", "source_row"],
            dropna=False,
        )
        .agg(n_patients=("subject_id", "nunique"), n_admissions=("hadm_id", "nunique"))
        .reset_index()
        .sort_values(["n_patients", "n_admissions", "icd_norm"], ascending=[False, False, True])
    )
    disease_frequency.to_csv(out / "disease_frequency_final.csv", index=False)
    distinct_codes = rare_final.groupby("subject_id")["icd_norm"].nunique()
    n_multiple_codes = int(distinct_codes.gt(1).sum())

    # Complete feature dictionary from the frozen vocabulary.
    with (data / "vocab.json").open() as handle:
        vocab = json.load(handle)
    dx_desc = load_code_dictionary(
        table_path(mimic, "hosp/d_icd_diagnoses.csv"), "icd_code", "long_title"
    )
    proc_desc = load_code_dictionary(
        table_path(mimic, "hosp/d_icd_procedures.csv"), "icd_code", "long_title"
    )
    lab_dictionary = pd.read_csv(
        table_path(mimic, "hosp/d_labitems.csv"),
        usecols=lambda c: c in {"itemid", "label", "fluid", "category"},
    )
    lab_meta = lab_dictionary.set_index("itemid").to_dict("index")
    lab_item_ids = {int(x) for x in vocab.get("lab", [])}
    lab_units = observed_lab_units(
        table_path(mimic, "hosp/labevents.csv"), final_hadm, lab_item_ids
    )
    feature_rows = add_sequence_features(vocab, dx_desc, proc_desc, lab_meta, lab_units)
    feature_rows.extend(patient_level_features())
    feature_dictionary = pd.DataFrame(feature_rows)
    feature_dictionary.to_csv(out / "feature_dictionary_complete.csv", index=False)
    excluded = excluded_variables()
    excluded.to_csv(out / "excluded_variables.csv", index=False)

    represented_codes = set(rare_final["icd_norm"].dropna())
    three_character_represented = sorted(c for c in represented_codes if len(c) == 3)
    mapping_checksum = sha256_file(orph_path)
    report = {
        "mode": "reporting-only; frozen tensors, splits, models, and predictions were not modified",
        "cohort_verified": True,
        "flow": flow.to_dict("records"),
        "final_patient_count": final_patients,
        "final_admission_count": final_admissions,
        "patients_with_more_than_one_distinct_matched_code": n_multiple_codes,
        "mapping": {
            "local_file": orph_path.name,
            "sha256": mapping_checksum,
            "orphanet_nomenclature_version": args.orphanet_version,
            "last_updated": args.orphanet_updated,
            "archived_page_url": args.archive_url,
            "access_date": args.access_date,
            "repository_commit_containing_mapping": args.repository_commit,
            "representative_rule": "first row in supplied mapping-file order after normalized-code deduplication; display only",
            "n_unique_normalized_codes": len(rare_set),
            "n_codes_with_multiple_mapping_rows": int(
                full_map.groupby("icd_norm").size().gt(1).sum()
            ),
        },
        "represented_exact_match_codes": sorted(represented_codes),
        "represented_three_character_exact_match_codes": three_character_represented,
        "feature_dictionary_rows": len(feature_dictionary),
        "excluded_variable_rows": len(excluded),
    }
    with (out / "comment4_report.json").open("w") as handle:
        json.dump(report, handle, indent=2, default=scalar)

    summary = f"""COMMENT 4 REPORTING-ONLY SUMMARY

Final cohort verified against frozen pipeline artifacts:
  Patients:   {final_patients:,}
  Admissions: {final_admissions:,}

Sequential participant flow:
  Raw:                    {raw_patients:,} patients / {raw_admissions:,} admissions
  Timestamp-valid:        {valid_patients:,} patients / {valid_admissions:,} admissions
  Exact-match final:      {final_patients:,} patients / {final_admissions:,} admissions
  Featurized:             {final_patients:,} patients / {final_admissions:,} admissions

Patients with >1 distinct matched normalized ICD-10 code in final cohort:
  {n_multiple_codes:,}

Mapping file:
  {orph_path.name}
  SHA-256: {mapping_checksum}
  Orphanet version: {args.orphanet_version}
  Last updated: {args.orphanet_updated}
  Access date: {args.access_date}
  Repository commit: {args.repository_commit} (repository provenance, not an Orphanet commit)

No tensors, splits, model weights, cluster assignments, or prediction outputs were changed.
"""
    (out / "comment4_summary.txt").write_text(summary)
    print(summary)
    print(f"Wrote reporting artifacts to: {out.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="pipeline_out", help="Frozen pipeline output directory")
    parser.add_argument("--mimic", required=True, help="MIMIC-IV v2.1 base directory")
    parser.add_argument("--orph", required=True, help="Exact icd10_diseases.csv used by pipeline")
    parser.add_argument("--out", default="comment4_reporting")
    parser.add_argument("--orphanet-version", default="1.3.0")
    parser.add_argument("--orphanet-updated", default="2025-06-24")
    parser.add_argument("--archive-url", default=DEFAULT_ARCHIVE_URL)
    parser.add_argument("--access-date", default="2025-09-27")
    parser.add_argument("--repository-commit", default="7b01868")
    main(parser.parse_args())
