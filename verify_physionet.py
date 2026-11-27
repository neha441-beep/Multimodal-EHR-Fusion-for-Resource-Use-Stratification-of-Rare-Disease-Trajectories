r"""
verify_physionet.py
=====================================================================
Integrity gate for the MIMIC-IV data-provenance claim (reviewer item on
ethics / data availability).

The analysis was built on a Kaggle mirror of MIMIC-IV v2.1
(mangeshwagle/mimic-iv-2-1). MIMIC's DUA prohibits redistribution, so the
paper must be based on credentialed PhysioNet data. This script proves the
mirror is byte-equivalent to the credentialed download, at the level that
actually feeds the results, by two independent checks:

  CHECK 1 (row counts): for each of the seven MIMIC-IV tables the pipeline
          reads, compare the row count in the PhysioNet copy against the
          Kaggle mirror (when both are provided). Fast, human-auditable.

  CHECK 2 (fingerprint reproduction): rebuild the exact-match cohort from
          the PhysioNet copy with the SAME pipeline and confirm it emits
          the reported fingerprint a2146d49229e22b1. Because the
          fingerprint is sha256 over the built feature tensor, a match
          proves the two sources yield an identical analysis tensor.

CHECK 2 is decisive; CHECK 1 localises any mismatch. Only if CHECK 2
passes may the true credentialing statement be used (see
ethics_statement_TEMPLATE.tex). Do NOT write a credentialing claim before
this passes, and do NOT claim CITI/DUA completion that has not genuinely
happened.

This runs on a plain CPU box (pandas/numpy only; no torch/GPU). The
pipeline reads UNCOMPRESSED .csv; if your PhysioNet copy is .csv.gz,
gunzip the seven tables first (or point --physionet at a decompressed
tree with the standard hosp/ and icu/ layout).

Run (from the folder holding rare_ehr_pipeline.py):
  python verify_physionet.py \
      --physionet /path/to/physionet/mimic-iv-2.1 \
      --kaggle    /path/to/kaggle/mimic-iv-2.1 \
      --orph      /path/to/icd10_diseases.csv
  # --kaggle is optional; omit it to run CHECK 2 only.
=====================================================================
"""
from __future__ import annotations
import argparse, gzip, os, sys

EXPECTED_FINGERPRINT = "a2146d49229e22b1"          # primary exact-match cohort
TABLES = [                                          # exactly what the pipeline reads
    "hosp/admissions", "hosp/diagnoses_icd", "hosp/patients",
    "hosp/procedures_icd", "hosp/prescriptions", "hosp/labevents",
    "icu/icustays",
]


def find_table(root, rel):
    """Return the path to <root>/<rel>.csv or .csv.gz, whichever exists."""
    for ext in (".csv", ".csv.gz"):
        p = os.path.join(root, rel + ext)
        if os.path.exists(p):
            return p
    return None


def count_rows(path):
    """Count data rows (excluding the header). Streams; handles .gz."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        n = sum(1 for _ in f)
    return max(n - 1, 0)


def check_row_counts(physionet, kaggle):
    print("\n=== CHECK 1: row counts ===")
    all_ok = True
    for rel in TABLES:
        pp = find_table(physionet, rel)
        if pp is None:
            print(f"  {rel:<24s} MISSING in PhysioNet copy"); all_ok = False; continue
        pn = count_rows(pp)
        if kaggle:
            kp = find_table(kaggle, rel)
            if kp is None:
                print(f"  {rel:<24s} MISSING in Kaggle mirror"); all_ok = False; continue
            kn = count_rows(kp)
            match = (pn == kn)
            all_ok &= match
            print(f"  {rel:<24s} PhysioNet={pn:>12,}  Kaggle={kn:>12,}  "
                  f"{'MATCH' if match else 'DIFFER'}")
        else:
            print(f"  {rel:<24s} PhysioNet={pn:>12,}  (no Kaggle copy to compare)")
    if kaggle:
        print(f"  -> row counts {'all match' if all_ok else 'DIFFER — investigate'}")
    return all_ok if kaggle else None


def check_fingerprint(physionet, orph):
    print("\n=== CHECK 2: pipeline fingerprint reproduction ===")
    try:
        from rare_ehr_pipeline import PipelineConfig, run_pipeline
    except Exception as e:
        print(f"  ERROR: cannot import rare_ehr_pipeline ({e}). Run this from "
              f"the folder that contains it."); return False
    cfg = PipelineConfig(mimic_base=physionet, orphanet_csv=orph,
                         out_dir="pipeline_out_physionet", icd_match="exact")
    out = run_pipeline(cfg)
    got = out["fingerprint"]
    ok = (got == EXPECTED_FINGERPRINT)
    print(f"  expected {EXPECTED_FINGERPRINT}\n  got      {got}\n  "
          f"-> {'MATCH' if ok else 'MISMATCH'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--physionet", required=True,
                    help="root of credentialed MIMIC-IV v2.1 (contains hosp/, icu/)")
    ap.add_argument("--kaggle", default=None,
                    help="root of the Kaggle mirror (optional; enables CHECK 1)")
    ap.add_argument("--orph", required=True, help="Orphanet ICD-10 csv path")
    a = ap.parse_args()

    rows_ok = check_row_counts(a.physionet, a.kaggle)
    fp_ok = check_fingerprint(a.physionet, a.orph)

    print("\n=== VERDICT ===")
    if fp_ok and (rows_ok is not False):
        print("  PASS. The credentialed PhysioNet data reproduces the reported\n"
              "  fingerprint. You MAY now use ethics_statement_TEMPLATE.tex,\n"
              "  filling in the genuine CITI/DUA details for the credentialed\n"
              "  author. Do not assert any credentialing step that did not occur.")
        sys.exit(0)
    else:
        print("  FAIL. Do NOT write a credentialing/ethics statement yet.\n"
              "  If CHECK 1 differed, a table differs between mirror and\n"
              "  PhysioNet; if CHECK 2 mismatched, the analysis tensor differs\n"
              "  and the results must be regenerated from PhysioNet data.")
        sys.exit(1)


if __name__ == "__main__":
    main()
