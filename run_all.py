#!/usr/bin/env python
"""
run_all.py
=====================================================================
End-to-end reproduction of the pipeline for
"Multimodal EHR Fusion for Resource-Use Stratification of Patients with
Orphanet-Associated ICD-10 Codes" (MIMIC-IV v2.1).

This is the portable version of the run notebook (bda-run.ipynb): the
same steps, in the same order, with the same arguments, but with the
Kaggle-specific paths and %%writefile / !cp / %cd cells removed. The
accompanying modules are imported/called from the repository instead of
being staged from a private dataset.

Expected pipeline fingerprint for the PRIMARY exact-match cohort:
    a2146d49229e22b1
printed by the cohort-construction step below. If your run prints a
different value, your MIMIC-IV copy differs from the one used in the paper.

-------------------------------------------------------------------
DATA ACCESS
-------------------------------------------------------------------
MIMIC-IV v2.1 is a credentialed-access resource and cannot be
redistributed. Obtain it yourself from PhysioNet under the Data Use
Agreement: https://physionet.org/content/mimiciv/2.1/ . You must complete
the required CITI training and sign the DUA before downloading.

-------------------------------------------------------------------
USAGE
-------------------------------------------------------------------
1. Set MIMIC_BASE to your local, credentialed MIMIC-IV v2.1 directory
   (the folder containing the hosp/ and icu/ subfolders), either by
   editing the CONFIG block below or via the MIMIC_BASE environment
   variable. ORPH_CSV defaults to the Orphanet mapping shipped in this
   repo (icd10_diseases.csv).
2. Make sure these modules are present in the same directory:
       rare_ehr_pipeline.py
       run_experiments.py
       stratification_deep.py
       prospective_enriched.py
       run_prediction_enriched.py
       stratification_sensitivity.py
       cohort_flow.py
       make_figures.py
       make_figures_extra.py
3. Run:  python run_all.py

Requires: Python 3.12, torch, scikit-learn, scipy, pandas, numpy,
matplotlib, umap-learn. A CUDA GPU is recommended (the paper used a single T4).
=====================================================================
"""
import os
import subprocess
import sys

# =========================== CONFIG ================================
# Point MIMIC_BASE at YOUR credentialed MIMIC-IV v2.1 download.
MIMIC_BASE = os.environ.get("MIMIC_BASE", "/path/to/mimic-iv-2.1")
ORPH_CSV   = os.environ.get("ORPH_CSV",   "icd10_diseases.csv")
# ==================================================================


def sh(cmd):
    """Run a shell step and stop the whole pipeline if it fails."""
    print(f"\n$ {cmd}", flush=True)
    if subprocess.run(cmd, shell=True).returncode:
        sys.exit(f"[run_all] step failed: {cmd}")


def main():
    if not os.path.isdir(MIMIC_BASE):
        sys.exit(
            f"[run_all] MIMIC_BASE not found: {MIMIC_BASE}\n"
            "Set it to your credentialed MIMIC-IV v2.1 directory "
            "(the folder with hosp/ and icu/ inside), e.g.\n"
            "    MIMIC_BASE=/data/mimic-iv-2.1 python run_all.py")
    if not os.path.isfile(ORPH_CSV):
        sys.exit(f"[run_all] Orphanet mapping not found: {ORPH_CSV}")

    from rare_ehr_pipeline import PipelineConfig, run_pipeline

    # 1. Cohort construction. PRIMARY = exact ICD-10 match (fingerprint
    #    a2146d49229e22b1); PREFIX = sensitivity cohort (2112bea1fa23e6d8).
    run_pipeline(PipelineConfig(
        mimic_base=MIMIC_BASE, orphanet_csv=ORPH_CSV,
        out_dir="pipeline_out", icd_match="exact"))
    run_pipeline(PipelineConfig(
        mimic_base=MIMIC_BASE, orphanet_csv=ORPH_CSV,
        out_dir="pipeline_out_prefix", icd_match="prefix"))

    # 2. Cohort reporting: sequential flow, full Orphanet mapping, code
    #    frequencies, feature dictionary (Supplement S1a).
    sh(f"python cohort_flow.py --data pipeline_out --mimic {MIMIC_BASE} "
       f"--orph {ORPH_CSV} --out cohort_reporting")

    # 3. Earlier base harness: provenance for the exploratory BIC sweep only.
    sh("python run_experiments.py --data pipeline_out --out results "
       "--seeds 5 --epochs 100")

    # 4. Primary stratification, K fixed to 3 (Tables 2-3, Figs 4-6).
    sh("python stratification_deep.py --data pipeline_out "
       "--out results_strat_k3 --seeds 5 --fixed_k 3")

    # 5. Sensitivity analysis excluding the most frequent common codes
    #    (Supplement S4).
    sh(f"python stratification_sensitivity.py --data pipeline_out "
       f"--mimic {MIMIC_BASE} --orph {ORPH_CSV} "
       f"--out results_strat_sensitivity --seeds 5")

    # 6. Prospective prediction on history-only tensors (--no_target_gap),
    #    the primary configuration (Table 4, Supplement S3). The gated
    #    residual comparator and the KL-weight-0 model always run.
    sh(f"python prospective_enriched.py --data pipeline_out "
       f"--mimic {MIMIC_BASE} --orph {ORPH_CSV} "
       f"--out prospective_enriched_nogap --no_target_gap")
    sh("python run_prediction_enriched.py --data prospective_enriched_nogap "
       "--out results_pred_nogap --seeds 5 --epochs 120")

    # 7. Figures (Fig 1 is the static figures/fig1_architecture.pdf).
    sh("python make_figures.py")
    sh("python make_figures_extra.py")

    print("\n[run_all] done. Outputs in results*/ and figures/.")
    print("[run_all] The primary exact-match cohort fingerprint printed in "
          "step 1 should read a2146d49229e22b1.")


if __name__ == "__main__":
    main()
