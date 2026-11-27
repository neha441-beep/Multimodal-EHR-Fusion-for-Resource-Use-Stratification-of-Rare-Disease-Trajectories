# Multimodal VaDeSC-EHR: Resource-Use Stratification of Patients with Orphanet-Associated ICD-10 Codes

Code for the paper *"Multimodal EHR Fusion for Resource-Use Stratification of
Patients with Orphanet-Associated ICD-10 Codes"* (MIMIC-IV v2.1).

The framework extends VaDeSC-EHR into a multimodal variational transformer that
fuses four structured EHR modalities (diagnoses, laboratory values,
medications, procedures) to learn unsupervised patient-trajectory
representations. The primary task is resource-use stratification (clustering
patients into stable strata that differ in resource use); the secondary task
is leakage-free prospective next-admission prediction of ICU use and length of
stay.

The cohort consists of MIMIC-IV patients carrying ICD-10 codes that the
Orphanet cross-reference associates with rare diseases. It is not a verified
rare-disease registry: several of the most frequent codes (for example G20,
C61, G35, I81, N10) describe conditions that are common in the general
population. A sensitivity analysis excluding them is included.

## Data access

MIMIC-IV v2.1 is a credentialed-access resource and **cannot be
redistributed**. Obtain it yourself from PhysioNet under the Data Use
Agreement, after completing the required CITI training:
<https://physionet.org/content/mimiciv/2.1/>.

**This repository contains no MIMIC data**, no derived patient-level tensors,
no predictions, and no split identifiers. Only code, the Orphanet ICD-10
mapping, and aggregate reporting tables are included.

## Environment

- Python 3.12 (the environment used in the paper: Kaggle image, one NVIDIA T4)
- PyTorch, scikit-learn, SciPy, pandas, NumPy, matplotlib, umap-learn

```bash
pip install torch scikit-learn scipy pandas numpy matplotlib umap-learn
```

The sequence encoders are implemented directly in PyTorch (transformer, GRU,
LSTM, temporal convolution, and a lightweight gated residual block). The gated
residual comparator has no state-space mechanism and is not a reproduction of
EHRMamba.

## Reproduce

```bash
export MIMIC_BASE=/path/to/mimic-iv-2.1     # folder containing hosp/ and icu/
python run_all.py
```

`run_all.py` runs, in order: cohort construction (exact and prefix match) ->
cohort reporting -> base harness (BIC provenance) -> primary stratification
(K = 3) -> sensitivity analysis excluding common codes -> history-only
prospective tensors -> prospective prediction -> figures.

**Provenance check.** The cohort-construction step prints a deterministic
fingerprint. For the primary exact-match cohort it must read
`a2146d49229e22b1`; the prefix (sensitivity) cohort reads `2112bea1fa23e6d8`.
A different value means your MIMIC-IV copy differs from the one used in the
paper.

GPU training is not bit-for-bit deterministic. Reruns reproduce the reported
values to the third decimal; small differences in the fourth decimal can occur.

## Repository contents

| File | Purpose | Paper |
|------|---------|-------|
| `run_all.py` | End-to-end runner. Set `MIMIC_BASE` and run this. | all |
| `rare_ehr_pipeline.py` | Cohort construction (MIMIC-IV to Orphanet ICD-10 matching), features, patient-level split, fingerprint. | Sec 4.1-4.2 |
| `cohort_flow.py` | Sequential participant flow, full one-to-many Orphanet mapping, code frequencies, feature dictionary. | Fig 2, Supplement S1a |
| `stratification_deep.py` | Primary stratification: modality ablation, stability, cross-algorithm agreement, outcome separation, strata profiles, test-set projections. | Tables 2-3, Figs 4-6 |
| `stratification_sensitivity.py` | Stratification repeated after excluding the most frequent common codes; agreement with the main strata. | Supplement S4 |
| `prospective_enriched.py` | Builds the leakage-free prospective tensors (pre-cutoff predictors only; use `--no_target_gap`). | Sec 4.2, Supplement S1 |
| `run_prediction_enriched.py` | Next-admission prediction across encoders and baselines; validation-selected thresholds, seed-averaged bootstrap CIs, Brier skill score, Friedman/Holm tests. | Table 4, Supplement S3 |
| `run_experiments.py` | Earlier base harness; provenance for the exploratory BIC sweep only. | Supplement S2 |
| `bic_exploratory.py` | BIC / stability sweep over K. | Supplement S2 |
| `make_figures.py` | Figs 2-6 (file names `fig2_...`, `fig6_disease_frequency` = paper Fig 3, `fig3_modality_ablation` = Fig 4, `fig4_cluster_profile` = Fig 5, `fig9_umap_tsne` = Fig 6). | Figs 2-6 |
| `make_figures_extra.py` | Calibration curve (paper Fig 7). | Fig 7 |
| `make_fig3_by_icd10.py` | Stand-alone version of paper Fig 3 from the cohort-reporting CSVs. | Fig 3 |
| `figures/fig1_architecture.pdf` | Architecture diagram (static). | Fig 1 |
| `verify_physionet.py` | Optional provenance check against a credentialed MIMIC-IV copy. | |
| `icd10_diseases.csv` | Orphanet ICD-10 cross-reference used to define the cohort. | Sec 4.1 |
| `orphanet_mapping_full.csv` | Full one-to-many mapping (ICD-10 code to Orphanet terms). | Fig 3 |
| `disease_frequency_final.csv` | Aggregate patient counts per matched code, with ICD-10-CM description and number of Orphanet terms. | Fig 3 |
| `excluded_variables.csv` | Variables excluded from all model inputs. | Supplement S1 |

## Notes

- Paths (`MIMIC_BASE`, `ORPH_CSV`) are set via environment variables or the
  `CONFIG` block in `run_all.py`. No paths are hardcoded to any account.
- Output folders (`pipeline_out*/`, `prospective_enriched*/`, `results*/`,
  `cohort_reporting/`, `figures/*.png`) contain or derive from patient-level
  data and are excluded by `.gitignore`. Do not commit them.

## Citation

If you use this code, please cite the paper (details to be added on
publication).
