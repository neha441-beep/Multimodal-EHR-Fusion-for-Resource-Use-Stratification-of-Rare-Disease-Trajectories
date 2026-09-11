# Multimodal VaDeSC-EHR: Resource-Use Stratification of Rare-Disease Trajectories

Code for the paper *"Multimodal EHR Fusion for Resource-Use Stratification of
Rare-Disease Trajectories"* (MIMIC-IV v2.1).

The framework extends VaDeSC-EHR into a multimodal variational transformer that
fuses four structured EHR modalities (diagnoses, laboratory values,
medications, procedures) to learn unsupervised patient-trajectory
representations. The primary task is resource-use stratification (clustering
patients into stable, clinically coherent strata); the secondary task is
leakage-free prospective next-admission prediction of ICU use and length of
stay.

## Data access

MIMIC-IV v2.1 is a credentialed-access resource and **cannot be
redistributed**. Obtain it yourself from PhysioNet under the Data Use
Agreement, after completing the required CITI training:
<https://physionet.org/content/mimiciv/2.1/>.

**This repository contains no MIMIC data**, no derived patient-level tensors,
and no split identifiers. Only code and the Orphanet ICD-10 mapping are
included.

## Environment

- Python 3.11 (the environment used in the paper)
- PyTorch, scikit-learn, SciPy, pandas, NumPy, matplotlib
- A CUDA GPU is recommended (the paper used a single NVIDIA T4)

The sequence encoders are implemented directly in PyTorch
(`torch.nn.TransformerEncoder`, GRU, LSTM, a temporal-convolutional block, and
a lightweight Mamba-style block). No PyTorch Lightning or Hugging Face
Transformers dependency is required.

```bash
pip install torch scikit-learn scipy pandas numpy matplotlib
```

## Reproduce

1. Download MIMIC-IV v2.1 from PhysioNet (the folder containing `hosp/` and
   `icu/`).
2. Point the pipeline at your local copy and run everything end to end:

   ```bash
   export MIMIC_BASE=/path/to/mimic-iv-2.1
   python run_all.py
   ```

`run_all.py` runs, in order: cohort construction (exact and prefix match) ->
encoder/prediction experiments -> stratification -> enriched prospective
prediction -> figures.

**Provenance check.** The cohort-construction step prints a deterministic
fingerprint. For the primary exact-match cohort it must read:

```
a2146d49229e22b1
```

A different value means your MIMIC-IV copy differs from the one used in the
paper. The prefix (sensitivity) cohort fingerprint is `2112bea1fa23e6d8`.

## Repository contents

| File | Purpose |
|------|---------|
| `run_all.py` | End-to-end runner. Set `MIMIC_BASE` and run this. |
| `rare_ehr_pipeline.py` | Cohort construction: MIMIC-IV to Orphanet ICD-10 matching, feature/sequence building, patient-level splits, fingerprint. |
| `run_experiments.py` | Defines the `MultimodalVaDeSC` encoder and runs the encoder experiments. |
| `stratification_deep.py` | Primary contribution: modality ablation, cluster stability, cross-algorithm agreement, independent-outcome separation, per-cluster profiles. |
| `prospective_enriched.py` | Builds the leakage-free enriched prospective tensors (pre-cutoff predictors only). |
| `run_prediction_enriched.py` | Next-admission prediction harness across encoder families (transformer, GRU, LSTM, TCN) plus tabular baselines. |
| `make_figures.py` | Figures 1 to 6. |
| `make_figures_extra.py` | Figure 7 (calibration) and Figure 8 (t-SNE), as vector PDFs. |
| `make_bic_sweep.py` | Supplement: BIC / stability sweep over K. |
| `verify_physionet.py` | Optional provenance check: rebuilds the cohort from a credentialed MIMIC-IV copy and confirms the fingerprint. |
| `icd10_diseases.csv` | Orphanet rare-disease ICD-10 mapping. |
| `bda-run.ipynb` | The original run notebook (outputs cleared). `run_all.py` is the portable equivalent. |

## Notes

- Paths (`MIMIC_BASE`, `ORPH_CSV`) are set via the `CONFIG` block or environment
  variables in `run_all.py`. No paths are hardcoded to any account or platform.
- Given the same MIMIC-IV input, the preprocessing pipeline is deterministic;
  the fingerprint lets anyone confirm they reproduced the same cohort and
  configuration.

## Citation

If you use this code, please cite the paper (details to be added on
publication).
