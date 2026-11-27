"""Figure 3 (fig6_disease_frequency): matched ICD-10 codes by patient count.

Labels show the ICD-10 code, its ICD-10-CM description, and the number of
Orphanet terms mapped to that code, as requested in Round 2 review.

Usage:
  python make_fig3_by_icd10.py disease_frequency_final.csv orphanet_mapping_full.csv OUTDIR \
      [--d_icd /path/to/mimic-iv-2.1/hosp/d_icd_diagnoses.csv]

If --d_icd is given, descriptions are taken from MIMIC-IV's own dictionary
(icd_version 10, long_title); otherwise the built-in ICD-10-CM titles are used.
"""
import argparse, os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ICD10CM = {
    "G20": "Parkinson's disease",
    "C61": "Malignant neoplasm of prostate",
    "I81": "Portal vein thrombosis",
    "G35": "Multiple sclerosis",
    "N10": "Acute pyelonephritis",
    "C541": "Malignant neoplasm of endometrium",
    "D65": "Disseminated intravascular coagulation",
    "J80": "Acute respiratory distress syndrome",
    "C20": "Malignant neoplasm of rectum",
    "C73": "Malignant neoplasm of thyroid gland",
    "C548": "Malignant neoplasm of overlapping sites of corpus uteri",
    "C543": "Malignant neoplasm of fundus uteri",
    "C540": "Malignant neoplasm of isthmus uteri",
    "C542": "Malignant neoplasm of myometrium",
}

ap = argparse.ArgumentParser()
ap.add_argument("freq"); ap.add_argument("mapping"); ap.add_argument("out")
ap.add_argument("--d_icd", default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

freq = pd.read_csv(a.freq)
mapping = pd.read_csv(a.mapping)
n_terms = mapping.groupby("icd_norm")["ORPHAcode"].nunique()

desc = dict(ICD10CM)
if a.d_icd:
    d = pd.read_csv(a.d_icd)
    d = d[d.icd_version == 10]
    d["icd_norm"] = d.icd_code.str.replace(".", "", regex=False).str.upper().str.strip()
    desc.update(dict(zip(d.icd_norm, d.long_title)))

freq["icd10_description"] = freq.icd_norm.map(desc)
freq["n_orphanet_terms"] = freq.icd_norm.map(n_terms).astype(int)
missing = freq.loc[freq.icd10_description.isna(), "icd_norm"].tolist()
if missing:
    raise SystemExit(f"No ICD-10 description for: {missing}")

freq = freq.sort_values(["n_patients", "n_admissions", "icd_norm"],
                        ascending=[False, False, True]).reset_index(drop=True)
cols = ["icd_norm", "icd10_description", "n_orphanet_terms", "n_patients",
        "n_admissions", "ORPHAcode", "PreferredTerm", "source_row"]
freq[cols].to_csv(os.path.join(a.out, "disease_frequency_final.csv"), index=False)


def label(r):
    t = "term" if r.n_orphanet_terms == 1 else "terms"
    return f"{r.icd_norm}  {r.icd10_description}\n({r.n_orphanet_terms} Orphanet {t})"


plt.rcParams.update({"font.size": 8.5, "font.family": "sans-serif",
                     "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(8.2, 7.4))
y = range(len(freq))[::-1]
ax.barh(list(y), freq.n_patients, color="#3a8a6e", height=0.72)
ax.set_yticks(list(y))
ax.set_yticklabels([label(r) for r in freq.itertuples()], fontsize=7.6)
for yi, n in zip(y, freq.n_patients):
    ax.text(n + 8, yi, f"{n:,}", va="center", fontsize=7.6)
ax.set_xlabel("Patients")
ax.set_xlim(0, freq.n_patients.max() * 1.12)
ax.grid(axis="x", alpha=0.25, lw=0.5); ax.set_axisbelow(True)
ax.set_title("Most frequent matched ICD-10 codes in the cohort", fontsize=10)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(a.out, f"fig6_disease_frequency.{ext}"),
                dpi=300, bbox_inches="tight")
print(freq[["icd_norm", "icd10_description", "n_orphanet_terms", "n_patients"]].to_string(index=False))
