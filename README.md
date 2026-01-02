# CTG-classification
This paper addresses the challenge of unbalanced data in the classification of fetal signals from the CTG dataset, a critical issue that can lead to misclassification of pathological cases and improper medical care. Although previous studies have identified this problem, limited research has focused on effectively mitigating it. 


# Cardiotocography (CTG) Data Processing Pipeline

## Source Code & Data
- **Source Code**: [https://github.com/williamsdoug/CTG_RP/tree/master](https://github.com/williamsdoug/CTG_RP/tree/master)
- **Source Data**: [CTU-UHB CTG Database on PhysioNet](https://www.physionet.org/content/ctu-uhb-ctgdb/1.0.0/)

---

## Pipeline Steps

### 1. Step One: Data Denoising & Indexing
Run `denoise_display.py` to identify valid data indices and generate output files (e.g., `validlist.npy`).

**Resulting Data Subsets:**
- `norm-hyposis` → 84 cases *(final list)*
- `hypoxiaAsuspi_id` → 30 cases *(final list)*
- `norm` → 39 cases *(reference)*
- `hypoxia` → 14 cases *(reference)*

---

### 2. Step Two: Dataset Generation
Use `data_generate.py` to create the dataset (`x`, `y`).

Alternatively, directly use `ph_generate.py` for data generation.

**Note:** `denoise_display.py` also generates visual figures for inspection.

---

### 3. Step Three: Classification
Run `myRFoptm_compareClassifier.py` to perform classification.

---

### 4. Feature Extraction
- Some features are derived from MATLAB code. Refer to `name_abbre (1).csv` for details.
- EMG feature extraction was performed using the [**EMG Feature Extraction Toolbox**](https://github.com/JingweiToo/EMG-Feature-Extraction-Toolbox#jx-emgt--electromyography--emg--feature-extraction-toolbox).

---

## Useful Resources & References
- [Perinatology: Intrapartum Fetal Monitoring](https://perinatology.com/Fetal%20Monitoring/Intrapartum%20Monitoring.htm)
- [Banner Health: Blue Fetal Monitoring Flyer](https://www.bannerhealth.com/-/media/files/project/bh/careers/bluefetalmonitoringflyer2.ashx)
- [pyHRV: Heart Rate Variability Analysis Toolkit](https://github.com/PGomes92/pyhrv)
- [Machine Learning Mastery: Threshold Moving for Imbalanced Classification](https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/)
- [FHRMA: Fetal Heart Rate Morphological Analysis](https://github.com/utsb-fmm/FHRMA)

---

## Repository Structure (Original Source)
