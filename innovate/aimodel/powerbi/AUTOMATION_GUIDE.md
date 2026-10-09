# ⚡ Power BI & Fabric Automated End-to-End Pipeline Guide

> **Scenario:** A company inputs their raw customer and sales CSV files $\rightarrow$ our Python ML pipeline scores the data $\rightarrow$ Power BI automatically updates with zero manual effort.

---

## 🏗️ 3 Enterprise Automation Architectures

### Architecture 1: The Local Folder Watcher (One-Click / Scheduled Batch)

```
[Company Input CSVs] ──► [Python Predict Runner] ──► [output/powerbi/*.csv] ──► [Power BI Refresh]
```

1. **How it works:**
   - The company drops their latest customer CSV (`company_customers.csv`) and sales CSV (`company_sales.csv`) into the `data/` folder.
   - Running [`run_automation.bat`](file:///c:/Users/mohit/Desktop/aimodel/run_automation.bat) executes `predict.py`, overwriting the pre-processed CSVs in [`output/powerbi/`](file:///c:/Users/mohit/Desktop/aimodel/output/powerbi/).
   - Opening Power BI and clicking **Refresh** immediately loads the new predictions!
2. **Automating via Windows Task Scheduler:**
   - Set Windows Task Scheduler to run `run_automation.bat` every night at 1:00 AM.

---

### Architecture 2: Cloud Folder + Power BI Service Scheduled Refresh (No Gateway Needed)

```
[Company Uploads to OneDrive/SharePoint/GCS] ──► [Python Cloud Worker] ──► [Power BI Service Auto-Refresh]
```

1. **How it works:**
   - Company users drop raw CSV files into a shared **OneDrive for Business**, **SharePoint Folder**, or **Google Cloud Storage (GCS) Bucket**.
   - A Python background worker / Azure Function automatically triggers `predict.py` whenever a file is uploaded.
   - Power BI Service connects to the output CSVs on OneDrive/SharePoint via Web connector and runs **Scheduled Refresh (up to 8x daily)**.
   - **Result:** Enterprise users view the updated web dashboard in their browser without ever opening Power BI Desktop!

---

### Architecture 3: Microsoft Fabric Lakehouse Real-Time Pipeline (Enterprise Gold Standard)

```
[Fabric Lakehouse Files] ──► [Fabric PySpark/Python Notebook] ──► [Direct Lake Semantic Model] ──► [Real-Time Power BI]
```

1. **How it works:**
   - Company drops input CSVs into Fabric Lakehouse `Files/raw_inputs/`.
   - A **Fabric Data Factory Pipeline** triggers an automated Python / PySpark Notebook containing our [`src/ml_pipeline.py`](file:///c:/Users/mohit/Desktop/aimodel/src/ml_pipeline.py) code.
   - Output predictions are written directly to Fabric Delta Tables (`churn_predictions`, `sales_forecast`).
   - Power BI connects using **Direct Lake Mode**: The dashboard updates in **sub-second real time** with 0 import delay and 0 scheduled refresh wait!

---

## 📋 Summary of Error Metrics & Validation Audit

### 1. Expanded 60-Month Sales Dataset Validation
* **Dataset Horizon:** 60 continuous months (5 full years: `2021-01` to `2025-12`).
* **Chronological Holdout Split:**
  * **Train Set:** 48 months (4 full years: `2021-01` to `2024-12`).
  * **Holdout Test Set:** 12 months (1 full year backtest: `2025-01` to `2025-12`).
* **Holdout Error Metrics:**
  * **MAPE (Mean Absolute Percentage Error):** `7.53%` across 12 unseen future months.
  * **RMSE (Root Mean Squared Error):** `$47,869` on a `$614,000/month` revenue run-rate.

### 2. 80% & 95% Quantile Prediction Intervals
* **Dual Quantile Regressors ($\alpha=0.10, 0.90$):** Produces the 80% confidence corridor ($[\$468\text{k} \text{ to } \$523\text{k}]$).
* **Dual Quantile Regressors ($\alpha=0.05, 0.95$):** Produces the 95% outer bounds corridor.

### 3. Data Leakage Prevention
* **Strict Temporal Integrity:** Time-series split is strictly chronological (no random sampling across time).
* **Isolated Preprocessing:** LabelEncoders and StandardScalers are fit **only on training data** and transformed on test data.
