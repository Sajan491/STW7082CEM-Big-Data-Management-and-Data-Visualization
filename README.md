# Kickstarter Campaign Analytics — PySpark + Tableau

An end-to-end big-data pipeline that analyses **378,661 Kickstarter campaigns**
(April 2009 – January 2018) to understand what drives crowdfunding success. All processing
and machine learning is done in **PySpark (Spark MLlib)**; pandas is used only for the final
result export, and the outputs are visualised in **Tableau**.

After cleaning (keeping only `successful` / `failed` campaigns), **331,675 campaigns** feed
three analytical tasks:

- **Classification** — predict whether a campaign succeeds (Logistic Regression baseline,
  **72.0% accuracy / 0.794 AUC**, vs. a cross-validated Random Forest → **85.1% accuracy,
  F1 0.852, 0.922 AUC**).
- **Regression** — predict how much a campaign raises, modelled on `log1p(pledged)` (Linear
  baseline, **R² 0.086**, vs. a tuned Gradient-Boosted Tree → **R² 0.224** on the log scale).
- **Clustering** — group campaigns into behavioural segments (K-Means, **k = 4**,
  silhouette **0.647**).

The pipeline is **leakage-aware**: outcome-derived columns (`pledged`, `usd_pledged_real`,
`funding_ratio`, `is_overfunded`) are excluded from the supervised models, the regression
vector also keeps `backers` out, and all preprocessing is fit on the **training split only**.
Everything uses a fixed seed of **42** for reproducibility.

## Dataset

[Kickstarter Projects](https://www.kaggle.com/datasets/kemical/kickstarter-projects)
(Kaggle) — `ks-projects-201801.csv`, **378,661 rows × 15 columns**.
A copy is already included in [Dataset/](Dataset/). To re-download:
`https://www.kaggle.com/api/v1/datasets/download/kemical/kickstarter-projects`

## Repository structure

```
kickstarter_code.py                      # main PySpark pipeline (10 phases, see below)
Kickstarter_Analysis.ipynb               # the same pipeline as a runnable notebook
requirements.txt                         # pinned Python dependencies
SETUP_WINDOWS.md                         # Windows setup (Java JDK + Hadoop winutils)
PROJECT_PROPOSAL.md                      # original project proposal
LICENSE
Dataset/
  └─ ks-projects-201801.csv              # input data (378,661 rows)
Output/                                  # exported result CSVs + dashboard_data.json (Tableau inputs)
figures/                                 # exported charts (fig_*.png) and evidence screenshots (ss_*.png)
BigDataAssignmentTableauVizualizations.twb   # Tableau workbook (dashboards)
```

### Pipeline phases (`kickstarter_code.py`)

1. Environment setup, data loading & validation (explicit schema, shape/null checks)
2. Cleaning & preprocessing (state filter → binary `label`, null/date handling, IQR outlier report)
3. Feature engineering (6 features: `campaign_duration_days`, `launch_month`,
   `launch_day_of_week`, `funding_ratio`, `goal_bucket`, `is_overfunded`)
4. ML preprocessing pipeline (StringIndexer + OneHotEncoder + StandardScaler + VectorAssembler)
5. Train/test split (80/20), pipeline fit on train only, clustering prep
6. Classification — Logistic Regression vs. cross-validated Random Forest
7. Regression — Linear Regression vs. cross-validated GBT (on `log1p(pledged)`)
8. Clustering — K-Means elbow + final model at k = 4 with silhouette
9. Results export for Tableau (CSVs)

### Output files (`Output/`)

| File | Contents |
|------|----------|
| `campaigns_enriched.csv` | Per-campaign table: raw dims + engineered features + cluster id |
| `classification_results.csv` | Per-test-campaign label, prediction, probability of success |
| `classification_metrics.csv` / `_long.csv` | Accuracy, F1, AUC, precision, recall per model |
| `classification_feature_importance.csv` | Random Forest feature importances |
| `confusion_matrix.csv` | Confusion matrix counts |
| `regression_results.csv` | Per-campaign actual vs. predicted pledged |
| `regression_metrics.csv` | RMSE / R² / MAE (log and $ scale) per model |
| `regression_feature_importance.csv` | GBT feature importances |
| `clustering_assignments.csv` | Per-campaign cluster id + key numeric features |
| `cluster_profiles.csv` | Per-cluster size, averages, success rate |
| `cluster_centers.csv` | K-Means centroids (scaled space) |
| `clustering_elbow.csv` | WCSS for k = 2…10 |
| `correlation_matrix.csv` | Numeric feature correlations |
| `dashboard_data.json` | Pre-aggregated KPIs/breakdowns for the dashboards |

## Requirements

- **Python 3.12** (also works on 3.9–3.11), **PySpark 3.5.4**, **Java JDK 8/11/17**
  (Java 17 recommended; Spark 3.5 does **not** support Java 21+)
- Python packages (pinned in `requirements.txt`): `pyspark`, `pandas`, `numpy`,
  `matplotlib`, `seaborn`, `jupyter`, `ipykernel`
  ```bash
  pip install -r requirements.txt
  ```
- **Windows only:** Hadoop native binaries (`winutils.exe` + `hadoop.dll`) — see
  [SETUP_WINDOWS.md](SETUP_WINDOWS.md) for the full Java + Hadoop setup.
- Tableau Desktop / Tableau Public (to open the workbook)

## How to run

1. **Get the data** — `Dataset/ks-projects-201801.csv` is already included. (Otherwise
   download it from the Kaggle link above into the `Dataset/` folder.)
2. **(Windows) set up the environment** — follow [SETUP_WINDOWS.md](SETUP_WINDOWS.md) to
   install the Java JDK and the `winutils.exe` / `hadoop.dll` binaries PySpark needs on Windows.
3. **Set the paths** — the scripts use absolute paths that currently point at
   `d:/Softwarica/Big Data/BigDataAssignment/...`. Update them to your local repo location:
   - In [kickstarter_code.py](kickstarter_code.py): `input_path` (≈ line 88) and
     `OUT_DIR` (≈ line 734).
   - In [Kickstarter_Analysis.ipynb](Kickstarter_Analysis.ipynb): the matching `input_path`
     and `OUT_DIR` cells.
4. **Run the pipeline**:
   ```bash
   python kickstarter_code.py
   ```
   This cleans the data, trains all models, and writes the result CSVs to `Output/`
   (takes a few minutes). Or open and run `Kickstarter_Analysis.ipynb` cell by cell for the
   same pipeline with inline output.
5. **Open `BigDataAssignmentTableauVizualizations.twb`** in Tableau. If prompted, point the
   data sources at the CSVs in `Output/` to view the interactive dashboards.

## Author

Sajan Mahat (Student ID 250289) — STW7082CEM Big Data Management and Data Visualisation,
Softwarica College / Coventry University.
