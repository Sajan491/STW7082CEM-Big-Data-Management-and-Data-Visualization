# Project Proposal: Kickstarter Campaign Analytics

**Title:** Kickstarter Campaign Analytics: A Big-Data Analysis and Visualization of Crowdfunding Success Patterns Using PySpark Pipelines and Tableau BI

**Student:** Sajan Mahat  
**Student ID:** 250289  
**Module:** STW7082CEM  
**Course:** Big Data Management and Data Visualisation  

---

## Problem Description

Crowdfunding has revolutionized the way that entrepreneurs, artists, and innovators finance themselves without entering the mainstream of the financial system. However, most Kickstarter campaigns fail to reach their funding targets, and creators lack evidence-based guidance on what makes a triumphant campaign.

Using a massive dataset of more than 378,000 Kickstarter campaigns spanning April 2009 to January 2018, this project aims to:

- **Identify structural, categorical, and temporal patterns** that influence campaign success
- **Forecast the outcome** of a campaign
- **Estimate capitalization levels** needed for successful funding
- **Segment campaigns** into behavioral funding segments
- **Provide actionable intelligence** about the crowdfunding phenomena for innovators

---

## Dataset Description

### Basic Information

- **Name:** Kickstarter Projects
- **Direct Link:** https://www.kaggle.com/datasets/kemical/kickstarter-projects
- **File Used:** ks-projects-201801.csv
- **Dimensions:** 378,661 rows; 15 columns
- **Time Period:** April 2009 to January 2018

### Columns (15 total)

1. ID
2. name
3. category
4. main_category
5. currency
6. deadline
7. goal
8. launched
9. pledged
10. state
11. backers
12. country
13. usd_pledged
14. usd_pledged_real
15. usd_goal_real

### Data Types (satisfies 3+ requirement)

- **Integers:** backers
- **Floats/Doubles:** goal, pledged, usd_pledged_real, usd_goal_real
- **Strings/Categories:** name, category, main_category, currency, state, country
- **Datetime:** deadline, launched

### Target Variable

**State** (binary classification after cleaning):
- Successful / Failed
- (Live, canceled, and suspended records will be removed)

---

## Planned Analysis Tasks (PySpark MLlib)

All algorithmic tasks will be designed exclusively using PySpark. Preprocessing will include an end-to-end PySpark pipeline for building a feature vector by combining StringIndexer and OneHotEncoder for high cardinality categorical features.

### Feature Engineering

Six engineered features will be created:
1. **campaign_duration_days** - Duration from launch to deadline
2. **launch_month** - Month campaign was launched
3. **launch_day_of_week** - Day of week campaign was launched
4. **funding_ratio** - Ratio of pledged to goal
5. **goal_bucket** - Categorical bucketing of funding goals
6. **is_overfunded** - Binary indicator of overfunding

### 1. Classification

- **Baseline Model:** Logistic Regression
- **Advanced Model:** Random Forest Classifier
- **Objective:** Predict campaign outcomes (state)
- **Validation:** Cross-validation
- **Evaluation Metrics:**
  - Accuracy
  - F1-Score
  - AUC-ROC

### 2. Regression

- **Data Preprocessing:** Logarithmic transformation (log1p) applied to normalize target distribution
- **Baseline Model:** Linear Regression
- **Advanced Model:** Gradient-Boosted Tree (GBT) Regressors
- **Objective:** Forecast capital raised
- **Evaluation Metrics:**
  - RMSE (Root Mean Squared Error)
  - R² (Coefficient of Determination)

### 3. Clustering

- **Algorithm:** K-Means Clustering
- **Objective:** Unsupervised segmentation to group campaigns based on continuous numerical profiles
- **Use Case:** Behavioral funding segment identification

---

## Visualisation Plan (Tableau)

Following visualizations will be produced in Tableau after exporting PySpark outputs as CSV:

1. **World Map** - Campaign success rate by country
2. **Ranked Bar Chart** - Success rate by main_category
3. **Heatmap** - Launch month vs. success rate (reveals optimal campaign timing)
4. **Log-Scale Scatter Plot** - Goal vs. pledged, colored by campaign state
5. **Cluster Dashboard** - Multi-layered scatter matrix of goal vs. pledged with:
   - K-Means cluster as color dimension
   - Average backer count as bubble size
   - Direct cross-reference to PySpark model outputs
6. **Line Chart** - Campaigns launched per year (tracks Kickstarter growth 2009-2018)

---

## Brief Work Plan

| Phase | Dates (2026) | Activities |
|---|---|---|
| **Data Loading and Setup** | Jun 3-5 | Establish PySpark environment; load ks-projects-201801.csv; perform schema validation and structural sanity checks |
| **Preprocessing and Feature Engineering** | Jun 6-7 | Filter state; engineer six features; encode categoricals; build VectorAssembler pipeline |
| **Modelling** | Jun 8-11 | Build and evaluate all three models; document metrics and feature importances |
| **Visualisation** | Jun 12-13 | Build all Tableau dashboards |
| **Report and Submission** | Jun 14-16 | Conduct critical evaluation of findings; draft final technical report |

---

## Expected Outcomes

### Deliverables

1. **Comprehensive Analysis Report** (up to 3000 words)
   - Introduction
   - Implementation details
   - Findings and analysis
   - Critical evaluation
   - Conclusion and recommendations

2. **PySpark Code** (reproducible pipeline)
   - Data loading and preprocessing
   - Feature engineering
   - Model training and evaluation
   - Results export

3. **Tableau Dashboards**
   - 6 visualizations as planned
   - Exportable datasets

4. **Evidence Documentation**
   - Screenshots of environment setup
   - Screenshots of code execution
   - Screenshots of results

### Expected Insights

- Key factors influencing Kickstarter campaign success
- Optimal campaign characteristics and timing
- Predictive model performance for campaign outcome forecasting
- Behavioral segments within the crowdfunding ecosystem
- Geographic and category-based success patterns
- Capital requirement estimation models
