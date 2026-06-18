"""
Kickstarter Campaign Analytics - PySpark Pipeline
Module: STW7082CEM Big Data Management and Data Visualisation
Author: Sajan Mahat (Student ID 250289)
All analysis in PySpark MLlib; pandas only for the final Tableau export. 
"""


# PHASE 1: ENVIRONMENT SETUP, DATA LOADING & VALIDATION

# Phase 1.1 Imports and Random Seeds
import sys
import os
import random
import numpy as np
import pandas as pd

# PySpark core
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, count, isnan, lit,
    datediff, to_timestamp, month, dayofweek,
    log1p, expm1, sum as F_sum
)
from pyspark.sql.types import (
    StructType, StructField, IntegerType, DoubleType,
    StringType, TimestampType, LongType
)

# PySpark ML (used in later phases)
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
)
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.regression import LinearRegression, GBTRegressor
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator, MulticlassClassificationEvaluator,
    RegressionEvaluator, ClusteringEvaluator
)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Ensure PySpark uses the same Python interpreter as the notebook kernel
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

print(f"Python version : {sys.version.split()[0]}")
print(f"PySpark version: {pyspark.__version__}")
print(f"Random seed    : {SEED}")
print("All libraries imported successfully")


# Phase 1.2 SparkSession Initialisation -> local[*] uses all cores; memory/partitions tuned for single machine
spark = (
    SparkSession.builder
    .appName("KickstarterAnalytics")
    .master("local[*]")
    .config("spark.driver.memory", "6g")
    .config("spark.sql.shuffle.partitions", "64")
    .config("spark.driver.maxResultSize", "2g")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
    .getOrCreate()
)

# Reduce log verbosity (only warnings and errors)
spark.sparkContext.setLogLevel("WARN")

print("=" * 55)
print("SPARK ENVIRONMENT INITIALISED")
print("=" * 55)
print(f"PySpark Version : {spark.version}")
print(f"App Name        : {spark.sparkContext.appName}")
print(f"Master          : {spark.sparkContext.master}")
print(f"Parallelism     : {spark.sparkContext.defaultParallelism} cores")
print(f"Spark UI        : http://localhost:4040")
print("=" * 55)


# Phase 1.3 Load the Dataset
input_path = "d:/Softwarica/Big Data/BigDataAssignment/Dataset/ks-projects-201801.csv"

# Explicit schema -> correct numeric types (inferSchema mis-types quoted fields); dates parsed in Phase 2.3
schema = StructType([
    StructField("ID", LongType(), True),
    StructField("name", StringType(), True),
    StructField("category", StringType(), True),
    StructField("main_category", StringType(), True),
    StructField("currency", StringType(), True),
    StructField("deadline", StringType(), True),
    StructField("goal", DoubleType(), True),
    StructField("launched", StringType(), True),
    StructField("pledged", DoubleType(), True),
    StructField("state", StringType(), True),
    StructField("backers", IntegerType(), True),
    StructField("country", StringType(), True),
    StructField("usd pledged", DoubleType(), True),       # note: space in column name (dataset quirk)
    StructField("usd_pledged_real", DoubleType(), True),
    StructField("usd_goal_real", DoubleType(), True),
])

try:
    df = (
        spark.read
        .option("header", "true")        # first row holds column names
        .option("multiLine", "true")     # allow quoted fields that span lines
        .option("escape", "\"")          # handle embedded double quotes
        .schema(schema)                  # explicit schema -> correct numeric types
        .csv(input_path)
    )
    # Cache: the raw DataFrame is reused across several validation steps
    df.cache()
    n_rows = df.count()   # triggers the read + populates the cache
    n_cols = len(df.columns)
    print(f"Data loaded successfully from:\n  {input_path}")
    print(f"Rows    : {n_rows:,}")
    print(f"Columns : {n_cols}")
except Exception as e:
    print(f"ERROR loading data from {input_path}")
    print(f"Reason: {e}")
    raise


# Phase 1.4 Validate Shape (Row & Column Counts)
EXPECTED_ROWS = 378661
EXPECTED_COLS = 15

assert n_rows == EXPECTED_ROWS, f"Row count mismatch: got {n_rows:,}, expected {EXPECTED_ROWS:,}"
assert n_cols == EXPECTED_COLS, f"Column count mismatch: got {n_cols}, expected {EXPECTED_COLS}"

print(f"Row count verified    : {n_rows:,} == {EXPECTED_ROWS:,}")
print(f"Column count verified : {n_cols} == {EXPECTED_COLS}")
print("\nColumn names:")
for i, c in enumerate(df.columns, start=1):
    print(f"  {i:2d}. {c}")


# Phase 1.5 Inspect Schema and Sample Rows
print("=== Schema (explicit) ===")
df.printSchema()

print("=== First 5 Rows ===")
df.show(5, truncate=True)


# Phase 1.6 Null Value Distribution -> count nulls for every column in one pass
null_counts_row = df.select([
    F_sum(col(c).isNull().cast("int")).alias(c) for c in df.columns
]).collect()[0]   # single-row result -> safe to collect

print("=== Null Counts per Column ===")
print(f"{'Column':<20} {'Nulls':>10} {'% of rows':>12}")
print("-" * 44)
for c in df.columns:
    nulls = null_counts_row[c]
    pct = (nulls / n_rows) * 100
    print(f"{c:<20} {nulls:>10,} {pct:>11.3f}%")


# Phase 1.7 Summary Statistics and Duplicate ID Check
print("=== Summary Statistics (key numerical columns) ===")
df.describe(["goal", "pledged", "backers", "usd_pledged_real", "usd_goal_real"]).show()

# State distribution (preview of the classification target)
print("=== Campaign State Distribution ===")
df.groupBy("state").count().orderBy(col("count").desc()).show()

# Duplicate ID check
distinct_ids = df.select("ID").distinct().count()
duplicates = n_rows - distinct_ids
print(f"Distinct IDs : {distinct_ids:,}")
print(f"Duplicate IDs: {duplicates:,}")
if duplicates == 0:
    print("No duplicate campaign IDs - each row is a unique campaign.")
else:
    print("WARNING: duplicate IDs present - investigate in Phase 2.")


# PHASE 2: DATA CLEANING & PREPROCESSING

# Phase 2.1 State Filtering and Binary Target -> keep successful/failed
rows_before = df.count()

# Keep only campaigns with a definitive outcome
df_filtered = df.filter(col("state").isin(["successful", "failed"]))
rows_after = df_filtered.count()
removed = rows_before - rows_after

print(f"Before state filter : {rows_before:,} rows")
print(f"After state filter  : {rows_after:,} rows")
print(f"Removed             : {removed:,} rows ({removed / rows_before * 100:.2f}%)")

# Binary target: label = 1 if successful, else 0 (failed)
df_filtered = df_filtered.withColumn(
    "label", when(col("state") == "successful", 1).otherwise(0)
)

# Class balance check
print("\n=== Class Distribution (classification target) ===")
label_dist = df_filtered.groupBy("label").count().orderBy("label")
label_dist.show()
for r in label_dist.collect():   # 2-row result -> safe to collect
    name = "successful (1)" if r["label"] == 1 else "failed (0)"
    print(f"  {name:<16}: {r['count']:>9,} ({r['count'] / rows_after * 100:.2f}%)")


# Phase 2.2 Null and Invalid Value Handling
critical_cols = ["ID", "goal", "launched", "pledged", "state", "backers"]

rows_before_null = df_filtered.count()

# Drop rows with nulls in any critical modelling column
df_clean = df_filtered.dropna(subset=critical_cols)
rows_after_null = df_clean.count()
dropped_null = rows_before_null - rows_after_null

print(f"Before null drop : {rows_before_null:,} rows")
print(f"After null drop  : {rows_after_null:,} rows")
print(f"Dropped (nulls)  : {dropped_null:,} rows ({dropped_null / rows_before_null * 100:.4f}%)")

# Fill non-critical categorical nulls with a domain-appropriate default
df_clean = df_clean.fillna({"name": "Unknown"})

# Verify no nulls remain in critical columns
remaining = df_clean.select([
    F_sum(col(c).isNull().cast("int")).alias(c) for c in critical_cols
]).collect()[0]
print("\nRemaining nulls in critical columns:")
for c in critical_cols:
    print(f"  {c:<10}: {remaining[c]}")


# Phase 2.3 Data Type Conversion -> parse deadline (yyyy-MM-dd) & launched (yyyy-MM-dd HH:mm:ss) to timestamps
df_clean = (
    df_clean
    .withColumn("deadline_ts", to_timestamp(col("deadline"), "yyyy-MM-dd"))
    .withColumn("launched_ts", to_timestamp(col("launched"), "yyyy-MM-dd HH:mm:ss"))
)

# Drop rows where either date failed to parse (invalid timeline)
rows_before_ts = df_clean.count()
df_clean = df_clean.dropna(subset=["deadline_ts", "launched_ts"])
rows_after_ts = df_clean.count()
dropped_ts = rows_before_ts - rows_after_ts

print(f"Rows with unparseable dates dropped: {dropped_ts:,}")
print(f"Rows after timestamp conversion    : {rows_after_ts:,}")

print("\n=== Converted timestamp columns ===")
df_clean.select("deadline", "deadline_ts", "launched", "launched_ts").show(5, truncate=False)
df_clean.select("deadline_ts", "launched_ts").printSchema()


# Phase 2.4 Outlier Detection (IQR rule) -> report only, outliers are KEPT
outlier_cols = ["goal", "pledged", "backers", "usd_pledged_real", "usd_goal_real"]
total = df_clean.count()

print(f"{'Column':<18} {'Q1':>14} {'Q3':>14} {'Outliers':>12} {'% of rows':>11}")
print("-" * 72)
for c in outlier_cols:
    # approxQuantile is far cheaper than exact quantiles on large data
    q1, q3 = df_clean.approxQuantile(c, [0.25, 0.75], 0.01)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = df_clean.filter((col(c) < lower) | (col(c) > upper)).count()
    print(f"{c:<18} {q1:>14,.1f} {q3:>14,.1f} {n_out:>12,} {n_out / total * 100:>10.2f}%")

print("\nDecision: outliers are KEPT (valid crowdfunding behaviour, not errors).")
print("Skew is addressed later via log-transform (pledged) and StandardScaler.")


# PHASE 3: FEATURE ENGINEERING

# Phase 3.1 Temporal Features -> campaign_duration_days, launch_month, launch_day_of_week
df_feat = (
    df_clean
    # Feature 1: campaign length in days = deadline - launched
    .withColumn("campaign_duration_days", datediff(col("deadline_ts"), col("launched_ts")))
    # Feature 2: launch month (1-12) to capture seasonality
    .withColumn("launch_month", month(col("launched_ts")))
    # Feature 3: launch day of week (1=Sunday ... 7=Saturday) for weekly patterns
    .withColumn("launch_day_of_week", dayofweek(col("launched_ts")))
)

print("Temporal features created: campaign_duration_days, launch_month, launch_day_of_week")
df_feat.select(
    "launched_ts", "deadline_ts",
    "campaign_duration_days", "launch_month", "launch_day_of_week"
).show(5, truncate=False)


# Phase 3.2 Numerical and Categorical Features -> funding_ratio, goal_bucket, is_overfunded
df_feat = (
    df_feat
    # Feature 4: funding_ratio = pledged / goal, guarded against divide-by-zero
    .withColumn(
        "funding_ratio",
        when(col("goal") > 0, col("pledged") / col("goal")).otherwise(0.0)
    )
    # Feature 5: goal_bucket = categorical size band of the funding goal
    .withColumn(
        "goal_bucket",
        when(col("goal") < 1000, "micro")
        .when(col("goal") < 10000, "small")
        .when(col("goal") < 50000, "medium")
        .when(col("goal") < 100000, "large")
        .otherwise("mega")
    )
    # Feature 6: is_overfunded = 1 if pledged exceeded goal, else 0
    .withColumn("is_overfunded", when(col("pledged") > col("goal"), 1).otherwise(0))
)

print("Numerical/categorical features created: funding_ratio, goal_bucket, is_overfunded")
df_feat.select("goal", "pledged", "funding_ratio", "goal_bucket", "is_overfunded").show(5)

print("=== goal_bucket distribution ===")
df_feat.groupBy("goal_bucket").count().orderBy(col("count").desc()).show()

print("=== is_overfunded distribution ===")
df_feat.groupBy("is_overfunded").count().orderBy("is_overfunded").show()


# Phase 3.3 Feature Validation
engineered = [
    "campaign_duration_days", "launch_month", "launch_day_of_week",
    "funding_ratio", "goal_bucket", "is_overfunded"
]

# 1) all engineered columns present
missing = [c for c in engineered if c not in df_feat.columns]
assert not missing, f"Missing engineered features: {missing}"
print("All 6 engineered features present.")
print(f"Total columns now: {len(df_feat.columns)} (15 original + 3 helpers [label, deadline_ts, launched_ts] + 6 engineered)")

# 2) null check on the engineered features
null_row = df_feat.select([
    F_sum(col(c).isNull().cast("int")).alias(c) for c in engineered
]).collect()[0]
print("\nNulls in engineered features:")
for c in engineered:
    print(f"  {c:<24}: {null_row[c]}")

# 3) value-range statistics for the numeric engineered features
print("\n=== Engineered numeric feature statistics ===")
df_feat.describe([
    "campaign_duration_days", "launch_month", "launch_day_of_week",
    "funding_ratio", "is_overfunded"
]).show()

# Cache the feature-complete DataFrame for the downstream phases
df_feat.cache()
print(f"df_feat cached. Row count: {df_feat.count():,}")


# PHASE 4: ML PREPROCESSING PIPELINE (ENCODING, SCALING, VECTOR ASSEMBLY)

# Phase 4.1 Feature Selection (Leakage-Safe) -> launch-time features shared by both tasks (no target leakage)
cont_features = ["goal", "usd_goal_real", "campaign_duration_days"]   # scaled
highcard_cat  = ["category", "currency", "country"]                   # StringIndex only
lowcard_cat   = ["main_category", "goal_bucket"]                      # StringIndex + OneHot
temporal_cat  = ["launch_month", "launch_day_of_week"]               # OneHot (already numeric)

supervised_inputs = cont_features + highcard_cat + lowcard_cat + temporal_cat
missing = [c for c in supervised_inputs if c not in df_feat.columns]
assert not missing, f"Missing feature columns: {missing}"

print("Launch-time feature groups (used by BOTH classification & regression):")
print(f"  Continuous (scaled)        : {cont_features}")
print(f"  High-card (index only)     : {highcard_cat}")
print(f"  Low-card  (index + onehot) : {lowcard_cat}")
print(f"  Temporal  (onehot)         : {temporal_cat}")
print("\nClassification ALSO uses 'backers' (engagement signal) -> vector 'features_clf'.")
print("Regression uses launch-time only -> vector 'features' (keeps backers->pledged out).")
print("Excluded everywhere (leakage): pledged, usd_pledged_real, funding_ratio, is_overfunded")


# Phase 4.2 String Indexing and One-Hot Encoding
index_cols = highcard_cat + lowcard_cat   # all string categoricals to index

# One StringIndexer per categorical column -> <col>_idx
indexers = [
    StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
    for c in index_cols
]

# OneHotEncode the low-cardinality indices + the (numeric) temporal columns
ohe_inputs  = [f"{c}_idx" for c in lowcard_cat] + temporal_cat
ohe_outputs = [f"{c}_ohe" for c in lowcard_cat] + [f"{c}_ohe" for c in temporal_cat]
encoder = OneHotEncoder(
    inputCols=ohe_inputs, outputCols=ohe_outputs,
    dropLast=True, handleInvalid="keep"
)

print(f"Indexers defined for : {index_cols}")
print(f"One-hot inputs       : {ohe_inputs}")
print(f"One-hot outputs      : {ohe_outputs}")


# Phase 4.3 Scaling and Vector Assembly
# 1) assemble continuous launch-time features, then standardise them
cont_assembler = VectorAssembler(inputCols=cont_features, outputCol="cont_vec", handleInvalid="keep")
scaler = StandardScaler(inputCol="cont_vec", outputCol="cont_scaled", withMean=True, withStd=True)

# 2) REGRESSION / base vector 'features' = scaled continuous + one-hot + high-card indices (launch-time only)
highcard_idx = [f"{c}_idx" for c in highcard_cat]
final_inputs = ["cont_scaled"] + ohe_outputs + highcard_idx
final_assembler = VectorAssembler(inputCols=final_inputs, outputCol="features", handleInvalid="keep")

# 3) CLASSIFICATION vector 'features_clf' = base 'features' + scaled backers (engagement; kept OUT of regression)
backers_assembler = VectorAssembler(inputCols=["backers"], outputCol="backers_vec", handleInvalid="keep")
backers_scaler = StandardScaler(inputCol="backers_vec", outputCol="backers_scaled", withMean=True, withStd=True)
clf_assembler = VectorAssembler(inputCols=["features", "backers_scaled"], outputCol="features_clf", handleInvalid="keep")

# 4) single pipeline (fit on TRAIN only in Phase 5)
preprocess_pipeline = Pipeline(
    stages=indexers + [encoder, cont_assembler, scaler, final_assembler,
                       backers_assembler, backers_scaler, clf_assembler]
)

print("Preprocessing pipeline defined with stages:")
for i, s in enumerate(preprocess_pipeline.getStages(), 1):
    print(f"  {i:2d}. {type(s).__name__}")
print(f"\\nRegression vector 'features' inputs  : {final_inputs}")
print("Classification vector 'features_clf' : features + backers_scaled")


# Phase 4.4 Targets and Modeling Base -> pledged_log = log1p(pledged) handles right-skew (log1p(0)=0)
df_model = df_feat.withColumn("pledged_log", log1p(col("pledged")))
df_model.cache()

print("Regression target 'pledged_log' = log1p(pledged) created.")
df_model.select("pledged", "pledged_log", "label").show(5)
print(f"df_model rows: {df_model.count():,}")


# PHASE 5: DATA PREPARATION FOR MODELLING (SPLIT, PIPELINE FIT, CLUSTERING PREP)

# Phase 5.1 Train-Test Split (80/20, seed=42)
train_df, test_df = df_model.randomSplit([0.8, 0.2], seed=42)
train_df.cache(); test_df.cache()

n_train, n_test = train_df.count(), test_df.count()
print(f"Train rows: {n_train:,} ({n_train/(n_train+n_test)*100:.1f}%)")
print(f"Test  rows: {n_test:,} ({n_test/(n_train+n_test)*100:.1f}%)")


# Phase 5.2 Fit Pipeline on Train Only, Transform Both (no leakage)
pipeline_model = preprocess_pipeline.fit(train_df)
train_prepared = pipeline_model.transform(train_df).cache()
test_prepared  = pipeline_model.transform(test_df).cache()

print(f"train_prepared rows: {train_prepared.count():,}")
print(f"test_prepared  rows: {test_prepared.count():,}")

def _names(vec_col):
    meta = train_prepared.schema[vec_col].metadata.get("ml_attr", {})
    n = meta.get("num_attrs", 0)
    names = [f"f{i}" for i in range(n)]
    for grp in meta.get("attrs", {}).values():
        for a in grp:
            names[a["idx"]] = a["name"]
    return names

feature_names = _names("features")           # regression vector
feature_names_clf = _names("features_clf")   # classification vector (+ backers)
print(f"\\nRegression vector 'features'     : {len(feature_names)} features")
print(f"Classification vector 'features_clf': {len(feature_names_clf)} features (adds backers)")


# Phase 5.3 Class Distribution Check
print("=== Train label distribution ===")
train_prepared.groupBy("label").count().orderBy("label").show()
print("=== Test label distribution ===")
test_prepared.groupBy("label").count().orderBy("label").show()

for name, d, n in [("Train", train_prepared, n_train), ("Test", test_prepared, n_test)]:
    succ = d.filter(col("label") == 1).count()
    print(f"{name}: {succ:,}/{n:,} successful = {succ/n*100:.2f}%")


# Phase 5.4 Clustering Dataset Prep (unsupervised, fit on all data)
clust_features = ["goal", "usd_pledged_real", "backers", "campaign_duration_days", "funding_ratio"]

clust_assembler = VectorAssembler(inputCols=clust_features, outputCol="clust_vec", handleInvalid="keep")
clust_scaler = StandardScaler(inputCol="clust_vec", outputCol="features", withMean=True, withStd=True)
clust_pipeline = Pipeline(stages=[clust_assembler, clust_scaler])

clustering_prepared = clust_pipeline.fit(df_model).transform(df_model).cache()
print(f"Clustering features: {clust_features}")
print(f"clustering_prepared rows: {clustering_prepared.count():,}")
clustering_prepared.select("features").show(3, truncate=False)


# PHASE 6: CLASSIFICATION - PREDICTING CAMPAIGN SUCCESS

# Phase 6.1 Logistic Regression (Baseline)
# Shared classification evaluation helper
acc_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
f1_eval  = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
auc_eval = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")

def evaluate_classifier(predictions, model_name):
    acc = acc_eval.evaluate(predictions)
    f1  = f1_eval.evaluate(predictions)
    auc = auc_eval.evaluate(predictions)
    tp = predictions.filter((col("label") == 1) & (col("prediction") == 1)).count()
    tn = predictions.filter((col("label") == 0) & (col("prediction") == 0)).count()
    fp = predictions.filter((col("label") == 0) & (col("prediction") == 1)).count()
    fn = predictions.filter((col("label") == 1) & (col("prediction") == 0)).count()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    print(f"--- {model_name} ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"AUC-ROC  : {auc:.4f}")
    print(f"Precision: {precision:.4f}   Recall: {recall:.4f}")
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(f"            pred=0     pred=1")
    print(f"  actual=0  {tn:>8,}  {fp:>8,}")
    print(f"  actual=1  {fn:>8,}  {tp:>8,}")
    return {"model": model_name, "accuracy": acc, "f1": f1, "auc": auc,
            "precision": precision, "recall": recall,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}

# Baseline: Logistic Regression (on features_clf = launch-time + backers)
lr = LogisticRegression(featuresCol="features_clf", labelCol="label", regParam=0.01, maxIter=100)
lr_model = lr.fit(train_prepared)
lr_pred = lr_model.transform(test_prepared)
lr_metrics = evaluate_classifier(lr_pred, "Logistic Regression")


# Phase 6.2 Random Forest (Advanced, 3-fold CV grid search on features_clf)
# Shallow depth keeps backers from overfitting (~90%); maxBins=256 handles category (~160 values)
rf = RandomForestClassifier(featuresCol="features_clf", labelCol="label", seed=42, maxBins=256)

rf_grid = (
    ParamGridBuilder()
    .addGrid(rf.numTrees, [30, 40, 50])
    .addGrid(rf.maxDepth, [2])   # depth 1 underfits (~60%), depth 3+ over-fits backers (~89%)
    .build()
)

rf_cv = CrossValidator(
    estimator=rf,
    estimatorParamMaps=rf_grid,
    evaluator=BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"),
    numFolds=3,
    seed=42,
    parallelism=2,
)

print("Fitting Random Forest cross-validator (3 combos x 3 folds = 9 fits)...")
rf_cv_model = rf_cv.fit(train_prepared)
rf_best = rf_cv_model.bestModel

print("\\nBest hyperparameters:")
print(f"  numTrees : {rf_best.getNumTrees}")
print(f"  maxDepth : {rf_best.getOrDefault('maxDepth')}")
print(f"  Best CV AUC (avg over folds): {max(rf_cv_model.avgMetrics):.4f}")


# Evaluate best Random Forest on test
rf_pred = rf_best.transform(test_prepared)
rf_metrics = evaluate_classifier(rf_pred, "Random Forest (tuned)")


# Phase 6.3 Feature Importance (Random Forest)
importances = rf_best.featureImportances.toArray()
fi_pairs = sorted(zip(feature_names_clf, importances), key=lambda x: x[1], reverse=True)

print("Top 15 features by importance:")
print(f"{'Feature':<40} {'Importance':>10}")
print("-" * 52)
for name, imp in fi_pairs[:15]:
    print(f"{name:<40} {imp:>10.4f}")

classification_feature_importance = spark.createDataFrame(
    [(n, float(i)) for n, i in fi_pairs], ["feature", "importance"]
)


# Phase 6.4 Model Comparison (Classification)
classification_metrics = [lr_metrics, rf_metrics]

print(f"{'Model':<26} {'Accuracy':>9} {'F1':>9} {'AUC':>9} {'Precision':>10} {'Recall':>9}")
print("-" * 76)
for m in classification_metrics:
    print(f"{m['model']:<26} {m['accuracy']:>9.4f} {m['f1']:>9.4f} {m['auc']:>9.4f} {m['precision']:>10.4f} {m['recall']:>9.4f}")

acc_gain = (rf_metrics['accuracy'] - lr_metrics['accuracy']) * 100
auc_gain = (rf_metrics['auc'] - lr_metrics['auc']) * 100
print(f"\nRandom Forest vs Logistic Regression: accuracy {acc_gain:+.2f} pts, AUC {auc_gain:+.2f} pts")


# PHASE 7: REGRESSION - PREDICTING AMOUNT PLEDGED

# Phase 7.1 Linear Regression (Baseline)
# Shared regression evaluation helper
rmse_log_eval = RegressionEvaluator(labelCol="pledged_log", predictionCol="prediction", metricName="rmse")
r2_log_eval   = RegressionEvaluator(labelCol="pledged_log", predictionCol="prediction", metricName="r2")
mae_log_eval  = RegressionEvaluator(labelCol="pledged_log", predictionCol="prediction", metricName="mae")

def evaluate_regressor(predictions, model_name):
    rmse_log = rmse_log_eval.evaluate(predictions)
    r2_log   = r2_log_eval.evaluate(predictions)
    mae_log  = mae_log_eval.evaluate(predictions)
    # back-transform predictions to original $ scale (expm1 inverts log1p)
    pred_orig = predictions.withColumn("pred_orig", expm1(col("prediction")))
    rmse_orig = RegressionEvaluator(labelCol="pledged", predictionCol="pred_orig", metricName="rmse").evaluate(pred_orig)
    mae_orig  = RegressionEvaluator(labelCol="pledged", predictionCol="pred_orig", metricName="mae").evaluate(pred_orig)
    print(f"--- {model_name} ---")
    print(f"  [log scale]  RMSE={rmse_log:.4f}   R2={r2_log:.4f}   MAE={mae_log:.4f}")
    print(f"  [orig $]     RMSE={rmse_orig:,.2f}   MAE={mae_orig:,.2f}")
    return {"model": model_name, "rmse_log": rmse_log, "r2_log": r2_log, "mae_log": mae_log,
            "rmse_orig": rmse_orig, "mae_orig": mae_orig}

# Baseline: Linear Regression
lin = LinearRegression(featuresCol="features", labelCol="pledged_log", regParam=0.01, maxIter=100)
lin_model = lin.fit(train_prepared)
lin_pred = lin_model.transform(test_prepared)
lin_metrics = evaluate_regressor(lin_pred, "Linear Regression")


# Phase 7.2 GBT Regressor (Advanced, 3-fold CV grid search)
gbt = GBTRegressor(featuresCol="features", labelCol="pledged_log", seed=42, maxBins=256)

gbt_grid = (
    ParamGridBuilder()
    .addGrid(gbt.maxIter, [20, 50])
    .addGrid(gbt.maxDepth, [3, 5])
    .build()
)

gbt_cv = CrossValidator(
    estimator=gbt,
    estimatorParamMaps=gbt_grid,
    evaluator=RegressionEvaluator(labelCol="pledged_log", predictionCol="prediction", metricName="rmse"),
    numFolds=3,
    seed=42,
    parallelism=2,
)

print("Fitting GBT cross-validator (4 combos x 3 folds = 12 fits)...")
gbt_cv_model = gbt_cv.fit(train_prepared)
gbt_best = gbt_cv_model.bestModel

print("\nBest hyperparameters:")
print(f"  maxIter  : {gbt_best.getOrDefault('maxIter')}")
print(f"  maxDepth : {gbt_best.getOrDefault('maxDepth')}")
print(f"  Best CV RMSE (avg over folds): {min(gbt_cv_model.avgMetrics):.4f}")


# Evaluate best GBT on test
gbt_pred = gbt_best.transform(test_prepared)
gbt_metrics = evaluate_regressor(gbt_pred, "GBT Regressor (tuned)")


# Phase 7.3 Feature Importance (GBT)
gbt_importances = gbt_best.featureImportances.toArray()
gbt_fi_pairs = sorted(zip(feature_names, gbt_importances), key=lambda x: x[1], reverse=True)

print("Top 15 features by importance:")
print(f"{'Feature':<40} {'Importance':>10}")
print("-" * 52)
for name, imp in gbt_fi_pairs[:15]:
    print(f"{name:<40} {imp:>10.4f}")

regression_feature_importance = spark.createDataFrame(
    [(n, float(i)) for n, i in gbt_fi_pairs], ["feature", "importance"]
)


# Phase 7.4 Model Comparison (Regression)
regression_metrics = [lin_metrics, gbt_metrics]

print(f"{'Model':<26} {'RMSE(log)':>10} {'R2(log)':>9} {'MAE(log)':>9} {'RMSE($)':>14} {'MAE($)':>14}")
print("-" * 86)
for m in regression_metrics:
    print(f"{m['model']:<26} {m['rmse_log']:>10.4f} {m['r2_log']:>9.4f} {m['mae_log']:>9.4f} {m['rmse_orig']:>14,.0f} {m['mae_orig']:>14,.0f}")

r2_gain = gbt_metrics['r2_log'] - lin_metrics['r2_log']
print(f"\nGBT vs Linear Regression: R2 (log) change {r2_gain:+.4f}")


# PHASE 8: CLUSTERING - CAMPAIGN SEGMENTATION (K-MEANS)

# Phase 8.1 Elbow Method -> WCSS for k = 2..10
wcss = []
for k in range(2, 11):
    km = KMeans(featuresCol="features", k=k, seed=42)
    km_model = km.fit(clustering_prepared)
    cost = km_model.summary.trainingCost   # within-set sum of squared errors
    wcss.append((k, cost))
    print(f"k={k:2d}  WCSS={cost:,.1f}")

# % reduction in WCSS as k increases (helps spot the elbow)
print("\nMarginal WCSS reduction:")
for i in range(1, len(wcss)):
    prev, curr = wcss[i-1][1], wcss[i][1]
    print(f"  k={wcss[i][0]:2d}: {(prev-curr)/prev*100:5.1f}% drop vs k={wcss[i-1][0]}")


# Phase 8.2 Final K-Means at Optimal k + Silhouette
OPTIMAL_K = 4   # chosen from the elbow above

kmeans = KMeans(featuresCol="features", k=OPTIMAL_K, seed=42)
kmeans_model = kmeans.fit(clustering_prepared)
clustered_full = kmeans_model.transform(clustering_prepared)   # has 'features' + 'prediction'

# silhouette needs the 'features' vector
silhouette = ClusteringEvaluator(featuresCol="features", predictionCol="prediction").evaluate(clustered_full)

# Lean cached copy (drop heavy vector cols) for profiles + export -> stable, no recompute
clustered = clustered_full.drop("clust_vec", "features").cache()
clustered.count()

print(f"Optimal k         : {OPTIMAL_K}")
print(f"Silhouette score  : {silhouette:.4f}")
print("\nCluster sizes:")
clustered.groupBy("prediction").count().orderBy("prediction").show()


# Phase 8.3 Cluster Profiles -> write enriched table via Spark JVM writer, then aggregate in pandas
import os, glob, shutil
from pyspark.sql.functions import year

OUT_DIR = "d:/Softwarica/Big Data/BigDataAssignment/Output"
os.makedirs(OUT_DIR, exist_ok=True)

def spark_write_single_csv(sdf, name):
    """Write one clean CSV via Spark's JVM writer (no driver collect / toPandas)."""
    tmp = os.path.join(OUT_DIR, "_tmp_" + name)
    sdf.coalesce(1).write.option("header", True).option("quote", '"').option("escape", '"').mode("overwrite").csv(tmp)
    part = glob.glob(os.path.join(tmp, "part-*.csv"))[0]
    dest = os.path.join(OUT_DIR, name)
    if os.path.exists(dest):
        os.remove(dest)
    shutil.move(part, dest)
    shutil.rmtree(tmp)
    return dest

# Per-campaign enriched table (~331,675 rows): raw dims + engineered + cluster id
campaigns_enriched_sdf = clustered.select(
    "ID", "category", "main_category", "currency", "country", "state", "label",
    "goal", "pledged", "usd_pledged_real", "usd_goal_real", "backers",
    year(col("launched_ts")).alias("launched_year"),
    "launch_month", "launch_day_of_week", "campaign_duration_days",
    "funding_ratio", "goal_bucket", "is_overfunded",
    col("prediction").alias("cluster"),
)
spark_write_single_csv(campaigns_enriched_sdf, "campaigns_enriched.csv")
print("campaigns_enriched.csv written (Spark JVM writer).")

# Read back (~36 MB) and compute cluster profiles in pandas (no Spark shuffle)
enriched_pd = pd.read_csv(os.path.join(OUT_DIR, "campaigns_enriched.csv"))
print(f"Loaded enriched: {enriched_pd.shape[0]:,} rows x {enriched_pd.shape[1]} cols")

cluster_profiles_pd = (enriched_pd.groupby("cluster").agg(
        size=("ID", "count"), avg_goal=("goal", "mean"),
        avg_pledged=("usd_pledged_real", "mean"), avg_backers=("backers", "mean"),
        avg_duration_days=("campaign_duration_days", "mean"),
        avg_funding_ratio=("funding_ratio", "mean"), success_rate=("label", "mean"))
    .round(3).reset_index())
print("\n=== Cluster Profiles ===")
print(cluster_profiles_pd.to_string(index=False))

# Cluster centers (standardized space) straight from the model -> pandas
cluster_centers_pd = pd.DataFrame(
    [[i] + [float(v) for v in c] for i, c in enumerate(kmeans_model.clusterCenters())],
    columns=["cluster"] + clust_features,
)
print("\n=== Cluster Centers (scaled space) ===")
print(cluster_centers_pd.round(3).to_string(index=False))


# PHASE 9: RESULTS EXPORT FOR TABLEAU

# Phase 9 Results Export for Tableau -> small data via pandas, large via Spark JVM writer
from pyspark.ml.functions import vector_to_array
print(f"Exporting all result CSVs to: {OUT_DIR}")


# Phase 9 Export Classification
print("Classification:")
# per-test-campaign predictions + probability of success (Spark JVM write, no toPandas)
clf_results = (rf_pred
    .select("ID", "label", "prediction", vector_to_array("probability").alias("_p"))
    .withColumn("prob_success", col("_p")[1]).drop("_p"))
spark_write_single_csv(clf_results, "classification_results.csv")
# metrics + feature importance straight from Python objects (no Spark)
pd.DataFrame(classification_metrics).to_csv(os.path.join(OUT_DIR, "classification_metrics.csv"), index=False)
pd.DataFrame(fi_pairs, columns=["feature", "importance"]).to_csv(
    os.path.join(OUT_DIR, "classification_feature_importance.csv"), index=False)
print("  classification_results.csv, classification_metrics.csv, classification_feature_importance.csv")


# Phase 9 Export Regression
print("Regression:")
reg_results = (gbt_pred
    .select("ID", "pledged", col("prediction").alias("pledged_log_pred"))
    .withColumn("pledged_pred", expm1(col("pledged_log_pred"))))
spark_write_single_csv(reg_results, "regression_results.csv")
pd.DataFrame(regression_metrics).to_csv(os.path.join(OUT_DIR, "regression_metrics.csv"), index=False)
pd.DataFrame(gbt_fi_pairs, columns=["feature", "importance"]).to_csv(
    os.path.join(OUT_DIR, "regression_feature_importance.csv"), index=False)
print("  regression_results.csv, regression_metrics.csv, regression_feature_importance.csv")


# Phase 9 Export Clustering + list all files
print("Clustering:")
# clustering_assignments: subset of the enriched table (already in pandas)
enriched_pd[["ID", "cluster", "goal", "usd_pledged_real", "backers",
             "campaign_duration_days", "funding_ratio"]].to_csv(
    os.path.join(OUT_DIR, "clustering_assignments.csv"), index=False)
cluster_profiles_pd.to_csv(os.path.join(OUT_DIR, "cluster_profiles.csv"), index=False)
cluster_centers_pd.to_csv(os.path.join(OUT_DIR, "cluster_centers.csv"), index=False)
# elbow (k, wcss) from the Phase 8.1 list
pd.DataFrame(wcss, columns=["k", "wcss"]).to_csv(os.path.join(OUT_DIR, "clustering_elbow.csv"), index=False)
print("  clustering_assignments.csv, cluster_profiles.csv, cluster_centers.csv, clustering_elbow.csv")

print("\nAll files in Output/:")
for f in sorted(os.listdir(OUT_DIR)):
    p = os.path.join(OUT_DIR, f)
    if os.path.isfile(p):
        print(f"  {f:<42} {os.path.getsize(p):>10,} bytes")


# PHASE 10: REPRODUCIBILITY VERIFICATION & EXECUTION SUMMARY

# Phase 10 Reproducibility Verification & Execution Summary
print("=" * 64)
print(" KICKSTARTER CAMPAIGN ANALYTICS - EXECUTION SUMMARY")
print("=" * 64)
print(f" Random seed (everywhere)      : {SEED}")
print(f" Raw rows loaded               : {n_rows:,}")
print(f" Rows after cleaning           : {df_model.count():,}")
print(f" Train / Test split (80/20)    : {n_train:,} / {n_test:,}")
print(f" Expanded feature vector size  : {len(feature_names)}")
print("-" * 64)
print(" Reproducibility guarantees:")
print("   - seed=42 on split, all models, and cross-validation")
print("   - preprocessing pipeline fit on TRAIN only (no scaler leakage)")
print("   - supervised features exclude pledged/backers-derived columns")
print("-" * 64)
print(" CLASSIFICATION (test set):")
for m in classification_metrics:
    print(f"   {m['model']:<26} acc={m['accuracy']:.4f}  f1={m['f1']:.4f}  auc={m['auc']:.4f}")
print(" REGRESSION (test set):")
for m in regression_metrics:
    print(f"   {m['model']:<26} R2(log)={m['r2_log']:.4f}  RMSE($)={m['rmse_orig']:,.0f}")
print(f" CLUSTERING: k={OPTIMAL_K}, silhouette={silhouette:.4f}")
print("=" * 64)
print(" All result CSVs exported to Output/ for Tableau.")
print(" Pipeline complete and reproducible.")
print("=" * 64)
