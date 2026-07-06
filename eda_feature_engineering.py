"""
Project 1: Advanced EDA & Feature Engineering
DecodeLabs Data Science Industrial Training Kit - Batch 2026

Dataset: Employee Performance & Salary Records (simulated, industry-style)
Author: [Your Name]

Pipeline follows the Input -> Process -> Output architecture:
  PHASE 1: Securing Input Fidelity   -> missing values, outlier boundaries
  PHASE 2: Vectorized Computation    -> encoding, collinearity eradication
  PHASE 3: Structural Contracts       -> schema validation, final feature set
"""

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
import pandera.pandas as pa
from pandera import Column, Check, DataFrameSchema

pd.set_option("display.max_columns", None)

# =============================================================
# LOAD RAW DATA
# =============================================================
df = pd.read_csv("raw_employee_data.csv")
print("Raw shape:", df.shape)
print("\nMissing value % per column:\n", (df.isna().mean() * 100).round(2))

# =============================================================
# PHASE 1: SECURING INPUT FIDELITY
# =============================================================

# ---- 1a. Missing Data Decision Matrix ----
missing_pct = df.isna().mean() * 100

for col in df.columns:
    pct = missing_pct[col]
    if pct == 0:
        continue

    if pct < 5:
        # < 5% -> drop rows (preserves distribution, avoids synthetic bias)
        df = df.dropna(subset=[col])
        print(f"[{col}] {pct:.1f}% missing -> rows dropped")

    elif 5 <= pct <= 20:
        if col == "performance_score":
            # Categorical/correlated case -> group-wise (sub-population) imputation
            df[col] = df.groupby("department")[col].transform(
                lambda x: x.fillna(x.median())
            )
            print(f"[{col}] {pct:.1f}% missing -> group-wise median imputation (by department)")
        else:
            # Skewed numeric -> global median (robust to outliers)
            df[col] = df[col].fillna(df[col].median())
            print(f"[{col}] {pct:.1f}% missing -> global median imputation")

    else:
        # > 20% -> KNN Imputation (multi-dimensional estimation)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        imputer = KNNImputer(n_neighbors=5)
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        print(f"[{col}] {pct:.1f}% missing -> KNN imputation (k=5)")

# Re-run age/experience/salary drop cleanup (dropna above only targets the < 5% col each pass,
# so consolidate here to ensure no NaNs remain in low-missingness columns)
df = df.dropna(subset=["age", "experience_years", "monthly_salary", "projects_completed", "tasks_completed"])
df = df.reset_index(drop=True)

# Ensure employee name (2nd column) survives cleaning unchanged
assert "name" in df.columns, "name column missing from raw data"

print("\nShape after missing-value handling:", df.shape)
assert df.isna().sum().sum() == 0, "Unhandled missing values remain!"

# ---- 1b. Outlier Detection & Neutralization via IQR ----
def iqr_bounds(series):
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    return Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

outlier_cols = ["monthly_salary", "monthly_hours_worked", "performance_score"]
outlier_report = {}

for col in outlier_cols:
    lower, upper = iqr_bounds(df[col])
    n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    outlier_report[col] = {"lower": round(lower, 2), "upper": round(upper, 2), "count": int(n_outliers)}
    # Winsorize: cap instead of delete -> preserves row count & sequence integrity
    df[col] = df[col].clip(lower, upper)

print("\nOutlier bounds & counts capped:")
for col, info in outlier_report.items():
    print(f"  {col}: bounds=({info['lower']}, {info['upper']}), outliers capped={info['count']}")

# =============================================================
# PHASE 2: VECTORIZED COMPUTATION ENGINE
# =============================================================

# ---- 2a. Categorical Translation into Coordinate Space (One-Hot Encoding) ----
df = pd.get_dummies(df, columns=["department", "education"], drop_first=False)

# ---- 2b. Multicollinearity Detection & Eradication ----
numeric_df = df.select_dtypes(include=[np.number]).drop(columns=["employee_id"], errors="ignore")
corr_matrix = numeric_df.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

THRESHOLD = 0.80
target = "monthly_salary"  # treat as the y-proxy for this exercise

to_drop = set()
for col in upper_tri.columns:
    correlated_with = upper_tri.index[upper_tri[col] > THRESHOLD].tolist()
    for other in correlated_with:
        if col == target or other == target:
            continue
        # keep the one more correlated with target, drop the weaker one
        corr_col = numeric_df[col].corr(numeric_df[target])
        corr_other = numeric_df[other].corr(numeric_df[target])
        weaker = col if abs(corr_col) < abs(corr_other) else other
        to_drop.add(weaker)

print(f"\nHighly correlated pairs (>{THRESHOLD}) found. Dropping weaker-target-correlated features: {to_drop}")
df = df.drop(columns=list(to_drop))

# =============================================================
# FEATURE ENGINEERING (3+ new predictive features required)
# =============================================================

# 1. Salary per year of experience (efficiency/value ratio)
df["salary_per_experience_year"] = (df["monthly_salary"] / (df["experience_years"] + 1)).round(2)

# 2. Tenure in years, derived from join_date
df["join_date"] = pd.to_datetime(df["join_date"])
REFERENCE_DATE = pd.Timestamp("2026-07-05")
df["tenure_years"] = ((REFERENCE_DATE - df["join_date"]).dt.days / 365.25).round(2)

# 3. Performance-to-workload ratio (productivity indicator)
df["performance_per_hour"] = (df["performance_score"] / df["monthly_hours_worked"]).round(4)

# 4. Seniority bucket (binned feature from continuous experience)
df["seniority_level"] = pd.cut(
    df["experience_years"],
    bins=[-0.01, 2, 5, 10, np.inf],
    labels=["Junior", "Mid", "Senior", "Lead"]
)

print("\nNew engineered features added: salary_per_experience_year, tenure_years, performance_per_hour, seniority_level")

# =============================================================
# PHASE 3: STRUCTURAL CONTRACTS (Pandera schema validation)
# =============================================================

schema = DataFrameSchema(
    {
        "experience_years": Column(float, Check.ge(0), nullable=False),
        "monthly_salary": Column(float, Check.gt(0), nullable=False),
        "performance_score": Column(float, Check.in_range(0, 100), nullable=False),
        "monthly_hours_worked": Column(float, Check.ge(0), nullable=False),
    },
    strict=False,  # allow other engineered/encoded columns to pass through
)

try:
    schema.validate(df, lazy=True)
    print("\n✅ Pandera schema validation PASSED — dataset meets structural contract.")
except pa.errors.SchemaErrors as err:
    print("\n⚠️ Schema validation issues found:")
    print(err.failure_cases)

# =============================================================
# SAVE FINAL CLEAN DATASET
# =============================================================
df.to_csv("cleaned_employee_data.csv", index=False)
print("\nFinal cleaned shape:", df.shape)
print("Saved to cleaned_employee_data.csv")
print("\nFinal columns:\n", list(df.columns))
