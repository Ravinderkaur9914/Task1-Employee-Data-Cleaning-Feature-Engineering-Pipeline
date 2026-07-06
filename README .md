# Project 1: Advanced EDA & Feature Engineering

**DecodeLabs Data Science Industrial Training Kit — Batch 2026**

## Overview

This project transforms a raw, messy employee dataset into a mathematically clean, ML-ready dataset using pure statistical logic — no arbitrary guessing. It follows the Input → Process → Output architecture:

1. **Phase 1 — Securing Input Fidelity**: handle missing values, neutralize outliers
2. **Phase 2 — Vectorized Computation**: encode categoricals, eliminate multicollinearity
3. **Phase 3 — Structural Contracts**: validate the final dataset against a schema

## Files

| File | Description |
|---|---|
| `raw_employee_data.csv` | Simulated raw dataset (500 rows) with missing values, outliers, and correlated features |
| `cleaned_employee_data.csv` | Final cleaned, encoded, feature-engineered dataset (485 rows) |
| `eda_feature_engineering.py` | Full pipeline script — run this to reproduce the cleaning end-to-end |
| `README.md` | This file |

## Dataset

Employee performance & salary records with the following raw columns:

- `employee_id`, `name` — identifiers
- `age`, `experience_years` — demographic/tenure info
- `department`, `education` — categorical attributes
- `monthly_salary` — target-like variable, contains outliers
- `performance_score` — 12% missing, correlated with department
- `monthly_hours_worked` — 24% missing (highest missingness column)
- `projects_completed`, `tasks_completed` — near-duplicate signals (engineered to be highly correlated, for demonstrating multicollinearity removal)
- `join_date` — used to derive tenure

## Methodology

### 1. Missing Data — The Decision Matrix

| Missingness | Strategy | Applied to |
|---|---|---|
| < 5% | Drop rows | `age`, `experience_years`, `monthly_salary`, `projects_completed`, `tasks_completed` |
| 5–20% | Group-wise median imputation | `performance_score` (grouped by `department`) |
| > 20% | KNN Imputation (k=5) | `monthly_hours_worked` |

**Why not one method for everything?** Dropping rows at low missingness avoids introducing synthetic bias while barely affecting sample size. Group-wise imputation preserves the natural relationship between a correlated categorical (department) and the missing numeric value. KNN is used only when missingness is too high to trust a single global statistic — it estimates values from the 5 most similar rows across all numeric features.

### 2. Outlier Neutralization — IQR Method

For `monthly_salary`, `monthly_hours_worked`, and `performance_score`:

```
Lower Bound = Q1 − 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

Outliers are **capped (winsorized)**, not deleted — this preserves row count and avoids destroying otherwise-valid data in adjacent columns.

### 3. Categorical Encoding

`department` and `education` are **one-hot encoded** rather than label-encoded. Label encoding would impose a false ordinal/distance relationship (e.g., implying "PhD" is mathematically 3x "Bachelors"), which corrupts distance-based and linear estimators. One-hot encoding maps each category to its own orthogonal axis.

### 4. Multicollinearity Eradication

- Built the absolute correlation matrix across all numeric features
- Isolated the upper triangle to avoid double-counting pairs
- Flagged pairs with correlation > 0.80
- For each flagged pair, kept the feature more strongly correlated with the target (`monthly_salary`) and dropped the weaker one

Dropped in this run: `age`, `projects_completed`, `tasks_completed` (all highly collinear with `experience_years`, which was retained).

### 5. Feature Engineering (4 new features)

| Feature | Formula / Logic | Rationale |
|---|---|---|
| `salary_per_experience_year` | `monthly_salary / (experience_years + 1)` | Normalizes pay by tenure — a value/efficiency signal |
| `tenure_years` | `(reference_date − join_date) / 365.25` | Converts a raw date into a usable numeric feature |
| `performance_per_hour` | `performance_score / monthly_hours_worked` | Productivity ratio, independent of raw hours worked |
| `seniority_level` | Binned `experience_years` into Junior/Mid/Senior/Lead | Captures non-linear seniority effects a linear model might miss |

### 6. Structural Contract Validation

A `pandera` schema enforces at runtime that key numeric columns fall within valid, expected ranges (e.g., `performance_score` between 0–100). This catches silent data corruption before the dataset reaches a model — the same principle production ML pipelines use to prevent training-serving skew.

## How to Run

```bash
pip install pandas numpy scikit-learn pandera --break-system-packages
python eda_feature_engineering.py
```

Output: `cleaned_employee_data.csv` (485 rows × 20 columns), plus a console log of every imputation, outlier bound, and dropped feature.

## Key Takeaway

Machine learning estimators have no qualitative reasoning — they optimize purely over numeric coordinate spaces. Every cleaning decision above (imputation method, outlier bound, encoding scheme, correlation threshold) exists to ensure the model receives high-fidelity, distortion-free numeric input. Data cleaning isn't preprocessing busywork — it's the structural engineering that determines whether the resulting model's predictions can be trusted.

---
*Batch: 2026 | DecodeLabs Industrial Training Kit*
