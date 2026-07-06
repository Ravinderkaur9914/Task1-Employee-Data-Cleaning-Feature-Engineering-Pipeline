 ## 1. Overview
This project implements a complete, production-style Exploratory Data Analysis (EDA) and Feature Engineering pipeline for a simulated employee performance and salary dataset. The objective is to transform raw, chaotic data into a mathematically clean, machine-learning-ready dataset using rigorous statistical logic rather than ad-hoc guessing.
The pipeline follows an Input → Process → Output architecture, mirroring how enterprise-grade data engineering systems are structured before feeding data into predictive models.
 ## 2. Dataset Description
A simulated dataset of 500 employee records was generated to closely mimic real-world industrial data — including natural missingness patterns, injected outliers, and intentionally correlated features. Columns include:
•	employee_id, name — unique identifiers
•	age, experience_years — demographic and tenure attributes
•	department, education — categorical attributes
•	monthly_salary — target-like variable containing outliers
•	performance_score — 12% missing, correlated with department
•	monthly_hours_worked — 24% missing (highest missingness column)
•	projects_completed, tasks_completed — engineered to be highly correlated, used to demonstrate multicollinearity removal
•	join_date — used to derive employee tenure
 ## 3. Methodology
## 3.1 Handling Missing Data
Rather than applying one blanket method, missing values were treated according to a missingness-based decision matrix:
Missingness	Strategy	Applied To
< 5%	Drop rows	age, experience_years, monthly_salary, projects_completed, tasks_completed
5% – 20%	Group-wise median imputation	performance_score (grouped by department)
> 20%	KNN Imputation (k = 5)	monthly_hours_worked

Dropping rows at low missingness avoids introducing synthetic bias while barely affecting overall sample size. Group-wise imputation preserves the natural relationship between a correlated categorical variable and the missing numeric value. KNN imputation is reserved for columns with missingness too high to trust a single global statistic, since it estimates values from the 5 most similar rows across all numeric features.
## 3.2 Outlier Neutralization (IQR Method)
Outliers in monthly_salary, monthly_hours_worked, and performance_score were identified using the Interquartile Range (IQR):
Lower Bound = Q1 − 1.5 × IQR       Upper Bound = Q3 + 1.5 × IQR
Rather than deleting outlier rows, values were capped (winsorized) at these bounds using numpy.clip(). This preserves row count and avoids destroying otherwise valid data in adjacent columns — an important consideration when downstream components require consistent sample volume.
## 3.3 Categorical Encoding
department and education were one-hot encoded rather than label-encoded. Label encoding assigns ascending integers to categories, which creates a false mathematical distance (e.g., implying a 'PhD' is three times greater than a 'Bachelors' degree). One-hot encoding instead maps each category onto its own orthogonal coordinate axis, which is mathematically appropriate for nominal data fed into numerical estimators.
## 3.4 Multicollinearity Eradication
A correlation matrix was computed across all numeric features, and the upper triangle was isolated to avoid double-counting feature pairs. Any pair with an absolute correlation above 0.80 was flagged. For each flagged pair, the feature more weakly correlated with the target variable (monthly_salary) was removed, preserving the more predictive feature.
In this run, age, projects_completed, and tasks_completed were dropped, as all three were highly collinear with experience_years, which was retained.
 ## 3.5 Feature Engineering
Four new predictive features were engineered from the existing columns:
Feature	Logic	Rationale
salary_per_experience_year	monthly_salary / (experience_years + 1)	Normalizes pay by tenure as a value/efficiency signal
tenure_years	(reference_date − join_date) / 365.25	Converts a raw date into a usable numeric feature
performance_per_hour	performance_score / monthly_hours_worked	Productivity ratio independent of raw hours worked
seniority_level	Binned experience_years into Junior / Mid / Senior / Lead	Captures non-linear seniority effects a linear model may miss
## 3.6 Structural Contract Validation

A Pandera schema was applied to the final dataset to enforce that key numeric columns fall within valid, expected ranges (e.g., performance_score between 0 and 100). This runtime validation step catches silent data corruption before the dataset reaches a model, mirroring the safeguards used in production machine learning pipelines to prevent training-serving skew.
## 4. Results
•	Raw dataset: 500 rows × 11 columns
•	Final cleaned dataset: 485 rows × 20 columns
•	All missing values resolved using the appropriate strategy per column
•	14 salary outliers, 9 working-hour outliers, and 2 performance-score outliers capped via IQR
•	3 redundant, highly collinear features removed
•	4 new engineered features added
•	Final dataset passed Pandera schema validation with zero errors
## 5. Tools & Libraries Used
•	Python — pandas, NumPy
•	scikit-learn — KNNImputer
•	Pandera — runtime schema validation
## 6. Conclusion
Machine learning estimators possess no qualitative reasoning — they are numerical optimization algorithms operating purely on coordinate spaces. Every decision in this pipeline, from the imputation method chosen to the correlation threshold used for feature removal, exists to ensure the resulting dataset is a high-fidelity, distortion-free numerical representation of the underlying real-world data.
Data cleaning is not preprocessing busywork — it is the structural engineering that determines whether a downstream model's predictions can be trusted. This project demonstrates that discipline end-to-end: from raw, chaotic input to a validated, ML-ready dataset.
