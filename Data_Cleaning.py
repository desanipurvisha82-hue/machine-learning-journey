import pandas as pd

# ============================================================
# DAY 4 — DATA CLEANING & PREPROCESSING
# ============================================================

# ------------------------------------------------------------
# 1. CREATE DATASET
# ------------------------------------------------------------

df = pd.DataFrame({
    "Name": [" John ", "Alice", "Bob", " John ", "David", "Emma"],
    "Age": ["22", None, "25", "22", "200", "28"],
    "Gender": ["Male", "Female", "Male", "Male", "Female", "Female"],
    "City": [" Delhi", "Mumbai ", "delhi", " Delhi", "Pune", "Mumbai"],
    "Math": [80, 90, 70, 80, 95, None],
    "Science": [75, 85, 80, 75, 90, 88],
    "Passed": ["Yes", "Yes", "Yes", "Yes", "Yes", "Yes"]
})

print("ORIGINAL DATA:")
print(df)


# ============================================================
# 2. INSPECT DATA
# ============================================================

print("\n--- HEAD ---")
print(df.head())

print("\n--- DATA TYPES ---")
print(df.dtypes)

print("\n--- DATA INFORMATION ---")
df.info()

print("\n--- STATISTICS ---")
print(df.describe())

print("\n--- SHAPE ---")
print(df.shape)


# ============================================================
# 3. MISSING VALUES
# ============================================================

print("\n--- MISSING VALUES ---")
print(df.isnull())

print("\n--- MISSING VALUES COUNT ---")
print(df.isnull().sum())


# ============================================================
# 4. CONVERT AGE TO NUMERIC
# ============================================================

df["Age"] = pd.to_numeric(
    df["Age"],
    errors="coerce"
)

print("\n--- AGE AFTER CONVERSION ---")
print(df["Age"])

print("\n--- DATA TYPES AFTER AGE CONVERSION ---")
print(df.dtypes)


# ============================================================
# 5. HANDLE MISSING VALUES
# ============================================================

# Fill Age missing value with median
df["Age"] = df["Age"].fillna(
    df["Age"].median()
)

# Fill Math missing value with mean
df["Math"] = df["Math"].fillna(
    df["Math"].mean()
)

print("\n--- AFTER HANDLING MISSING VALUES ---")
print(df)

print("\n--- MISSING VALUES AFTER CLEANING ---")
print(df.isnull().sum())


# ============================================================
# 6. CLEAN TEXT DATA
# ============================================================

# Remove unnecessary spaces
df["Name"] = df["Name"].str.strip()
df["City"] = df["City"].str.strip()

# Convert text to lowercase
df["Name"] = df["Name"].str.lower()
df["City"] = df["City"].str.lower()

print("\n--- AFTER TEXT CLEANING ---")
print(df)


# ============================================================
# 7. CHECK DUPLICATES
# ============================================================

print("\n--- DUPLICATES ---")
print(df.duplicated())

print("\n--- NUMBER OF DUPLICATES ---")
print(df.duplicated().sum())

print("\n--- DUPLICATE ROWS ---")
print(df[df.duplicated()])


# ============================================================
# 8. REMOVE DUPLICATES
# ============================================================

df = df.drop_duplicates()

print("\n--- AFTER REMOVING DUPLICATES ---")
print(df)


# ============================================================
# 9. DETECT OUTLIERS USING IQR
# ============================================================

Q1 = df["Age"].quantile(0.25)
Q3 = df["Age"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("\n--- OUTLIER CALCULATION ---")
print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)


# Find outliers
outliers = df[
    (df["Age"] < lower_bound) |
    (df["Age"] > upper_bound)
]

print("\n--- OUTLIERS ---")
print(outliers)


# ============================================================
# 10. REMOVE OUTLIERS
# ============================================================

df = df[
    (df["Age"] >= lower_bound) &
    (df["Age"] <= upper_bound)
]

print("\n--- AFTER REMOVING OUTLIERS ---")
print(df)


# ============================================================
# 11. SORT DATA
# ============================================================

# Sort by Age
df = df.sort_values(
    "Age",
    ascending=True
)

print("\n--- SORTED BY AGE ---")
print(df)


# Sort by Math from highest to lowest
df = df.sort_values(
    "Math",
    ascending=False
)

print("\n--- SORTED BY MATH ---")
print(df)


# ============================================================
# 12. CREATE NEW FEATURES
# ============================================================

# Total marks
df["Total"] = df["Math"] + df["Science"]

# Average marks
df["Average"] = df["Total"] / 2

print("\n--- AFTER CREATING TOTAL AND AVERAGE ---")
print(df)


# ============================================================
# 13. CREATE PASS/FAIL COLUMN
# ============================================================

df["Result"] = df["Average"] >= 50

print("\n--- RESULT COLUMN ---")
print(df)


# ============================================================
# 14. ONE-HOT ENCODING
# ============================================================

# Encode categorical columns
df = pd.get_dummies(
    df,
    columns=["Gender", "City"],
    drop_first=True
)

print("\n--- AFTER ONE-HOT ENCODING ---")
print(df)


# ============================================================
# 15. SEPARATE FEATURES (X) AND TARGET (y)
# ============================================================

# Target
y = df["Passed"]

# Features
X = df.drop(
    "Passed",
    axis=1
)

print("\n--- FEATURES (X) ---")
print(X)

print("\n--- TARGET (y) ---")
print(y)


# ============================================================
# 16. FINAL DATA TYPES
# ============================================================

print("\n--- FINAL DATA TYPES ---")
print(X.dtypes)


# ============================================================
# 17. FINAL DATASET
# ============================================================

print("\n--- FINAL CLEAN DATASET ---")
print(df)