import pandas as pd

# Load dataset
df = pd.read_csv("data/churn.csv")

print("\n===== DATASET OVERVIEW =====")
print("Shape:", df.shape)

print("\n===== COLUMNS =====")
print(df.columns.tolist())

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== DUPLICATE ROWS =====")
print(df.duplicated().sum())

print("\n===== STATISTICAL SUMMARY =====")
print(df.describe(include="all"))

print("\n===== CHURN DISTRIBUTION =====")
print(df["churn"].value_counts())

print("\n===== CHURN PERCENTAGE =====")
print(df["churn"].value_counts(normalize=True) * 100)