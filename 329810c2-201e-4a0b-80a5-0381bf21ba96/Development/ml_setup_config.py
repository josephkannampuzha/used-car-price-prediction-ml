# ==========================
# Imports
# ==========================
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)

# ==========================
# Load Dataset
# ==========================
car_df = pd.read_csv("used_car_price_prediction_1M.csv")

# ==========================
# Dataset Overview
# ==========================
print("=" * 50)
print("Dataset Overview")
print("=" * 50)
print(f"Rows: {car_df.shape[0]:,}")
print(f"Columns: {car_df.shape[1]}\n")

print("Dataset Information:")
car_df.info()

# ==========================
# Missing Values
# ==========================
print("\nMissing Values:")
print(car_df.isnull().sum())

# ==========================
# Duplicate Records
# ==========================
_duplicates = car_df.duplicated().sum()
print(f"\nDuplicate Rows: {_duplicates:,}")

if _duplicates > 0:
    car_df.drop_duplicates(inplace=True)
    print(f"Duplicates removed. New shape: {car_df.shape}")

# ==========================
# Fix Invalid Values
# ==========================

# Horsepower cannot be negative
car_df.loc[car_df["Horsepower"] < 0, "Horsepower"] = np.nan

# Remove leading/trailing whitespace and standardize text
text_cols = [
    "Brand",
    "Model",
    "Fuel_Type",
    "Transmission",
    "Owner_Type",
    "Color",
    "City"
]

for col in text_cols:
    car_df[col] = car_df[col].str.strip().str.title()

# ==========================
# Handle Missing Values
# ==========================

# Numerical columns -> Median
numeric_cols = [
    "Mileage_kmpl",
    "Engine_CC",
    "Horsepower"
]

for col in numeric_cols:
    car_df[col].fillna(car_df[col].median(), inplace=True)

# Categorical columns -> Mode
categorical_cols = [
    "Fuel_Type",
    "Transmission",
    "Color",
    "City"
]

for col in categorical_cols:
    car_df[col].fillna(car_df[col].mode()[0], inplace=True)

# ==========================
# Feature Engineering
# ==========================

CURRENT_YEAR = 2026

car_df["Car_Age"] = CURRENT_YEAR - car_df["Year"]

# ==========================
# Final Quality Check
# ==========================

print("\nRemaining Missing Values:")
print(car_df.isnull().sum())

print(f"\nFinal Dataset Shape: {car_df.shape}")

print("\nUnique Brands:", car_df["Brand"].nunique())
print("Unique Models:", car_df["Model"].nunique())

print("\nCleaned dataset ready for exploratory analysis and machine learning.")
