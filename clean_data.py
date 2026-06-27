import pandas as pd
import numpy as np

# Load Dataset
df = pd.read_csv("retail_sales_dataset.csv")

# ---------------------------------
# Basic Information
# ---------------------------------

print("Shape:", df.shape)
print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# ---------------------------------
# Remove Duplicates
# ---------------------------------

df.drop_duplicates(inplace=True)

# ---------------------------------
# Convert Date Column
# ---------------------------------

df['transaction_date'] = pd.to_datetime(
    df['transaction_date'],
    format='%d-%m-%Y'
)

# ---------------------------------
# Standardize Text Columns
# ---------------------------------

text_columns = [
    'customer_gender',
    'customer_age_group',
    'customer_segment',
    'product_name',
    'category',
    'brand',
    'payment_method',
    'sales_channel',
    'region'
]

for col in text_columns:
    df[col] = df[col].str.strip()

# ---------------------------------
# Check Negative Values
# ---------------------------------

numeric_columns = [
    'quantity',
    'unit_price',
    'discount_pct',
    'sales_amount'
]

for col in numeric_columns:
    print(f"{col} Negative Values:",
          (df[col] < 0).sum())

# ---------------------------------
# Validate Sales Formula
# ---------------------------------

expected_sales = (
    df['quantity']
    * df['unit_price']
    * (1 - df['discount_pct']/100)
).round(2)

sales_errors = df[
    abs(expected_sales - df['sales_amount']) > 0.01
]

print("\nSales Calculation Errors:")
print(len(sales_errors))

# ---------------------------------
# Create New Features
# ---------------------------------

df['year'] = df['transaction_date'].dt.year
df['month'] = df['transaction_date'].dt.month
df['month_name'] = df['transaction_date'].dt.month_name()
df['quarter'] = df['transaction_date'].dt.quarter
df['day_name'] = df['transaction_date'].dt.day_name()

# ---------------------------------
# Revenue Before Discount
# ---------------------------------

df['gross_sales'] = (
    df['quantity']
    * df['unit_price']
).round(2)

# ---------------------------------
# Discount Amount
# ---------------------------------

df['discount_amount'] = (
    df['gross_sales']
    - df['sales_amount']
).round(2)

# ---------------------------------
# Check Outliers Using IQR
# ---------------------------------

for col in ['quantity',
            'unit_price',
            'sales_amount']:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[
        (df[col] < lower) |
        (df[col] > upper)
    ]

    print(f"\n{col} Outliers:",
          len(outliers))

# ---------------------------------
# Final Check
# ---------------------------------

print("\nFinal Dataset Info:")
print(df.info())

# ---------------------------------
# Save Cleaned Dataset
# ---------------------------------

df.to_csv(
    "retail_sales_cleaned.csv",
    index=False
)

print("\nCleaned Dataset Saved Successfully!")