import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==================================================
# SETTINGS
# ==================================================

sns.set_style("whitegrid")

os.makedirs("outputs/charts", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)

# ==================================================
# LOAD DATA
# ==================================================

df = pd.read_csv(
    "retail_sales_cleaned.csv",
    parse_dates=["transaction_date"]
)

print("=" * 60)
print("DATASET LOADED")
print("=" * 60)

print(df.shape)

# ==================================================
# BASIC INFORMATION
# ==================================================

print("\nDATA INFO")
print(df.info())

print("\nDESCRIPTIVE STATISTICS")
print(df.describe())

df.describe().to_csv(
    "outputs/reports/descriptive_statistics.csv"
)

# ==================================================
# MISSING VALUES
# ==================================================

missing = df.isnull().sum()

missing.to_csv(
    "outputs/reports/missing_values.csv"
)

print("\nMISSING VALUES")
print(missing)

# ==================================================
# DUPLICATES
# ==================================================

duplicates = df.duplicated().sum()

print("\nDUPLICATES:", duplicates)

# ==================================================
# NUMERICAL ANALYSIS
# ==================================================

numerical_cols = df.select_dtypes(
    include=np.number
).columns

for col in numerical_cols:

    plt.figure(figsize=(8,5))

    sns.histplot(
        df[col],
        kde=True
    )

    plt.title(f"Distribution of {col}")

    plt.tight_layout()

    plt.savefig(
        f"outputs/charts/{col}_distribution.png"
    )

    plt.close()

# ==================================================
# BOXPLOTS
# ==================================================

for col in numerical_cols:

    plt.figure(figsize=(8,5))

    sns.boxplot(x=df[col])

    plt.title(f"Outliers in {col}")

    plt.tight_layout()

    plt.savefig(
        f"outputs/charts/{col}_boxplot.png"
    )

    plt.close()

# ==================================================
# CORRELATION
# ==================================================

plt.figure(figsize=(12,8))

corr = df[numerical_cols].corr()

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Matrix")

plt.tight_layout()

plt.savefig(
    "outputs/charts/correlation_heatmap.png"
)

plt.close()

# ==================================================
# CATEGORY ANALYSIS
# ==================================================

cat_cols = df.select_dtypes(
    include="object"
).columns

for col in cat_cols:

    top = (
        df[col]
        .value_counts()
        .head(10)
    )

    plt.figure(figsize=(10,6))

    sns.barplot(
        x=top.values,
        y=top.index
    )

    plt.title(f"Top Values in {col}")

    plt.tight_layout()

    plt.savefig(
        f"outputs/charts/{col}_top_values.png"
    )

    plt.close()

# ==================================================
# SALES ANALYSIS
# ==================================================

print("\nTOTAL SALES")
print(df["sales_amount"].sum())

print("\nAVERAGE SALES")
print(df["sales_amount"].mean())

print("\nMAX SALE")
print(df["sales_amount"].max())

print("\nMIN SALE")
print(df["sales_amount"].min())

# ==================================================
# MONTHLY SALES
# ==================================================

monthly_sales = (
    df.groupby("month_name")["sales_amount"]
    .sum()
    .sort_values(ascending=False)
)

monthly_sales.to_csv(
    "outputs/reports/monthly_sales.csv"
)

plt.figure(figsize=(12,6))

monthly_sales.plot(
    kind="bar"
)

plt.title("Monthly Sales")

plt.ylabel("Revenue")

plt.tight_layout()

plt.savefig(
    "outputs/charts/monthly_sales.png"
)

plt.close()

# ==================================================
# REGION ANALYSIS
# ==================================================

region_sales = (
    df.groupby("region")["sales_amount"]
    .sum()
    .sort_values(ascending=False)
)

region_sales.to_csv(
    "outputs/reports/region_sales.csv"
)

plt.figure(figsize=(10,6))

region_sales.plot(
    kind="bar"
)

plt.title("Sales by Region")

plt.tight_layout()

plt.savefig(
    "outputs/charts/region_sales.png"
)

plt.close()

# ==================================================
# PRODUCT ANALYSIS
# ==================================================

product_sales = (
    df.groupby("product_name")["sales_amount"]
    .sum()
    .sort_values(ascending=False)
)

product_sales.head(20).to_csv(
    "outputs/reports/top_products.csv"
)

plt.figure(figsize=(12,8))

product_sales.head(20).plot(
    kind="bar"
)

plt.title("Top 20 Products")

plt.tight_layout()

plt.savefig(
    "outputs/charts/top_products.png"
)

plt.close()

# ==================================================
# BRAND ANALYSIS
# ==================================================

brand_sales = (
    df.groupby("brand")["sales_amount"]
    .sum()
    .sort_values(ascending=False)
)

brand_sales.to_csv(
    "outputs/reports/brand_sales.csv"
)

# ==================================================
# CATEGORY ANALYSIS
# ==================================================

category_sales = (
    df.groupby("category")["sales_amount"]
    .sum()
    .sort_values(ascending=False)
)

category_sales.to_csv(
    "outputs/reports/category_sales.csv"
)

plt.figure(figsize=(10,6))

category_sales.plot(
    kind="bar"
)

plt.title("Sales by Category")

plt.tight_layout()

plt.savefig(
    "outputs/charts/category_sales.png"
)

plt.close()

# ==================================================
# PAYMENT METHOD ANALYSIS
# ==================================================

payment_sales = (
    df.groupby("payment_method")
    ["sales_amount"]
    .sum()
    .sort_values(ascending=False)
)

payment_sales.to_csv(
    "outputs/reports/payment_analysis.csv"
)

# ==================================================
# CUSTOMER SEGMENT ANALYSIS
# ==================================================

segment_sales = (
    df.groupby("customer_segment")
    ["sales_amount"]
    .sum()
    .sort_values(ascending=False)
)

segment_sales.to_csv(
    "outputs/reports/customer_segment_sales.csv"
)

# ==================================================
# GENDER ANALYSIS
# ==================================================

gender_sales = (
    df.groupby("customer_gender")
    ["sales_amount"]
    .sum()
)

gender_sales.to_csv(
    "outputs/reports/gender_sales.csv"
)

# ==================================================
# AGE GROUP ANALYSIS
# ==================================================

age_sales = (
    df.groupby("customer_age_group")
    ["sales_amount"]
    .sum()
)

age_sales.to_csv(
    "outputs/reports/age_group_sales.csv"
)

# ==================================================
# DISCOUNT ANALYSIS
# ==================================================

discount_effect = (
    df.groupby("discount_pct")
    ["sales_amount"]
    .mean()
)

discount_effect.to_csv(
    "outputs/reports/discount_effect.csv"
)

# ==================================================
# SALES CHANNEL ANALYSIS
# ==================================================

channel_sales = (
    df.groupby("sales_channel")
    ["sales_amount"]
    .sum()
)

channel_sales.to_csv(
    "outputs/reports/channel_sales.csv"
)

# ==================================================
# DAILY TREND
# ==================================================

daily_sales = (
    df.groupby("transaction_date")
    ["sales_amount"]
    .sum()
)

plt.figure(figsize=(15,6))

daily_sales.plot()

plt.title("Daily Sales Trend")

plt.tight_layout()

plt.savefig(
    "outputs/charts/daily_sales_trend.png"
)

plt.close()

# ==================================================
# TOP CUSTOMERS
# ==================================================

top_customers = (
    df.groupby("customer_id")
    ["sales_amount"]
    .sum()
    .sort_values(ascending=False)
)

top_customers.head(50).to_csv(
    "outputs/reports/top_customers.csv"
)

# ==================================================
# SUMMARY REPORT
# ==================================================

with open(
    "outputs/summary.txt",
    "w"
) as f:

    f.write("RETAIL SALES EDA SUMMARY\n")
    f.write("="*50 + "\n\n")

    f.write(
        f"Rows: {df.shape[0]}\n"
    )

    f.write(
        f"Columns: {df.shape[1]}\n"
    )

    f.write(
        f"Total Sales: {df['sales_amount'].sum():,.2f}\n"
    )

    f.write(
        f"Average Sale: {df['sales_amount'].mean():,.2f}\n"
    )

    f.write(
        f"Highest Sale: {df['sales_amount'].max():,.2f}\n"
    )

print("\nEDA COMPLETED SUCCESSFULLY")
print("Results saved in outputs folder")