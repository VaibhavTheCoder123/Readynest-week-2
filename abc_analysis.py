import pandas as pd

df = pd.read_csv("retail_sales_cleaned.csv")

product_sales = (
    df.groupby("product_name")["sales_amount"]
    .sum()
    .reset_index()
)

product_sales = product_sales.sort_values(
    "sales_amount",
    ascending=False
)

total_sales = product_sales["sales_amount"].sum()

product_sales["cum_pct"] = (
    product_sales["sales_amount"].cumsum()
    / total_sales
) * 100

def classify(x):

    if x <= 70:
        return "A"

    elif x <= 90:
        return "B"

    return "C"

product_sales["ABC_Class"] = (
    product_sales["cum_pct"]
    .apply(classify)
)

product_sales.to_csv(
    "abc_product_analysis.csv",
    index=False
)

print(product_sales.head())