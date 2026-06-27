import pandas as pd

df = pd.read_csv(
    "retail_sales_cleaned.csv"
)

clv = df.groupby(
    "customer_id"
).agg({
    "sales_amount":["sum","mean","count"]
})

clv.columns = [
    "TotalRevenue",
    "AverageOrderValue",
    "PurchaseFrequency"
]

clv["CLV"] = (
    clv["AverageOrderValue"]
    * clv["PurchaseFrequency"]
)

clv = clv.sort_values(
    "CLV",
    ascending=False
)

clv.to_csv(
    "customer_clv.csv"
)

print(
    clv.head(10)
)