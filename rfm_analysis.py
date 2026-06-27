import pandas as pd

# Load Data
df = pd.read_csv(
    "retail_sales_cleaned.csv",
    parse_dates=["transaction_date"]
)

# Reference Date
snapshot_date = df["transaction_date"].max() + pd.Timedelta(days=1)

# RFM Table
rfm = df.groupby("customer_id").agg({
    "transaction_date": lambda x: (snapshot_date - x.max()).days,
    "customer_id": "count",
    "sales_amount": "sum"
})

rfm.columns = ["Recency", "Frequency", "Monetary"]

# Scores
rfm["R"] = pd.qcut(
    rfm["Recency"],
    5,
    labels=[5,4,3,2,1]
)

rfm["F"] = pd.qcut(
    rfm["Frequency"].rank(method="first"),
    5,
    labels=[1,2,3,4,5]
)

rfm["M"] = pd.qcut(
    rfm["Monetary"],
    5,
    labels=[1,2,3,4,5]
)

# Segment Function
def segment(row):

    r = int(row["R"])
    f = int(row["F"])

    if r >= 4 and f >= 4:
        return "VIP"

    elif r >= 3 and f >= 3:
        return "Loyal"

    elif r >= 3:
        return "Regular"

    return "At Risk"

rfm["Segment"] = rfm.apply(
    segment,
    axis=1
)

# Save Results
rfm.to_csv("rfm_customers.csv")

print("\nCustomer Segments\n")
print(rfm["Segment"].value_counts())