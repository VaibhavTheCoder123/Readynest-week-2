import pandas as pd
from prophet import Prophet

df = pd.read_csv(
    "retail_sales_cleaned.csv",
    parse_dates=["transaction_date"]
)

sales = (
    df.groupby("transaction_date")
    ["sales_amount"]
    .sum()
    .reset_index()
)

sales.columns = ["ds","y"]

model = Prophet()

model.fit(sales)

future = model.make_future_dataframe(
    periods=90
)

forecast = model.predict(future)

forecast.to_csv(
    "sales_forecast.csv",
    index=False
)

fig = model.plot(forecast)

print("Forecast Completed")