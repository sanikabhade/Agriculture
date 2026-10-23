
import pandas as pd

# 1. Load the dataset
DATA_FILE = (
    r"Dataset\Maharashtra_Agri_Master_2024_Jul2025.csv"
)

df = pd.read_csv(DATA_FILE)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

print("========== UNIT CHECK ==========")

print("\nPrice units:")
print(df["Price Unit"].value_counts(dropna=False))

print("\nArrival units:")
print(df["Arrival Unit"].value_counts(dropna=False))

print("\n========== DATE CHECK ==========")

print("\nInvalid dates:", df["Date"].isna().sum())

print("\nRepeated crop-market-date records:")

group_columns = [
    "District",
    "Market",
    "Commodity",
    "Date"
]

repeated = df.duplicated(
    subset=group_columns,
    keep=False
)

print("Number of rows belonging to repeated groups:", repeated.sum())

print("\n========== NUMERICAL VALIDATION ==========")

print("\nPrices less than or equal to zero:")
print((df["Modal Price"] <= 0).sum())

print("\nNegative arrival quantities:")
print((df["Arrival Quantity"] < 0).sum())

print("\nHumidity outside 0 to 100:")
print(
    (
        (df["Relative_Humidity_Percent"] < 0)
        | (df["Relative_Humidity_Percent"] > 100)
    ).sum()
)

print("\nTemperature consistency check:")
print(
    (
        (df["Min_Temperature_C"] > df["Temperature_C"])
        | (df["Temperature_C"] > df["Max_Temperature_C"])
    ).sum()
)

print("\n========== EXISTING DATE FEATURES ==========")

print("Year mismatches:", (
    df["Date"].dt.year != df["Year"]
).sum())

print("Month mismatches:", (
    df["Date"].dt.month != df["Month"]
).sum())

print("\nFeature validation completed.")
