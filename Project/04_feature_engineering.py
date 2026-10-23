import os
import pandas as pd
DATA_FILE = (
    r"Dataset\Maharashtra_Agri_Master_2024_Jul2025.csv"
)

df = pd.read_csv(DATA_FILE)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

df = df.sort_values(
    by=["District", "Market", "Commodity", "Date"]
).copy()

df["DayOfWeek"] = df["Date"].dt.dayofweek
df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)

group_columns = [
    "District",
    "Market",
    "Commodity",
    "Price Unit"
]
df["Previous_Price"] = (
    df.groupby(group_columns)["Modal Price"].shift(1)
)

df["Previous_Arrival"] = (
    df.groupby(group_columns)["Arrival Quantity"].shift(1)
)

df["Previous_7_Records_Avg_Price"] = (
    df.groupby(group_columns)["Modal Price"]
    .transform(
        lambda s: s.shift(1).rolling(
            window=7, min_periods=3
        ).mean()
    )
)

df["Previous_7_Records_Avg_Arrival"] = (
    df.groupby(group_columns)["Arrival Quantity"]
    .transform(
        lambda s: s.shift(1).rolling(
            window=7, min_periods=3
        ).mean()
    )
)

new_features = [
    "DayOfWeek",
    "IsWeekend",
    "Previous_Price",
    "Previous_Arrival",
    "Previous_7_Records_Avg_Price",
    "Previous_7_Records_Avg_Arrival"
]

print("========== FEATURE ENGINEERING ==========")
print("Original dataset shape:", pd.read_csv(DATA_FILE).shape)
print("New dataset shape:", df.shape)

print("\nNew features:")
print(new_features)

print("\nFeature preview:")
print(
    df[
        ["Date", "District", "Market", "Commodity",
         "Price Unit", "Modal Price"] + new_features
    ].head(15).to_string(index=False)
)

print("\nMissing values in new features:")
print(df[new_features].isna().sum())

os.makedirs("outputs", exist_ok=True)

OUTPUT_FILE = r"outputs\feature_engineered_dataset.csv"
df.to_csv(OUTPUT_FILE, index=False)

print("\nFeature-engineered dataset saved successfully!")
print("Saved at:", OUTPUT_FILE)
