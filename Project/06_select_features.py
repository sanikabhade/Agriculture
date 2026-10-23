
import os
import pandas as pd

# 1. Load dataset
file_path = r"Dataset\Maharashtra_Agri_Master_2024_Jul2025.csv"

if not os.path.exists(file_path):
    print("ERROR: Dataset file not found:", file_path)
    raise SystemExit(1)

df = pd.read_csv(file_path)
print("Dataset loaded:", df.shape)

required_columns = [
    "District",
    "Market",
    "Commodity Group",
    "Commodity",
    "Date",
    "Arrival Quantity",
    "Modal Price",
    "Price Unit",
    "Latitude",
    "Longitude",
    "Temperature_C",
    "Min_Temperature_C",
    "Max_Temperature_C",
    "Precipitation_mm",
    "Relative_Humidity_Percent",
    "pH_0_30",
    "Organic_Carbon_g_kg_0_30",
    "Nitrogen_g_kg_0_30",
    "Clay_percent_0_30",
    "CEC_cmolc_kg_0_30",
    "Crop_Duration_Days",
    "Typical_Sowing_Season"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    print("ERROR: Required columns missing:", missing_columns)
    raise SystemExit(1)

df = df[required_columns].copy()

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date", "Modal Price"])


group_columns = ["District", "Market", "Commodity", "Price Unit"]

df = df.sort_values(group_columns + ["Date"]).copy()


df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["DayOfWeek"] = df["Date"].dt.dayofweek


grouped = df.groupby(group_columns, sort=False)

df["Previous_Price"] = grouped["Modal Price"].shift(1)
df["Previous_Arrival"] = grouped["Arrival Quantity"].shift(1)

df["Previous_7_Records_Avg_Price"] = (
    grouped["Modal Price"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=3).mean()
    )
)

df["Previous_7_Records_Avg_Arrival"] = (
    grouped["Arrival Quantity"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=3).mean()
    )
)

os.makedirs("outputs", exist_ok=True)

output_file = r"outputs\selected_feature_dataset.csv"
df.to_csv(output_file, index=False)


print("\nFeature engineering completed successfully!")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nFinal columns:")
for col in df.columns:
    print("-", col)

print("\nMissing values in historical features:")
print(
    df[
        [
            "Previous_Price",
            "Previous_Arrival",
            "Previous_7_Records_Avg_Price",
            "Previous_7_Records_Avg_Arrival"
        ]
    ].isna().sum()
)

print("\nOutput saved to:", os.path.abspath(output_file))
print("File exists:", os.path.exists(output_file))
