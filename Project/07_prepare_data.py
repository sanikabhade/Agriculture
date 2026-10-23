import os
import pandas as pd
file_path = r"outputs\selected_feature_dataset.csv"

if not os.path.exists(file_path):
    print("ERROR: Dataset not found. Run 06_select_features.py first.")
    raise SystemExit(1)

df = pd.read_csv(file_path)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

print("Dataset loaded:", df.shape)

df = df.dropna(subset=["Previous_Price"]).copy()

y = df["Modal Price"]


feature_columns = [
    "District",
    "Market",
    "Commodity Group",
    "Commodity",
    "Price Unit",
    "Arrival Quantity",
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
    "Typical_Sowing_Season",
    "Year",
    "Month",
    "DayOfWeek",
    "Previous_Price",
    "Previous_Arrival",
    "Previous_7_Records_Avg_Price",
    "Previous_7_Records_Avg_Arrival"
]

X = df[feature_columns].copy()

df = df.sort_values("Date").reset_index(drop=True)
split_index = int(len(df) * 0.80)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

X_train = train_df[feature_columns]
y_train = train_df["Modal Price"]

X_test = test_df[feature_columns]
y_test = test_df["Modal Price"]

os.makedirs("outputs", exist_ok=True)

X_train.to_csv(r"outputs\X_train.csv", index=False)
X_test.to_csv(r"outputs\X_test.csv", index=False)
y_train.to_csv(r"outputs\y_train.csv", index=False)
y_test.to_csv(r"outputs\y_test.csv", index=False)

print("\n========== DATA PREPARATION ==========")
print("Rows available after removing missing previous prices:", len(df))
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

print("\nX_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

print("\nTraining date range:")
print(train_df["Date"].min(), "to", train_df["Date"].max())

print("\nTesting date range:")
print(test_df["Date"].min(), "to", test_df["Date"].max())

print("\nData preparation completed successfully!")
