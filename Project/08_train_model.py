import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
file_path = r"outputs\selected_feature_dataset.csv"

if not os.path.exists(file_path):
    print("ERROR: Dataset not found.")
    print("Run 06_select_features.py first.")
    raise SystemExit(1)

df = pd.read_csv(file_path)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date", "Modal Price", "Previous_Price"])
df = df.sort_values("Date").reset_index(drop=True)

print("Dataset loaded:", df.shape)

target = "Modal Price"

features = [
    "District", "Market", "Commodity Group", "Commodity",
    "Price Unit", "Arrival Quantity", "Latitude", "Longitude",
    "Temperature_C", "Min_Temperature_C", "Max_Temperature_C",
    "Precipitation_mm", "Relative_Humidity_Percent",
    "pH_0_30", "Organic_Carbon_g_kg_0_30",
    "Nitrogen_g_kg_0_30", "Clay_percent_0_30",
    "CEC_cmolc_kg_0_30", "Crop_Duration_Days",
    "Typical_Sowing_Season", "Year", "Month", "DayOfWeek",
    "Previous_Price", "Previous_Arrival",
    "Previous_7_Records_Avg_Price",
    "Previous_7_Records_Avg_Arrival"
]

missing_columns = [col for col in features if col not in df.columns]
if missing_columns:
    print("ERROR: Missing columns:", missing_columns)
    raise SystemExit(1)

X = df[features].copy()
y = df[target].copy()

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_columns = [
    col for col in X.columns if col not in categorical_columns
]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numerical_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

print("\nTraining Random Forest model...")
model.fit(X_train, y_train)
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, predictions)

print("\n========== MODEL RESULTS ==========")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")

os.makedirs("outputs", exist_ok=True)

results = pd.DataFrame({
    "Actual_Price": y_test.to_numpy(),
    "Predicted_Price": predictions
})

results.to_csv(r"outputs\random_forest_predictions.csv", index=False)

print("\nPredictions saved to outputs\\random_forest_predictions.csv")
print("\nFirst 10 predictions:")
print(results.head(10).to_string(index=False))
