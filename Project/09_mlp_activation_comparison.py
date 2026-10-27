
# 09_mlp_activation_comparison.py

import os
import time
import ast
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore", category=UserWarning)

# ==================================================
# STEP 1: FIND AND LOAD THE SELECTED FEATURE DATASET
# ==================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

possible_paths = [
    os.path.join(
        PROJECT_DIR, "Dataset", "selected_feature_dataset.csv"
    ),
    os.path.join(
        PROJECT_DIR, "outputs", "selected_feature_dataset.csv"
    ),
    os.path.join(
        PROJECT_DIR, "selected_feature_dataset.csv"
    ),
    # Your previously reported dataset location
    r"C:\Users\91776\OneDrive\Desktop\Agriculture\Project"
    r"\outputs\selected_feature_dataset.csv"
]

file_path = next(
    (path for path in possible_paths if os.path.isfile(path)),
    None
)

if file_path is None:
    print("CSV files found in Dataset folder:")
    dataset_dir = os.path.join(PROJECT_DIR, "Dataset")

    if os.path.isdir(dataset_dir):
        print(os.listdir(dataset_dir))
    else:
        print("Dataset folder was not found:", dataset_dir)

    raise FileNotFoundError(
        "Could not find selected_feature_dataset.csv. "
        "Check the filename and folder."
    )

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Dataset path:", file_path)
print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# ==================================================
# STEP 2: SELECT FEATURES AND TARGET
# ==================================================

selected_features = [
    "District",
    "Market",
    "Commodity Group",
    "Commodity",
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

target = "Modal Price"

required_columns = selected_features + [target, "Date"]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in CSV: {missing_columns}"
    )

# ==================================================
# STEP 3: CLEAN DATA AND PRICE UNIT
# ==================================================

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df[target] = pd.to_numeric(df[target], errors="coerce")

df = df.dropna(subset=["Date", target]).copy()
df = df[df[target] > 0].copy()

if "Price Unit" in df.columns:
    print("\nPrice units before filtering:")
    print(df["Price Unit"].value_counts())

    df = df[
        df["Price Unit"].astype(str).str.strip()
        == "Rs./Quintal"
    ].copy()

if df.empty:
    raise ValueError(
        "No rows remain. Check your Price Unit column."
    )

df = df.sort_values("Date").reset_index(drop=True)

X = df[selected_features].copy()
y = df[target].copy()

print("\nRows after cleaning:", len(df))
print("Selected features:", len(selected_features))

# ==================================================
# STEP 4: TIME-BASED TRAIN/TEST SPLIT
# ==================================================

split_date = pd.Timestamp("2025-04-01")

train_mask = df["Date"] < split_date
test_mask = df["Date"] >= split_date

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_train = y.loc[train_mask]
y_test = y.loc[test_mask]

if X_train.empty or X_test.empty:
    raise ValueError(
        "Training or testing data is empty. "
        "Check your date range."
    )

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Training dates:",
      df.loc[train_mask, "Date"].min(),
      "to", df.loc[train_mask, "Date"].max())
print("Testing dates:",
      df.loc[test_mask, "Date"].min(),
      "to", df.loc[test_mask, "Date"].max())

# ==================================================
# STEP 5: PREPROCESSING
# ==================================================

categorical_features = X.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

numeric_features = [
    column for column in selected_features
    if column not in categorical_features
]

print("\nCategorical features:", categorical_features)
print("Numeric features:", numeric_features)

try:
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True
    )
except TypeError:
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse=True
    )

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", encoder)
])

transformers = []

if numeric_features:
    transformers.append(
        ("numeric", numeric_pipeline, numeric_features)
    )

if categorical_features:
    transformers.append(
        ("categorical", categorical_pipeline,
         categorical_features)
    )

preprocessor = ColumnTransformer(
    transformers=transformers
)

# ==================================================
# STEP 6: ACTIVATION FUNCTIONS AND ARCHITECTURES
# ==================================================

activation_functions = [
    "relu",
    "tanh",
    "logistic",
    "identity"
]

architectures = [
    (64,),
    (64, 32),
    (128, 64, 32)
]

results = []

# ==================================================
# STEP 7: TRAIN AND COMPARE 12 MLP MODELS
# ==================================================

for activation in activation_functions:

    for architecture in architectures:

        print("\n" + "=" * 55)
        print("Activation function:", activation)
        print("Hidden layers:", architecture)

        mlp = MLPRegressor(
            hidden_layer_sizes=architecture,
            activation=activation,
            solver="adam",
            learning_rate_init=0.001,
            max_iter=100,
            batch_size=256,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42
        )

        model = Pipeline([
            ("preprocessor", preprocessor),
            ("mlp", mlp)
        ])

        start_time = time.time()

        model.fit(X_train, y_train)

        training_time = time.time() - start_time
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)

        rmse = np.sqrt(
            mean_squared_error(y_test, predictions)
        )

        r2 = r2_score(y_test, predictions)

        actual = y_test.to_numpy()

        mape = np.mean(
            np.abs(actual - predictions) / np.abs(actual)
        ) * 100

        results.append({
            "Activation": activation,
            "Architecture": str(architecture),
            "MAE": mae,
            "RMSE": rmse,
            "R2_Score": r2,
            "MAPE_Percent": mape,
            "Training_Time_Seconds": training_time,
            "Iterations": model.named_steps["mlp"].n_iter_
        })

        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R2 Score: {r2:.4f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"Training time: {training_time:.2f} seconds")

# ==================================================
# STEP 8: SAVE AND DISPLAY RESULTS
# ==================================================

if not results:
    raise RuntimeError("No models were trained.")

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "RMSE", ascending=True
).reset_index(drop=True)

print("\n========== FINAL MLP COMPARISON ==========")
print(results_df.round(4).to_string(index=False))

OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

results_path = os.path.join(
    OUTPUT_DIR, "mlp_activation_results.csv"
)

results_df.to_csv(results_path, index=False)

# ==================================================
# STEP 9: SAVE THE BEST MODEL AND PREDICTIONS
# ==================================================

# Initial ranking is by test RMSE. For final research,
# select using validation data and evaluate once on a
# separate, untouched test period.
best_row = results_df.iloc[0]
best_activation = best_row["Activation"]
best_architecture = ast.literal_eval(
    best_row["Architecture"]
)

print("\n========== BEST CONFIGURATION ==========")
print("Activation:", best_activation)
print("Architecture:", best_architecture)
print("MAE:", round(best_row["MAE"], 2))
print("RMSE:", round(best_row["RMSE"], 2))
print("R2:", round(best_row["R2_Score"], 4))
print("MAPE:", round(best_row["MAPE_Percent"], 2), "%")

best_model = Pipeline([
    ("preprocessor", preprocessor),
    ("mlp", MLPRegressor(
        hidden_layer_sizes=best_architecture,
        activation=best_activation,
        solver="adam",
        learning_rate_init=0.001,
        max_iter=100,
        batch_size=256,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42
    ))
])

best_model.fit(X_train, y_train)
best_predictions = best_model.predict(X_test)

prediction_results = pd.DataFrame({
    "Date": df.loc[test_mask, "Date"].to_numpy(),
    "Actual_Price": y_test.to_numpy(),
    "Predicted_Price": best_predictions,
    "Absolute_Error": np.abs(
        y_test.to_numpy() - best_predictions
    )
})

prediction_path = os.path.join(
    OUTPUT_DIR, "best_mlp_predictions.csv"
)

model_path = os.path.join(
    OUTPUT_DIR, "best_mlp_model.joblib"
)

prediction_results.to_csv(prediction_path, index=False)
joblib.dump(best_model, model_path)

print("\nAll files saved successfully:")
print(results_path)
print(prediction_path)
print(model_path)
