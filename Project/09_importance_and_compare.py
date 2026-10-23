import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

TARGET_COL = "Modal Price"     
SAMPLE_ROWS = 100000           
DROP_COLS = ["State", "Commodity Group", "Crop_Data_Source", "Duration_Definition",
             "Date", "Price Unit", "Arrival Unit"]
os.makedirs("outputs", exist_ok=True)

files = glob.glob("Dataset/Maharashtra_Agri*")
if not files:
    raise FileNotFoundError("No file starting with 'Maharashtra_Agri' found in Dataset/")
path = files[0]
print("Loading:", path)
df = pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_excel(path)

if TARGET_COL not in df.columns:
    raise KeyError(f"'{TARGET_COL}' not found. Your columns are:\n{list(df.columns)}")

df = df.dropna(subset=[TARGET_COL])

if "Price Unit" in df.columns:
    df = df[df["Price Unit"] == "Rs./Quintal"]

if SAMPLE_ROWS and len(df) > SAMPLE_ROWS:
    df = df.sample(SAMPLE_ROWS, random_state=42)

df = df.drop(columns=DROP_COLS, errors="ignore")
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]
print("Data shape:", df.shape, "\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def build_model(model, columns):
    num_cols = X[columns].select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in columns if c not in num_cols]
    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore",
                                 max_categories=50, sparse_output=False)),
        ]), cat_cols),
    ])
    return Pipeline([("prep", prep), ("model", model)])

def get_models():
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": HistGradientBoostingRegressor(random_state=42),
    }

def compare(columns, title):
    print("=" * 60, f"\n{title}\n" + "=" * 60)
    rows, fitted = [], {}
    for name, model in get_models().items():
        print(f"Training {name}...")
        pipe = build_model(model, columns)
        pipe.fit(X_train[columns], y_train)
        pred = pipe.predict(X_test[columns])
        rows.append([name,
                     round(mean_absolute_error(y_test, pred), 2),
                     round(np.sqrt(mean_squared_error(y_test, pred)), 2),
                     round(r2_score(y_test, pred), 4)])
        fitted[name] = pipe
    res = pd.DataFrame(rows, columns=["Model", "MAE", "RMSE", "R2"])
    print(res.sort_values("R2", ascending=False).to_string(index=False), "\n")
    return res, fitted

all_cols = X.columns.tolist()
res_all, fitted_all = compare(all_cols, "ALL FEATURES")
best_name = res_all.sort_values("R2", ascending=False).iloc[0]["Model"]
print("Best model:", best_name, "\n")

n = min(2000, len(X_test))                   
Xs = X_test.sample(n, random_state=42)
ys = y_test.loc[Xs.index]

perm = permutation_importance(fitted_all[best_name], Xs, ys,
                              n_repeats=5, random_state=42,
                              scoring="r2", n_jobs=1)   # n_jobs=1: no temp files needed
perm_df = pd.DataFrame({
    "feature": all_cols,
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std,
}).sort_values("importance_mean", ascending=False)
print("PERMUTATION IMPORTANCE (drop in R2 when shuffled)")
print(perm_df.to_string(index=False), "\n")
perm_df.to_csv("outputs/permutation_importance.csv", index=False)

plt.figure(figsize=(8, 6))
p = perm_df.head(20).iloc[::-1]
plt.barh(p["feature"], p["importance_mean"], xerr=p["importance_std"])
plt.xlabel("Drop in R2 when feature is shuffled")
plt.title(f"Permutation importance - {best_name}")
plt.tight_layout()
plt.savefig("outputs/permutation_importance.png", dpi=150)

keep = perm_df[(perm_df["importance_mean"] - perm_df["importance_std"]) > 0]["feature"].tolist()
drop = [c for c in all_cols if c not in keep]
if not keep:
    keep = perm_df.head(3)["feature"].tolist()
    drop = [c for c in all_cols if c not in keep]
print("KEEP:", keep)
print("DROP:", drop, "\n")
pd.DataFrame({"keep": pd.Series(keep)}).to_csv("outputs/keep_features.csv", index=False)
pd.DataFrame({"drop": pd.Series(drop)}).to_csv("outputs/drop_features.csv", index=False)

# ------------------------------------------------------------
# STEP 6: Re-train with only the KEPT features and compare
# ------------------------------------------------------------
res_keep, _ = compare(keep, "ONLY KEPT FEATURES")

final = res_all.merge(res_keep, on="Model", suffixes=("_all", "_kept"))
print("FINAL COMPARISON\n", final.to_string(index=False))
final.to_csv("outputs/model_comparison.csv", index=False)
print("\nSaved everything in the outputs/ folder.")