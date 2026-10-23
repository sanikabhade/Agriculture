import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROJECT_FOLDER = Path(__file__).resolve().parent

DATA_FILE = (
    PROJECT_FOLDER
    / "Dataset"
    / "Maharashtra_Agri_Master_2024_Jul2025.csv"
)

OUTPUT_FOLDER = PROJECT_FOLDER / "outputs"
OUTPUT_FOLDER.mkdir(exist_ok=True)

df = pd.read_csv(DATA_FILE)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

sns.set_theme(style="whitegrid")

top_crops = df["Commodity"].value_counts().head(10)

plt.figure(figsize=(11, 6))
top_crops.sort_values().plot(kind="barh")

plt.title("Top 10 Crops by Number of Records")
plt.xlabel("Number of Records")
plt.ylabel("Crop")

plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "01_top_crops.png", dpi=150)
plt.show()

price = pd.to_numeric(df["Modal Price"], errors="coerce")
price_limit = price.quantile(0.99)

plt.figure(figsize=(10, 6))
sns.histplot(price[price <= price_limit].dropna(), bins=50)

plt.title("Modal Price Distribution (Up to 99th Percentile)")
plt.xlabel("Modal Price (reported units)")
plt.ylabel("Number of Records")

plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "02_price_distribution.png", dpi=150)
plt.show()

print("99th percentile of modal price:", price_limit)
print("Records above 99th percentile:", (price > price_limit).sum())


onion = df[df["Commodity"].str.strip().str.lower() == "onion"].copy()

onion["Modal Price"] = pd.to_numeric(
    onion["Modal Price"], errors="coerce"
)

daily_onion_price = (
    onion.groupby("Date")["Modal Price"]
    .median()
    .dropna()
    .sort_index()
)

plt.figure(figsize=(12, 6))
plt.plot(daily_onion_price.index, daily_onion_price.values)

plt.title("Daily Median Onion Modal Price Across Markets")
plt.xlabel("Date")
plt.ylabel("Median Modal Price (reported units)")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "03_onion_price_trend.png", dpi=150)
plt.show()

numeric_columns = [
    "Modal Price",
    "Arrival Quantity",
    "Temperature_C",
    "Min_Temperature_C",
    "Max_Temperature_C",
    "Precipitation_mm",
    "Relative_Humidity_Percent",
    "pH_0_30",
    "Organic_Carbon_g_kg_0_30",
    "Nitrogen_g_kg_0_30",
    "Clay_percent_0_30",
    "CEC_cmolc_kg_0_30"
]

numeric_df = df[numeric_columns].apply(
    pd.to_numeric, errors="coerce"
)

correlation = numeric_df.corr()

plt.figure(figsize=(12, 9))
sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

plt.title("Correlation Between Numerical Features")
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "04_correlation_heatmap.png", dpi=150)
plt.show()

print("\nPrice units in dataset:")
print(df["Price Unit"].value_counts(dropna=False))

print("\nEDA graphs saved in:", OUTPUT_FOLDER)
