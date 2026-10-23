import pandas as pd
DATA_FILE = r"Dataset\Maharashtra_Agri_Master_2024_Jul2025.csv"

df = pd.read_csv(DATA_FILE)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

print("========== BASIC INFORMATION ==========")

print("\nDataset shape:")
print(df.shape)

print("\nData types:")
print(df.dtypes)

print("\nMissing values in each column:")
print(df.isnull().sum())

print("\nTotal missing values:")
print(df.isnull().sum().sum())

print("\nTotal duplicate rows:")
print(df.duplicated().sum())

print("\nDate range:")
print("Starting date:", df["Date"].min())
print("Ending date:", df["Date"].max())

print("\n========== NUMERICAL ANALYSIS ==========")

print(df[[
    "Modal Price",
    "Arrival Quantity",
    "Temperature_C",
    "Precipitation_mm",
    "Relative_Humidity_Percent"
]].describe())

print("\n========== CROP ANALYSIS ==========")

print("Number of unique crops:")
print(df["Commodity"].nunique())

print("\nNumber of unique districts:")
print(df["District"].nunique())

print("\nNumber of unique markets:")
print(df["Market"].nunique())

print("\nTop 10 crops by number of records:")
print(df["Commodity"].value_counts().head(10))
