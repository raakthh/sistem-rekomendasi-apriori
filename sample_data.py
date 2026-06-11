import pandas as pd

print("Membaca dataset...")

df = pd.read_excel(
    "retail_recommender/data/online_retail_II.xlsx",
    sheet_name="Year 2010-2011"
)

print("Total data:", len(df))

sample_df = df.sample(
    n=20000,
    random_state=42
)

sample_df.to_excel(
    "retail_recommender/data/online_retail_sample_20k.xlsx",
    index=False
)

print("Berhasil membuat dataset 20.000 baris")