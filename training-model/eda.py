import pandas as pd

# Baca dataset yang baru diunduh
df = pd.read_csv("sosmed_engagement.csv")

print("--- INFO DATASET ---")
print(df.info())

print("\n--- CEK DATA KOSONG ---")
print(df.isnull().sum())

print("\n--- 5 BARIS PERTAMA ---")
print(df.head())