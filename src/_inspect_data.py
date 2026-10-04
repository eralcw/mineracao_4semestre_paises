import sys
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

df = pd.read_csv("data/paises_help_international.csv")
print("shape:", df.shape)
print("cols:", df.columns.tolist())
print("\nhead:")
print(df.head())
print("\ndtypes:")
print(df.dtypes)

num = df.select_dtypes(include=np.number).columns.tolist()

print("\n--- sentinelas 99999 / 999 / 100000 ---")
for col in num:
    for v in (99999, 100000, 999):
        c = int((df[col] == v).sum())
        if c:
            print(f"{col} == {v}: {c}")

print("\n--- valores impossiveis ---")
for idx, row in df.iterrows():
    for col in ["acesso_agua_pct","alfabetizacao_pct","internet_pct","extrema_pobreza_pct","child_mort","life_expec","total_fer"]:
        if col in df.columns and pd.notna(row[col]):
            val = row[col]
            if col in ("acesso_agua_pct","alfabetizacao_pct","internet_pct","extrema_pobreza_pct") and (val < 0 or val > 100):
                print(f"{row['country']:15s} {col:22s} = {val}")
            if col == "child_mort" and val < 0:
                print(f"{row['country']:15s} {col:22s} = {val}")
            if col == "life_expec" and (val <= 0 or val > 120):
                print(f"{row['country']:15s} {col:22s} = {val}")
            if col == "total_fer" and (val < 0 or val > 20):
                print(f"{row['country']:15s} {col:22s} = {val}")

print("\n--- faltantes reais ---")
miss = df.isna().sum()
print(miss[miss > 0])

print("\n--- valores extremos gdpp/income/inflation ---")
print(df[["country","gdpp","income","inflation"]].sort_values("gdpp").head(3).to_string())
print(df[["country","gdpp","income","inflation"]].sort_values("gdpp").tail(3).to_string())
