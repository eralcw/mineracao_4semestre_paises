"""Aula 02 -- Limpeza e qualidade dos dados.

Sentinela (99999/100000/999) -> NaN, valores impossiveis -> NaN, e comparacao
ANTES x DEPOIS para as 4 FEATURES. Gera figuras em outputs/figuras/.

Rodar: python src/aula02_limpeza.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _comum import FEATURES, SENTINELAS, carregar_bruto, colunas_numericas, limpar, salvar, OUT_DIR

print("=" * 70)
print("AULA 02 -- LIMPEZA E QUALIDADE DOS DADOS")
print("=" * 70)

# ---------------------------------------------------------------------------
# 1) Base bruta e diagnostico
# ---------------------------------------------------------------------------
df_raw = carregar_bruto()
print(f"Base bruta: {df_raw.shape[0]} linhas x {df_raw.shape[1]} colunas")
print(f"Duplicatas: {df_raw.duplicated().sum()}")
print(f"country e identificador unico: {df_raw['country'].nunique() == len(df_raw)}\n")

num_cols = colunas_numericas(df_raw)

# faltantes reais (NaN) por coluna
faltantes_reais = df_raw[num_cols].isna().sum()
print("Faltantes reais (NaN) por coluna:")
print(faltantes_reais[faltantes_reais > 0].to_string())

# sentinelas encontradas
print("\nSentinelas encontradas (candidatos a 'sem dado'):")
for col in num_cols:
    for v in SENTINELAS:
        qtd = int((df_raw[col] == v).sum())
        if qtd:
            print(f"  {col:24s} == {v} : {qtd}")

# outliers grosseiros (impossiveis) -- exemplos
print("\nExemplos de valores impossiveis detectados por regra de dominio (viram NaN).")

# ---------------------------------------------------------------------------
# 2) Limpeza
# ---------------------------------------------------------------------------
df = limpar(df_raw)
n_antes = df_raw[num_cols].isna().sum().sum()
n_depois = df[num_cols].isna().sum().sum()
print(f"\nTotal de NaN na base bruta (so faltantes reais): {n_antes}")
print(f"Total de NaN na base limpa (reais + sentinelas + impossiveis): {n_depois}")
print(f"Valores convertidos para NaN na limpeza: {n_depois - n_antes}")

# salvar base limpa para inspecao
OUT_DIR.mkdir(parents=True, exist_ok=True)
caminho_limpo = OUT_DIR / "dados_limpos.csv"
df.to_csv(caminho_limpo, index=False, encoding="utf-8-sig")
print(f"[csv] {caminho_limpo}")

# ---------------------------------------------------------------------------
# 3) Figura: faltantes reais na base bruta
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 5))
fal = faltantes_reais[faltantes_reais > 0].sort_values(ascending=False)
ax.bar(range(len(fal)), fal.values, color="#3b6ea5")
ax.set_xticks(range(len(fal)))
ax.set_xticklabels(fal.index, rotation=60, ha="right", fontsize=9)
ax.set_ylabel("Quantidade de valores faltantes (NaN)")
ax.set_title("Faltantes reais (vazios) por coluna -- base bruta")
for i, v in enumerate(fal.values):
    ax.text(i, v + 0.03, str(v), ha="center", fontsize=8)
fig.tight_layout()
salvar(fig, "aula02_01_faltantes_bruto.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# 4) Figura: histogramas ANTES x DEPOIS das 4 FEATURES
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for j, col in enumerate(FEATURES):
    # linha de cima: bruto
    ax = axes[0, j]
    ax.hist(df_raw[col].dropna(), bins=20, color="#c0504d", edgecolor="white")
    ax.set_title(f"{col}\n(bruto)", fontsize=10)
    ax.tick_params(labelsize=8)
    # linha de baixo: limpo
    ax = axes[1, j]
    ax.hist(df[col].dropna(), bins=20, color="#4f81bd", edgecolor="white")
    ax.set_title(f"{col}\n(limpo)", fontsize=10)
    ax.tick_params(labelsize=8)
fig.suptitle("4 indicadores: base bruta (topo) x base limpa (base)", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
salvar(fig, "aula02_02_antes_depois_hist.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# 5) Figura: boxplots ANTES x DEPOIS das 4 FEATURES
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for j, col in enumerate(FEATURES):
    ax = axes[0, j]
    ax.boxplot(df_raw[col].dropna(), vert=True, patch_artist=True,
               boxprops=dict(facecolor="#f2b3b0"))
    ax.set_title(f"{col}\n(bruto)", fontsize=10)
    ax.tick_params(labelsize=8)
    ax = axes[1, j]
    ax.boxplot(df[col].dropna(), vert=True, patch_artist=True,
               boxprops=dict(facecolor="#b8cbe6"))
    ax.set_title(f"{col}\n(limpo)", fontsize=10)
    ax.tick_params(labelsize=8)
fig.suptitle("Boxplots -- base bruta (topo, sentinela 'achata' a escala) x base limpa (base)",
             fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
salvar(fig, "aula02_03_antes_depois_box.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# 6) Figura: efeito da sentinela em `health` e `income`
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for j, col in enumerate(["health", "income"]):
    ax = axes[0, j]
    vals = df_raw[col].dropna()
    ax.scatter(range(len(vals)), vals.values, s=18, color="#c0504d")
    # destacar sentinelas
    sent = vals[vals.isin(SENTINELAS)]
    for idx in sent.index:
        ax.annotate(f"{vals.loc[idx]:.0f}", (list(vals.index).index(idx), vals.loc[idx]),
                    textcoords="offset points", xytext=(0, 6), ha="center",
                    fontsize=9, color="red")
    ax.set_title(f"{col} -- BRUTO (sentinela 99999/999 visivel)", fontsize=10)
    ax.set_ylabel(col)
    ax.tick_params(labelsize=8)

    ax = axes[1, j]
    vals2 = df[col].dropna()
    ax.scatter(range(len(vals2)), vals2.values, s=18, color="#4f81bd")
    ax.set_title(f"{col} -- LIMPO (sentinela -> NaN)", fontsize=10)
    ax.set_ylabel(col)
    ax.tick_params(labelsize=8)
fig.suptitle("Efeito da sentinela: 'sem dado' fingindo ser valor real", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
salvar(fig, "aula02_04_efeito_sentinela.png")
plt.close(fig)

print("\nAula 02 concluida.")
