"""Aula 03 -- Normalizacao/escalonamento e assimetria.

Demonstra por que escalonar e como (min-max, z-score, robust), o efeito na
distancia, e o efeito do log1p nas variaveis assimetricas.

Rodar: python src/aula03_escalonamento.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from _comum import FEATURES, LOG_FEATURES, carregar_bruto, limpar, salvar

print("=" * 70)
print("AULA 03 -- ESCALONAMENTO E ASSIMETRIA")
print("=" * 70)

df = limpar(carregar_bruto())

# base imputada (mediana) -- sem log ainda, para comparar escalas
X_imp = pd.DataFrame(
    SimpleImputer(strategy="median").fit_transform(df[FEATURES]),
    columns=FEATURES, index=df.index,
)

# ---------------------------------------------------------------------------
# 1) Figura: `income` em 4 reguas de escala
# ---------------------------------------------------------------------------
s = X_imp["income"]
bruta = s
minmax = (s - s.min()) / (s.max() - s.min())
zscore = (s - s.mean()) / s.std()
iqr = s.quantile(0.75) - s.quantile(0.25)
robust = (s - s.median()) / iqr

fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=False)
for ax, (nome, vals, cor) in zip(
    axes,
    [("Bruta (US$)", bruta, "#c0504d"),
     ("Min-max (0 a 1)", minmax, "#9bbb59"),
     ("Z-score (media 0, desvio 1)", zscore, "#4f81bd"),
     ("Robust (mediana 0, IQR 1)", robust, "#8064a2")],
):
    ax.hist(vals, bins=25, color=cor, edgecolor="white")
    ax.set_title(nome, fontsize=10)
    ax.tick_params(labelsize=8)
fig.suptitle("income em 4 reguas: a escolha do scaler muda a escala, nao a forma", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
salvar(fig, "aula03_01_scalers_comparacao.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# 2) Figura: efeito da escala na distancia (2 paises)
# ---------------------------------------------------------------------------
# escolher o pais mais rico e o mais pobre
rico = X_imp["income"].idxmax()
pobre = X_imp["income"].idxmin()
nome_rico = df.loc[rico, "country"]
nome_pobre = df.loc[pobre, "country"]

a_b = X_imp.loc[rico, FEATURES] - X_imp.loc[pobre, FEATURES]
contrib_bruta = (a_b ** 2)

# padronizado por z-score
Z = (X_imp - X_imp.mean()) / X_imp.std()
a_bz = Z.loc[rico, FEATURES] - Z.loc[pobre, FEATURES]
contrib_z = (a_bz ** 2)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
axes[0].bar(FEATURES, contrib_bruta.values, color="#c0504d")
axes[0].set_title(f"Contribuicao (a-b)^2 SEM escalonar\n{nome_rico} x {nome_pobre}", fontsize=11)
axes[0].set_ylabel("Contribuicao para a distancia")
axes[0].tick_params(axis="x", rotation=20, labelsize=9)

axes[1].bar(FEATURES, contrib_z.values, color="#4f81bd")
axes[1].set_title(f"Contribuicao (a-b)^2 com Z-SCORE\n{nome_rico} x {nome_pobre}", fontsize=11)
axes[1].set_ylabel("Contribuicao para a distancia")
axes[1].tick_params(axis="x", rotation=20, labelsize=9)
fig.suptitle("Sem escala, 'income' domina a distancia; escalonando, fica equilibrado", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
salvar(fig, "aula03_02_efeito_escala_distancia.png")
plt.close(fig)

print(f"Sem escalonar, income responde por "
      f"{100 * contrib_bruta['income'] / contrib_bruta.sum():.1f}% da distancia bruta.")
print(f"Com z-score, income responde por "
      f"{100 * contrib_z['income'] / contrib_z.sum():.1f}%.")

# ---------------------------------------------------------------------------
# 3) Figura: assimetria antes x depois do log
# ---------------------------------------------------------------------------
print("\nAssimetria (skew) antes e depois do log1p:")
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for j, col in enumerate(LOG_FEATURES):
    s0 = X_imp[col]
    s1 = np.log1p(s0.clip(lower=0))

    ax = axes[0, j]
    ax.hist(s0, bins=25, color="#c0504d", edgecolor="white")
    ax.set_title(f"{col} -- BRUTO (skew={s0.skew():.2f})", fontsize=10)
    ax.tick_params(labelsize=8)

    ax = axes[1, j]
    ax.hist(s1, bins=25, color="#4f81bd", edgecolor="white")
    ax.set_title(f"{col} -- log1p (skew={s1.skew():.2f})", fontsize=10)
    ax.tick_params(labelsize=8)

    print(f"  {col:14s}: skew antes = {s0.skew():+.3f} | depois = {s1.skew():+.3f}")

fig.suptitle("Assimetria: log1p muda a FORMA (escalar nao resolve isso)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
salvar(fig, "aula03_03_assimetria_log.png")
plt.close(fig)

print("\nAula 03 concluida.")
