"""Aula 04 -- Distancia, correlacao e EDA.

EDA da base limpa + heatmap de correlacao (base da escolha dos 4 indicadores).

Rodar: python src/aula04_eda_correlacao.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _comum import FEATURES, carregar_bruto, colunas_numericas, limpar, salvar

print("=" * 70)
print("AULA 04 -- EDA E CORRELACAO")
print("=" * 70)

df = limpar(carregar_bruto())
num_cols = colunas_numericas(df)

# ---------------------------------------------------------------------------
# 1) Figura: heatmap de correlacao (Pearson, base limpa)
# ---------------------------------------------------------------------------
corr = df[num_cols].corr()

fig, ax = plt.subplots(figsize=(13, 11))
im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
ax.set_xticks(range(len(num_cols)))
ax.set_xticklabels(num_cols, rotation=70, ha="right", fontsize=8)
ax.set_yticks(range(len(num_cols)))
ax.set_yticklabels(num_cols, fontsize=8)
ax.set_title("Matriz de correlacao de Pearson -- base limpa")
fig.colorbar(im, ax=ax, label="Correlacao")
fig.tight_layout()
salvar(fig, "aula04_01_heatmap_correlacao.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# 2) Top 15 pares mais correlacionados (justificam rejeitar redundancias)
# ---------------------------------------------------------------------------
pares = (
    corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    .stack().sort_values(key=np.abs, ascending=False)
)
print("Top 15 pares mais correlacionados (|r|):")
print(pares.head(15).round(3).to_string())

# ---------------------------------------------------------------------------
# 3) Figura: dispersao entre pares das 4 FEATURES
# ---------------------------------------------------------------------------
logger = np.errstate(invalid="ignore")
pares_scatter = [
    ("child_mort", "income"),
    ("child_mort", "acesso_agua_pct"),
    ("child_mort", "alfabetizacao_pct"),
    ("acesso_agua_pct", "alfabetizacao_pct"),
]
fig, axes = plt.subplots(2, 2, figsize=(13, 11))
for ax, (x, y) in zip(axes.ravel(), pares_scatter):
    ax.scatter(df[x], df[y], s=22, alpha=0.7, color="#4f81bd")
    r = df[x].corr(df[y])
    ax.set_xlabel(x, fontsize=10)
    ax.set_ylabel(y, fontsize=10)
    ax.set_title(f"{x} x {y} (r={r:.2f})", fontsize=10)
    ax.tick_params(labelsize=8)
fig.suptitle("Relacoes entre os 4 indicadores (base limpa)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
salvar(fig, "aula04_02_dispersao_features.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# 4) Figura: histogramas das 4 FEATURES (forma)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
for ax, col in zip(axes, FEATURES):
    ax.hist(df[col].dropna(), bins=20, color="#9bbb59", edgecolor="white")
    ax.set_title(f"{col}\n(skew={df[col].skew():.2f})", fontsize=10)
    ax.tick_params(labelsize=8)
fig.suptitle("Distribuicao dos 4 indicadores (base limpa) -- note a assimetria", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.93])
salvar(fig, "aula04_03_histogramas_limpo.png")
plt.close(fig)

print("\nAula 04 concluida.")
