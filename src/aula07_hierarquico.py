"""Aula 07 -- Agrupamento hierarquico (Ward).

Dendrograma e comparacao com o K-Means (crosstab + ARI).

Rodar: python src/aula07_hierarquico.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from _comum import SEED, carregar_bruto, limpar, matriz_modelo, salvar

print("=" * 70)
print("AULA 07 -- AGRUPAMENTO HIERARQUICO (WARD)")
print("=" * 70)

df = limpar(carregar_bruto())
X = matriz_modelo(df)
K_FINAL = 3

# ---------------------------------------------------------------------------
# 1) Linkage de Ward
# ---------------------------------------------------------------------------
Z = linkage(X, method="ward", metric="euclidean")

# ---------------------------------------------------------------------------
# 2) Figura: dendrograma
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(16, 7))
dendrogram(Z, labels=df["country"].values, leaf_rotation=90, leaf_font_size=6, ax=ax)
ax.set_title("Dendrograma -- Ward  (ATENCAO: a altura NAO e o numero de grupos k)")
ax.set_ylabel("Altura de fusao (distancia de Ward)")
fig.tight_layout()
salvar(fig, "aula07_01_dendrograma.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# 3) Corte por numero de grupos (criterion="maxclust", nunca "distance")
# ---------------------------------------------------------------------------
labels_ward = fcluster(Z, t=K_FINAL, criterion="maxclust")
print(f"Ward cortado em {K_FINAL} grupos (criterion='maxclust').")
print("Tamanho dos grupos (Ward):")
print(pd.Series(labels_ward).value_counts().sort_index().to_string())

# ---------------------------------------------------------------------------
# 4) K-Means para comparar
# ---------------------------------------------------------------------------
labels_km = KMeans(n_clusters=K_FINAL, n_init=50, random_state=SEED).fit_predict(X)
ari = adjusted_rand_score(labels_km, labels_ward)
crosstab = pd.crosstab(labels_km, labels_ward)

print(f"\nSilhueta K-Means : {silhouette_score(X, labels_km):.4f}")
print(f"Silhueta Ward    : {silhouette_score(X, labels_ward):.4f}")
print(f"ARI K-Means x Ward: {ari:.4f}")
print("\nTabela de concordancia (linhas = K-Means, colunas = Ward):")
print(crosstab.to_string())

# ---------------------------------------------------------------------------
# 5) Figura: heatmap da concordancia + ARI
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(crosstab.values, cmap="Blues")
ax.set_xticks(range(crosstab.shape[1]))
ax.set_xticklabels([f"Ward {c}" for c in crosstab.columns])
ax.set_yticks(range(crosstab.shape[0]))
ax.set_yticklabels([f"K-Means {r}" for r in crosstab.index])
for i in range(crosstab.shape[0]):
    for j in range(crosstab.shape[1]):
        ax.text(j, i, int(crosstab.values[i, j]), ha="center", va="center",
                color="black", fontsize=11)
ax.set_title(f"Concordancia K-Means x Ward (k={K_FINAL})\nARI = {ari:.3f}")
fig.colorbar(im, ax=ax, label="Numero de paises")
fig.tight_layout()
salvar(fig, "aula07_02_comparacao_ward.png")
plt.close(fig)

print("\nAula 07 concluida.")
