"""Aula 06 -- Avaliacao de agrupamento (metricas).

Metricas por k lado a lado, estabilidade entre sementes (ARI) e comparacao
K-Means x Ward.

Rodar: python src/aula06_metricas.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from _comum import SEED, carregar_bruto, limpar, matriz_modelo, salvar

print("=" * 70)
print("AULA 06 -- METRICAS DE AVALIACAO")
print("=" * 70)

df = limpar(carregar_bruto())
X = matriz_modelo(df)
K_RANGE = list(range(2, 11))

# ---------------------------------------------------------------------------
# 1) Metricas por k (linhas)
# ---------------------------------------------------------------------------
linhas = []
for k in K_RANGE:
    m = KMeans(n_clusters=k, n_init=50, random_state=SEED)
    lab = m.fit_predict(X)
    linhas.append({
        "k": k,
        "silhueta": silhouette_score(X, lab),
        "calinski_harabasz": calinski_harabasz_score(X, lab),
        "davies_bouldin": davies_bouldin_score(X, lab),
    })
met = pd.DataFrame(linhas)

# ---------------------------------------------------------------------------
# 2) Figura: metricas por k lado a lado
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].plot(met["k"], met["silhueta"], marker="o", color="#9bbb59")
axes[0].set_title("Silhueta (maior = melhor)")
axes[0].set_xlabel("k")
axes[0].grid(alpha=0.3)

axes[1].plot(met["k"], met["calinski_harabasz"], marker="o", color="#f79646")
axes[1].set_title("Calinski-Harabasz (maior = melhor)")
axes[1].set_xlabel("k")
axes[1].grid(alpha=0.3)

axes[2].plot(met["k"], met["davies_bouldin"], marker="o", color="#8064a2")
axes[2].set_title("Davies-Bouldin (menor = melhor)")
axes[2].set_xlabel("k")
axes[2].grid(alpha=0.3)
fig.suptitle("Evidencia da escolha de k: tres metricas lado a lado", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
salvar(fig, "aula06_01_metricas_por_k.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# 3) Estabilidade entre sementes (ARI) -- k=3
# ---------------------------------------------------------------------------
K_FINAL = 3
base = KMeans(n_clusters=K_FINAL, n_init=50, random_state=SEED).fit_predict(X)
aris = []
for seed in range(1, 11):
    lab_seed = KMeans(n_clusters=K_FINAL, n_init=50, random_state=seed).fit_predict(X)
    aris.append(adjusted_rand_score(base, lab_seed))
aris = np.array(aris)

print(f"Estabilidade (k={K_FINAL}) entre 10 sementes:")
print(f"  ARI medio  : {aris.mean():.4f}")
print(f"  ARI minimo : {aris.min():.4f}  (>= 0.8 = solucao estavel)")

fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(range(1, 11), aris, color="#4f81bd")
ax.axhline(0.8, color="red", linestyle="--", linewidth=1.2, label="limite 0,8")
ax.axhline(aris.mean(), color="green", linestyle=":", linewidth=1.2,
           label=f"media = {aris.mean():.3f}")
ax.set_xticks(range(1, 11))
ax.set_xlabel("Semente (random_state)")
ax.set_ylabel("ARI vs solucao base")
ax.set_ylim(0, 1.05)
ax.set_title(f"Estabilidade do K-Means (k={K_FINAL}) entre sementes")
ax.legend(fontsize=9)
fig.tight_layout()
salvar(fig, "aula06_02_estabilidade_sementes.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# 4) Comparacao K-Means x Ward (mesmo k)
# ---------------------------------------------------------------------------
Z = linkage(X, method="ward", metric="euclidean")
labels_ward = fcluster(Z, t=K_FINAL, criterion="maxclust")
ari_kw = adjusted_rand_score(base, labels_ward)
crosstab = pd.crosstab(base, labels_ward)

print(f"\nComparacao K-Means x Ward (k={K_FINAL}):")
print(f"  Silhueta K-Means : {silhouette_score(X, base):.4f}")
print(f"  Silhueta Ward    : {silhouette_score(X, labels_ward):.4f}")
print(f"  ARI K-Means x Ward: {ari_kw:.4f}")
print("\nTabela de concordancia (linhas = K-Means, colunas = Ward):")
print(crosstab.to_string())

print("\nAula 06 concluida.")
