"""Aula 05 -- K-Means e escolha de k.

Cotovelo (inercia) + silhueta, e a dispersao dos 3 clusters finais (k=3).

Rodar: python src/aula05_kmeans.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from _comum import FEATURES, SEED, carregar_bruto, limpar, matriz_modelo, salvar

print("=" * 70)
print("AULA 05 -- K-MEANS E ESCOLHA DE k")
print("=" * 70)

df = limpar(carregar_bruto())
X = matriz_modelo(df)
K_RANGE = list(range(2, 11))

# ---------------------------------------------------------------------------
# 1) Metricas por k
# ---------------------------------------------------------------------------
linhas = []
for k in K_RANGE:
    m = KMeans(n_clusters=k, n_init=50, random_state=SEED)
    lab = m.fit_predict(X)
    linhas.append({
        "k": k,
        "inercia": m.inertia_,
        "silhueta": silhouette_score(X, lab),
        "calinski_harabasz": calinski_harabasz_score(X, lab),
        "davies_bouldin": davies_bouldin_score(X, lab),
    })
met = pd.DataFrame(linhas)
print("Metricas por k:")
print(met.round(4).to_string(index=False))

# ---------------------------------------------------------------------------
# 2) Figura: cotovelo + silhueta
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(met["k"], met["inercia"], marker="o", color="#4f81bd")
axes[0].set_title("Cotovelo -- Inercia por k")
axes[0].set_xlabel("k")
axes[0].set_ylabel("Inercia (SSE)")
axes[0].grid(alpha=0.3)

axes[1].plot(met["k"], met["silhueta"], marker="o", color="#9bbb59")
axes[1].set_title("Silhueta por k (maior = melhor)")
axes[1].set_xlabel("k")
axes[1].set_ylabel("Silhueta")
axes[1].grid(alpha=0.3)
fig.suptitle("Escolha de k: cotovelo e silhueta", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
salvar(fig, "aula05_01_cotovelo_silhueta.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# 3) Figura: Calinski-Harabasz e Davies-Bouldin
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(met["k"], met["calinski_harabasz"], marker="o", color="#f79646")
axes[0].set_title("Calinski-Harabasz por k (maior = melhor)")
axes[0].set_xlabel("k")
axes[0].set_ylabel("CH")
axes[0].grid(alpha=0.3)

axes[1].plot(met["k"], met["davies_bouldin"], marker="o", color="#8064a2")
axes[1].set_title("Davies-Bouldin por k (menor = melhor)")
axes[1].set_xlabel("k")
axes[1].set_ylabel("DB")
axes[1].grid(alpha=0.3)
fig.suptitle("Metricas de apoio para a escolha de k", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
salvar(fig, "aula05_02_ch_db.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# 4) K-Means final (k=3)
# ---------------------------------------------------------------------------
K_FINAL = 3
kmeans = KMeans(n_clusters=K_FINAL, n_init=50, random_state=SEED)
labels = kmeans.fit_predict(X)

print(f"\n=== K-Means final (k={K_FINAL}) ===")
print(f"Silhueta          : {silhouette_score(X, labels):.4f}")
print(f"Calinski-Harabasz : {calinski_harabasz_score(X, labels):.4f}")
print(f"Davies-Bouldin    : {davies_bouldin_score(X, labels):.4f}")
print("Tamanho dos clusters:")
print(pd.Series(labels).value_counts().sort_index().to_string())

# tabela de perfis (mediana por cluster, dado limpo)
perfil = df[["country"]].copy()
perfil[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())
perfil["cluster"] = labels
perfil_med = perfil.groupby("cluster")[FEATURES].median().round(2)
perfil_med["n_paises"] = perfil.groupby("cluster").size()
print("\nPerfil mediano por cluster (dado limpo):")
print(perfil_med.to_string())

# ---------------------------------------------------------------------------
# 5) Figura: dispersao dos clusters
# ---------------------------------------------------------------------------
pares = [("child_mort", "income"), ("acesso_agua_pct", "alfabetizacao_pct")]
cores = {0: "#4f81bd", 1: "#9bbb59", 2: "#c0504d"}
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, (xcol, ycol) in zip(axes, pares):
    for c in sorted(np.unique(labels)):
        mask = labels == c
        ax.scatter(df.loc[mask, xcol], df.loc[mask, ycol], s=28, alpha=0.8,
                   color=cores.get(c, "gray"), label=f"Cluster {c} (n={mask.sum()})")
    ax.set_xlabel(xcol, fontsize=10)
    ax.set_ylabel(ycol, fontsize=10)
    ax.set_title(f"{xcol} x {ycol}", fontsize=11)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
fig.suptitle("K-Means (k=3): paises coloridos por perfil", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
salvar(fig, "aula05_03_clusters_dispersao.png")
plt.close(fig)

print("\nAula 05 concluida.")
