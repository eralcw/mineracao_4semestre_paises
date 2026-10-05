"""Figuras do relatorio -- uma resposta direta por pergunta (FIGURAS_DO_RELATORIO.md).

Reaproveita a logica/dados das aulas e gera as figuras Q1..Q9 em outputs/figuras/,
com titulos AFIRMATIVOS e regras de clareza (mesma escala antes x depois, fontes
maiores, grid leve, unidades, anotacao de valores-chave).

NAO muda numeros, limpeza, indicadores, k=3, escalonamento nem distancia.
Mantem as cores atuais das figuras das aulas.

Rodar: python src/figuras_relatorio.py
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
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import RobustScaler

from _comum import (
    FEATURES,
    LOG_FEATURES,
    SEED,
    carregar_bruto,
    colunas_numericas,
    limpar,
    matriz_modelo,
    salvar,
)

# ---------------------------------------------------------------------------
# Paleta ATUAL das figuras das aulas (NAO alterar)
# ---------------------------------------------------------------------------
AZUL, VERM, VERDE, LARANJA, ROXO = "#4f81bd", "#c0504d", "#9bbb59", "#f79646", "#8064a2"

# Estilo padrao de clareza
plt.rcParams.update({
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})


def grade(ax):
    ax.grid(alpha=0.3, linestyle=":")


# ---------------------------------------------------------------------------
# Dados e pipeline (identicos ao notebook / aulas)
# ---------------------------------------------------------------------------
print("=" * 70)
print("FIGURAS DO RELATORIO (Q1..Q9)")
print("=" * 70)

df_raw = carregar_bruto()
df = limpar(df_raw)
num_cols = colunas_numericas(df)

# base imputada por mediana (dado limpo) -- igual ao notebook
X_imp = pd.DataFrame(
    SimpleImputer(strategy="median").fit_transform(df[FEATURES]),
    columns=FEATURES, index=df.index,
)
# base do modelo (log + RobustScaler)
X = matriz_modelo(df)
K_FINAL = 3


# ===========================================================================
# Q1 -- QUAIS INDICADORES USOU?
# ===========================================================================
print("\n[Q1] indicadores")

corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(15, 12))
im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
ax.set_xticks(range(len(num_cols)))
ax.set_xticklabels(num_cols, rotation=70, ha="right", fontsize=10)
ax.set_yticks(range(len(num_cols)))
ax.set_yticklabels(num_cols, fontsize=10)
ax.set_title(
    "Q1 - Quais indicadores? Heatmap de correlacao (Pearson, base limpa).\n"
    "Escolhemos: child_mort, income, acesso_agua_pct, alfabetizacao_pct",
    fontsize=14,
)
fig.colorbar(im, ax=ax, label="Correlacao")
fig.tight_layout()
salvar(fig, "Q1_indicadores_heatmap.png")
plt.close(fig)

pares_rejeitados = [
    ("acesso_agua_pct", "acesso_esgoto_pct", 0.960),
    ("acesso_agua_pct", "acesso_energia_pct", 0.957),
    ("acesso_agua_pct", "internet_pct", 0.950),
    ("income", "gdpp", 0.891),
    ("child_mort", "life_expec", -0.885),
    ("child_mort", "total_fer", 0.848),
]
labels = [f"{a} x {b}" for a, b, _ in pares_rejeitados]
valores = [abs(r) for _, _, r in pares_rejeitados]

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
ax = axes[0]
ax.barh(range(len(labels)), valores, color=VERM)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=10)
ax.invert_yaxis()
ax.axvline(0.8, color="black", linestyle="--", linewidth=1.2,
           label="limiar de redundancia |r|=0,8")
for i, v in enumerate(valores):
    ax.text(v + 0.01, i, f"{v:.2f}".replace(".", ","), va="center", fontsize=10)
ax.set_xlim(0, 1.05)
ax.set_xlabel("|correlacao| (Pearson)")
ax.set_title("Pares REJEITADOS por redundancia", fontsize=12)
ax.legend(loc="lower right")
grade(ax)

ax = axes[1]
ax.axis("off")
ax.set_title("Indicadores ESCOLHIDOS (4 facetas)", fontsize=12)
escolhidos = [
    ("child_mort", "Saude infantil"),
    ("income", "Renda / capacidade economica"),
    ("acesso_agua_pct", "Infraestrutura / saneamento"),
    ("alfabetizacao_pct", "Educacao"),
]
txt = "\n".join(f"  {i+1}.  {nome:18s} - {faceta}"
                for i, (nome, faceta) in enumerate(escolhidos))
ax.text(0.02, 0.62, txt, fontsize=13, family="monospace", va="top")
ax.text(
    0.02, 0.30,
    "Criterio: descartar pares quase identicos (|r| >= 0,8)\n"
    "e manter 4 facetas que orientam estrategias diferentes.\n\n"
    "Ressalva honesta: os 4 sao COLINEARES entre si\n"
    "(VIF ate 8,8; 1o componente = 82,7% da variancia),\n"
    "sao facetas de um mesmo fator de desenvolvimento.",
    fontsize=11, va="top",
)
fig.suptitle("Q1 - Escolha dos indicadores: por que sobraram esses 4", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.95])
salvar(fig, "Q1_indicadores_escolhidos.png")
plt.close(fig)


# ===========================================================================
# Q2 -- QUAIS METODOS PARA PREENCHIMENTO DOS FALTANTES?
# ===========================================================================
print("\n[Q2] faltantes")

faltantes_reais = df_raw[num_cols].isna().sum()
fig, ax = plt.subplots(figsize=(12, 6))
fal = faltantes_reais[faltantes_reais > 0].sort_values(ascending=False)
ax.bar(range(len(fal)), fal.values, color=AZUL)
ax.set_xticks(range(len(fal)))
ax.set_xticklabels(fal.index, rotation=60, ha="right", fontsize=10)
ax.set_ylabel("Numero de valores faltantes (NaN)")
n_total = int(fal.sum())
ax.set_title(
    f"Q2 - Faltantes: {n_total} valores reais + 19 sentinelas/impossiveis = 34 "
    "-> imputados pela MEDIANA",
    fontsize=13,
)
for i, v in enumerate(fal.values):
    ax.text(i, v + 0.03, str(v), ha="center", fontsize=10)
grade(ax)
fig.tight_layout()
salvar(fig, "Q2_faltantes.png")
plt.close(fig)

cols_com_falta = [c for c in FEATURES if df[c].isna().any()]
n = len(cols_com_falta)
fig, axes = plt.subplots(2, n, figsize=(5 * n, 8))
if n == 1:
    axes = axes.reshape(2, 1)
for j, col in enumerate(cols_com_falta):
    s_antes = df[col].dropna()
    s_depois = X_imp[col]
    bins = np.histogram_bin_edges(pd.concat([s_antes, s_depois]), bins=20)
    ymax = max(np.histogram(s, bins=bins)[0].max() for s in (s_antes, s_depois))

    ax = axes[0, j]
    ax.hist(s_antes, bins=bins, color=VERM, edgecolor="white")
    ax.set_ylim(0, ymax * 1.15)
    ax.set_title(f"{col} - ANTES (com NaN)", fontsize=10)
    ax.tick_params(labelsize=9)
    grade(ax)

    ax = axes[1, j]
    ax.hist(s_depois, bins=bins, color=AZUL, edgecolor="white")
    ax.axvline(s_depois.median(), color="black", linestyle="--", linewidth=1.5,
               label=f"mediana = {s_depois.median():.1f}")
    ax.set_ylim(0, ymax * 1.15)
    ax.set_title(f"{col} - DEPOIS (mediana)", fontsize=10)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=9)
    grade(ax)

nfalt = int(df[FEATURES].isna().sum().sum())
fig.suptitle(
    "Q2 - Imputacao por MEDIANA (robusta a outliers): forma preservada "
    "(mesma escala nos dois paineis)\n"
    f"{nfalt} valores imputados nas 4 variaveis",
    fontsize=13,
)
fig.tight_layout(rect=[0, 0, 1, 0.93])
salvar(fig, "Q2_imputacao_antes_depois.png")
plt.close(fig)


# ===========================================================================
# Q3 -- QUAIS METODOS PARA OUTLIERS?
# ===========================================================================
print("\n[Q3] outliers")

fig, axes = plt.subplots(1, 4, figsize=(20, 6))
pct_out = {}
for ax, col in zip(axes, FEATURES):
    s = X_imp[col]
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    pct = 100 * ((s < lo) | (s > hi)).mean()
    pct_out[col] = pct

    ax.boxplot(s, orientation="vertical", patch_artist=True,
               boxprops=dict(facecolor=AZUL, alpha=0.7),
               medianprops=dict(color="black"))
    ax.axhline(hi, color=VERM, linestyle="--", linewidth=1.4)
    ax.axhline(lo, color=VERM, linestyle="--", linewidth=1.4)
    titulo = f"{col}\n(log) outliers: {pct:.0f}%" if col == "income" \
        else f"{col}\noutliers: {pct:.0f}%"
    if col == "income":
        ax.set_yscale("log")
    ax.set_title(titulo, fontsize=11)
    ax.set_ylabel("valor (unidade da variavel)")
    grade(ax)
    ax.annotate("cercas do IQR\n(Q1-1,5.IQR e Q3+1,5.IQR)", xy=(1, hi),
                xytext=(0.55, hi), fontsize=9, color=VERM,
                arrowprops=dict(arrowstyle="->", color=VERM))
fig.suptitle(
    "Q3 - Outliers: detectados por IQR e MANTIDOS (paises genuinamente ricos/pobres)",
    fontsize=14,
)
fig.tight_layout(rect=[0, 0, 1, 0.94])
salvar(fig, "Q3_outliers_iqr.png")
plt.close(fig)
print("  % de outliers por variavel:", {k: round(v, 1) for k, v in pct_out.items()})


# ===========================================================================
# Q4 -- FOI NECESSARIO TRATAMENTO DA ASSIMETRIA?
# ===========================================================================
print("\n[Q4] assimetria")

fig, axes = plt.subplots(2, 2, figsize=(15, 9))
skews = {}
for j, col in enumerate(LOG_FEATURES):
    s0 = X_imp[col]
    s1 = np.log1p(s0.clip(lower=0))
    skews[col] = (s0.skew(), s1.skew())

    ax = axes[0, j]
    ax.hist(s0, bins=25, color=VERM, edgecolor="white")
    ax.set_title(f"{col} - ANTES\n(skew = {s0.skew():.2f})", fontsize=11)
    ax.set_ylabel("numero de paises")
    grade(ax)
    ax.annotate(f"skew = {s0.skew():.2f}".replace(".", ","), xy=(0.60, 0.85),
                xycoords="axes fraction", fontsize=11, color=VERM)

    ax = axes[1, j]
    ax.hist(s1, bins=25, color=AZUL, edgecolor="white")
    ax.set_title(f"{col} - DEPOIS do log1p\n(skew = {s1.skew():.2f})", fontsize=11)
    ax.set_ylabel("numero de paises")
    grade(ax)
    ax.annotate(f"skew = {s1.skew():.2f}".replace(".", ","), xy=(0.60, 0.85),
                xycoords="axes fraction", fontsize=11, color=AZUL)
fig.suptitle(
    "Q4 - Assimetria: SIM, foi necessario - log1p em child_mort e income (skew -> ~0)",
    fontsize=14,
)
fig.tight_layout(rect=[0, 0, 1, 0.95])
salvar(fig, "Q4_assimetria_log.png")
plt.close(fig)
for col, (a, b) in skews.items():
    print(f"  {col:14s}: skew {a:+.3f} -> {b:+.3f}")


# ===========================================================================
# Q5 -- HOUVE ESCALONAMENTO?
# ===========================================================================
print("\n[Q5] escalonamento")

X_transf = X_imp.copy()
for col in LOG_FEATURES:
    X_transf[col] = np.log1p(X_transf[col].clip(lower=0))
X_robust = pd.DataFrame(RobustScaler().fit_transform(X_transf),
                        columns=FEATURES, index=df.index)

fig, axes = plt.subplots(2, 4, figsize=(20, 9))
for j, col in enumerate(FEATURES):
    ax = axes[0, j]
    ax.hist(X_transf[col], bins=20, color=LARANJA, edgecolor="white")
    ax.set_title(f"{col}\n(ANTES - unidade bruta)", fontsize=10)
    ax.set_ylabel("n de paises")
    grade(ax)

    ax = axes[1, j]
    ax.hist(X_robust[col], bins=20, color=AZUL, edgecolor="white")
    ax.axvline(0, color="black", linestyle="--", linewidth=1.3)
    ax.set_title(f"{col}\n(DEPOIS - RobustScaler: mediana 0, IQR 1)", fontsize=10)
    ax.set_ylabel("n de paises")
    grade(ax)
fig.suptitle(
    "Q5 - Escalonamento: SIM - RobustScaler poe as 4 variaveis na mesma regua "
    "(mediana 0, IQR 1)",
    fontsize=14,
)
fig.tight_layout(rect=[0, 0, 1, 0.94])
salvar(fig, "Q5_escalonamento_antes_depois.png")
plt.close(fig)


# ===========================================================================
# Q6 -- QUAL METRICA DE DISTANCIA? (contribuicao em %)
# ===========================================================================
print("\n[Q6] distancia")

rico = X_imp["income"].idxmax()
pobre = X_imp["income"].idxmin()
nome_rico, nome_pobre = df.loc[rico, "country"], df.loc[pobre, "country"]

a_b = X_imp.loc[rico, FEATURES] - X_imp.loc[pobre, FEATURES]
contrib_bruta = (a_b ** 2)

Z = (X_imp - X_imp.mean()) / X_imp.std()
a_bz = Z.loc[rico, FEATURES] - Z.loc[pobre, FEATURES]
contrib_z = (a_bz ** 2)

pct_bruta = 100 * contrib_bruta / contrib_bruta.sum()
pct_z = 100 * contrib_z / contrib_z.sum()

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
x = np.arange(len(FEATURES))

ax = axes[0]
ax.bar(x, pct_bruta.values, color=VERM)
ax.set_xticks(x)
ax.set_xticklabels(FEATURES, rotation=20, fontsize=9)
ax.set_ylabel("contribuicao para a distancia (%)")
ax.set_title("SEM escalonar", fontsize=12)
for xi, v in zip(x, pct_bruta.values):
    ax.text(xi, v + 1, f"{v:.0f}%", ha="center", fontsize=10)
ax.set_ylim(0, 105)
grade(ax)

ax = axes[1]
ax.bar(x, pct_z.values, color=AZUL)
ax.set_xticks(x)
ax.set_xticklabels(FEATURES, rotation=20, fontsize=9)
ax.set_title("COM z-score", fontsize=12)
for xi, v in zip(x, pct_z.values):
    ax.text(xi, v + 1, f"{v:.0f}%", ha="center", fontsize=10)
ax.set_ylim(0, 105)
grade(ax)
fig.suptitle(
    f"Q6 - Distancia Euclidiana (L2): sem escalonar a renda concentra "
    f"{pct_bruta['income']:.0f}%; escalonado, distribui ({pct_z['income']:.0f}%)\n"
    f"exemplo: {nome_rico} x {nome_pobre}",
    fontsize=13,
)
fig.tight_layout(rect=[0, 0, 1, 0.90])
salvar(fig, "Q6_distancia.png")
plt.close(fig)


# ===========================================================================
# Q7 -- ESCOLHA DE k (cotovelo/silhueta + metricas)
# ===========================================================================
print("\n[Q7] escolha de k")

K_RANGE = list(range(2, 11))
met = []
for k in K_RANGE:
    m = KMeans(n_clusters=k, n_init=50, random_state=SEED)
    lab = m.fit_predict(X)
    met.append({"k": k, "inercia": m.inertia_,
                "silhueta": silhouette_score(X, lab),
                "calinski_harabasz": calinski_harabasz_score(X, lab),
                "davies_bouldin": davies_bouldin_score(X, lab)})
met = pd.DataFrame(met)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
ax = axes[0]
ax.plot(met["k"], met["inercia"], marker="o", color=AZUL)
ax.axvline(3, color=VERM, linestyle="--", linewidth=1.4)
ax.annotate("k=3", xy=(3, met.loc[met.k == 3, "inercia"].values[0]),
            xytext=(4.2, met["inercia"].max() * 0.85), fontsize=11, color=VERM,
            arrowprops=dict(arrowstyle="->", color=VERM))
ax.set_title("Cotovelo - Inercia (SSE) por k", fontsize=12)
ax.set_xlabel("k (numero de grupos)")
ax.set_ylabel("Inercia (SSE)")
grade(ax)

ax = axes[1]
ax.plot(met["k"], met["silhueta"], marker="o", color=VERDE)
ax.axvline(3, color=VERM, linestyle="--", linewidth=1.4)
ax.annotate("k=3", xy=(3, met.loc[met.k == 3, "silhueta"].values[0]),
            xytext=(4.2, met["silhueta"].max() * 0.9), fontsize=11, color=VERM,
            arrowprops=dict(arrowstyle="->", color=VERM))
ax.set_title("Silhueta por k (maior = melhor)", fontsize=12)
ax.set_xlabel("k (numero de grupos)")
ax.set_ylabel("Silhueta")
grade(ax)
fig.suptitle(
    "Q7 - k = 3: cotovelo aponta 3; silhueta maxima em k=2 (+0,07 em relacao a k=3)",
    fontsize=14,
)
fig.tight_layout(rect=[0, 0, 1, 0.94])
salvar(fig, "Q7_cotovelo_silhueta.png")
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(19, 6))
dados = [("silhueta", VERDE, "maior = melhor"),
         ("calinski_harabasz", LARANJA, "maior = melhor"),
         ("davies_bouldin", ROXO, "menor = melhor")]
for ax, (col, cor, nota) in zip(axes, dados):
    ax.plot(met["k"], met[col], marker="o", color=cor)
    ax.axvline(3, color=VERM, linestyle="--", linewidth=1.4)
    ax.annotate("k=3", xy=(3, met.loc[met.k == 3, col].values[0]),
                xytext=(4.0, met[col].max() * 0.9), fontsize=11, color=VERM,
                arrowprops=dict(arrowstyle="->", color=VERM))
    ax.set_title(f"{col} ({nota})", fontsize=12)
    ax.set_xlabel("k (numero de grupos)")
    grade(ax)
fig.suptitle("Q7 - Metricas de apoio: Calinski-Harabasz maximiza em k=3", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.94])
salvar(fig, "Q7_metricas_k.png")
plt.close(fig)


# ===========================================================================
# Q8 -- INTERPRETAR OS GRUPOS (perfis)
# ===========================================================================
print("\n[Q8] perfis")

labels = KMeans(n_clusters=K_FINAL, n_init=50, random_state=SEED).fit_predict(X)

perfil = df[["country"]].copy()
perfil[FEATURES] = X_imp[FEATURES]
perfil["cluster"] = labels
perfil_med = perfil.groupby("cluster")[FEATURES].median()
perfil_med["n_paises"] = perfil.groupby("cluster").size()

ordem = perfil_med["child_mort"].sort_values(ascending=False).index.tolist()
nomes = ["Alta necessidade", "Intermediaria", "Baixa necessidade"]
mapa = dict(zip(ordem, nomes))

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
Zperfil = ((perfil_med[FEATURES] - X_imp[FEATURES].mean())
           / X_imp[FEATURES].std()).loc[ordem]
x = np.arange(len(FEATURES))
larg = 0.25
cores_perfil = [VERM, LARANJA, VERDE]

ax = axes[0]
for i, c in enumerate(ordem):
    ax.bar(x + i * larg, Zperfil.loc[c].values, larg,
           label=f"{nomes[i]} (n={int(perfil_med.loc[c, 'n_paises'])})",
           color=cores_perfil[i])
ax.set_xticks(x + larg)
ax.set_xticklabels(FEATURES, rotation=20, fontsize=9)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("valor padronizado (z-score)")
ax.set_title("Perfil dos grupos (padronizado)", fontsize=12)
ax.legend(fontsize=9)
grade(ax)

ax = axes[1]
cores_c = {ordem[0]: VERM, ordem[1]: LARANJA, ordem[2]: VERDE}
for c in ordem:
    mask = perfil["cluster"] == c
    ax.scatter(df.loc[mask, "child_mort"], df.loc[mask, "income"], s=30, alpha=0.8,
               color=cores_c[c], label=f"{mapa[c]} (n={int(mask.sum())})")
ax.set_xlabel("child_mort (por mil nascidos)")
ax.set_ylabel("income (US$)")
ax.set_title("Paises coloridos pelos 3 clusters", fontsize=12)
ax.legend(fontsize=9)
grade(ax)
fig.suptitle(
    "Q8 - 3 perfis de necessidade (K-Means, k=3): alta / intermediaria / baixa",
    fontsize=14,
)
fig.tight_layout(rect=[0, 0, 1, 0.94])
salvar(fig, "Q8_perfis.png")
plt.close(fig)
print("  perfis (n):", {mapa[c]: int(perfil_med.loc[c, "n_paises"]) for c in ordem})


# ===========================================================================
# Q9 -- HIERARQUICO (dendrograma + Ward vs K-Means)
# ===========================================================================
print("\n[Q9] hierarquico")

Z = linkage(X, method="ward", metric="euclidean")

fig, ax = plt.subplots(figsize=(20, 9))
dendrogram(Z, labels=df["country"].values, leaf_rotation=90, leaf_font_size=8,
           color_threshold=0, above_threshold_color="black", ax=ax)
alturas = sorted(Z[:, 2], reverse=True)
corte = (alturas[K_FINAL - 2] + alturas[K_FINAL - 1]) / 2
ax.axhline(corte, color=VERM, linestyle="--", linewidth=1.6)
ax.annotate("corte -> k=3", xy=(len(df) * 0.02, corte),
            xytext=(len(df) * 0.02, corte * 1.05), fontsize=12, color=VERM)
ax.set_title("Q9 - Dendrograma (Ward): corte em k=3  (a altura NAO e k!)", fontsize=14)
ax.set_ylabel("Altura de fusao (distancia de Ward)")
fig.tight_layout()
salvar(fig, "Q9_dendrograma.png")
plt.close(fig)

labels_ward = fcluster(Z, t=K_FINAL, criterion="maxclust")
ari = adjusted_rand_score(labels, labels_ward)
crosstab = pd.crosstab(labels, labels_ward)

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(crosstab.values, cmap="Blues")
ax.set_xticks(range(crosstab.shape[1]))
ax.set_xticklabels([f"Ward {c}" for c in crosstab.columns], fontsize=11)
ax.set_yticks(range(crosstab.shape[0]))
ax.set_yticklabels([f"K-Means {r}" for r in crosstab.index], fontsize=11)
for i in range(crosstab.shape[0]):
    for j in range(crosstab.shape[1]):
        ax.text(j, i, int(crosstab.values[i, j]), ha="center", va="center",
                color="black", fontsize=13)
ax.set_title(f"Q9 - Concordancia K-Means x Ward (k=3)\nARI = {ari:.2f}", fontsize=14)
fig.colorbar(im, ax=ax, label="Numero de paises")
fig.tight_layout()
salvar(fig, "Q9_ward_vs_kmeans.png")
plt.close(fig)
print(f"  ARI K-Means x Ward = {ari:.4f}")

print("\nFiguras do relatorio concluidas.")
