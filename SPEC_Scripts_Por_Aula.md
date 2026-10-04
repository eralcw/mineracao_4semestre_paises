# SPEC — Scripts `.py` por aula + figuras para o relatório

> **Como usar:** cole este arquivo no chat do Continue (VS Code) e peça para executar.
> Objetivo: **quebrar o pipeline em um `.py` por aula** (aula02…aula07), cada um gerando
> **todas as figuras pertinentes (antes × depois)** em `outputs/figuras/`, para ir no relatório.
> Não mexer no notebook existente; criar arquivos novos.

---

## 1. Contexto (não invente outra coisa)

Projeto 1 de Mineração de Dados — ONG *Help International*. Agrupar **167 países** por
**perfil de necessidade** (não ranking), usando **4 indicadores**. Dados em
`data/paises_help_international.csv` (167 × 18). Método: K-Means (k=3) + agrupamento
hierárquico (Ward). Isso já está pronto e validado no notebook
`notebooks/Projeto1_Mineracao_Pipeline_Final.ipynb` — **os scripts `.py` devem reproduzir
exatamente a mesma lógica** (não inventar métodos novos).

## 2. Arquivos a criar

Pasta: **`src/`**. Um arquivo por aula:

| Arquivo | Aula | Tema |
|---|---|---|
| `src/_comum.py` | — | utilidades e constantes compartilhadas |
| `src/aula02_limpeza.py` | 02 | limpeza e qualidade dos dados |
| `src/aula03_escalonamento.py` | 03 | normalização/escalonamento e assimetria |
| `src/aula04_eda_correlacao.py` | 04 | distância, correlação e EDA |
| `src/aula05_kmeans.py` | 05 | K-Means e escolha de *k* |
| `src/aula06_metricas.py` | 06 | avaliação de agrupamento (métricas) |
| `src/aula07_hierarquico.py` | 07 | agrupamento hierárquico (Ward) |

Figuras → **`outputs/figuras/`** (criar a pasta se não existir). Formato PNG, `dpi=150`.

## 3. Convenções obrigatórias (para os scripts serem consistentes)

- `import matplotlib; matplotlib.use("Agg")` — **nunca** `plt.show()`; só `fig.savefig(...)`.
- Títulos e rótulos dos gráficos **em português**, legíveis (o professor vai ler as figuras).
- `SEED = 42` fixo; `KMeans(..., n_init=50, random_state=SEED)`.
- Cada script precisa **rodar sozinho**: `python src/aula0X_....py` (de qualquer pasta).
  Para importar o `_comum`, usar no topo de cada script:
  ```python
  import sys
  from pathlib import Path
  sys.path.insert(0, str(Path(__file__).resolve().parent))
  from _comum import ...
  ```
- **Não** instalar nada; usar `pandas, numpy, matplotlib, scikit-learn, scipy` do `.venv`.
- Não apagar nem alterar o notebook nem os CSVs de `outputs/`.

## 4. `src/_comum.py` — deve conter

```python
"""Utilidades compartilhadas — Projeto 1 (Mineração de Dados)."""
from pathlib import Path
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent          # raiz do repositório
FIG_DIR = RAIZ / "outputs" / "figuras"

# --- códigos de "sem dado" (faltantes disfarçados) ---
SENTINELAS = [99999, 100000, 999]

# --- valores fora da faixa válida (viram NaN) ---
DOMAIN_RULES = {
    "health":                   lambda s: (s < 0) | (s > 100),
    "child_mort":               lambda s: s < 0,
    "life_expec":               lambda s: (s <= 0) | (s > 120),
    "total_fer":                lambda s: (s < 0) | (s > 20),
    "acesso_agua_pct":          lambda s: (s < 0) | (s > 100),
    "acesso_esgoto_pct":        lambda s: (s < 0) | (s > 100),
    "acesso_energia_pct":       lambda s: (s < 0) | (s > 100),
    "alfabetizacao_pct":        lambda s: (s < 0) | (s > 100),
    "internet_pct":             lambda s: (s < 0) | (s > 100),
    "medicos_por_1000":         lambda s: s < 0,
    "extrema_pobreza_pct":      lambda s: (s < 0) | (s > 100),
    "cesta_basica_salario_pct": lambda s: s < 0,   # >100 é LEGÍTIMO
}

FEATURES = ["child_mort", "income", "acesso_agua_pct", "alfabetizacao_pct"]  # os 4 indicadores
LOG_FEATURES = ["child_mort", "income"]        # assimétricos -> log1p
```

E as funções:

- `carregar_bruto()` → lê o CSV procurando em `data/paises_help_international.csv`
  (e caminhos alternativos: `./data/...`, `../data/...`) e devolve `df_raw`.
- `colunas_numericas(df)` → lista das colunas numéricas.
- `limpar(df_raw)` → copia, substitui `SENTINELAS` por `NaN`, aplica `DOMAIN_RULES`
  (mask → `NaN`). Devolve o `df` limpo. **Mesma lógica do notebook.**
- `matriz_modelo(df)` → sobre `FEATURES`: imputa por **mediana** (`SimpleImputer`),
  aplica `log1p` em `LOG_FEATURES` (com `.clip(lower=0)`), e **`RobustScaler`**.
  Devolve `DataFrame` padronizado pronto para o K-Means.
- `salvar(fig, nome)` → cria `FIG_DIR` e faz `fig.savefig(FIG_DIR/nome, dpi=150, bbox_inches="tight")`,
  imprimindo o caminho.

## 5. O que cada script faz e **quais figuras** gera

### `aula02_limpeza.py`
Carrega bruto, mostra `shape`/duplicatas, monta o diagnóstico e **compara antes × depois**.
- `aula02_01_faltantes_bruto.png` — barras dos faltantes reais por coluna (na base bruta).
- `aula02_02_antes_depois_hist.png` — grade `2 × 4`: histograma das **4 `FEATURES`** na
  base bruta (linha de cima) vs base limpa (linha de baixo). É a figura "antes e depois".
- `aula02_03_antes_depois_box.png` — mesma grade, com **boxplots** (mostra a sentinela
  `99999` "achatando" a escala antes, e a cauda real depois).
- `aula02_04_efeito_sentinela.png` — `health` e `income` antes vs depois da limpeza,
  destacando o ponto `99999`/`999` (mostra por que converter sentinela importa).
- Salvar também `outputs/dados_limpos.csv` (`encoding="utf-8-sig"`) para inspeção.

### `aula03_escalonamento.py`
Demonstra **por que escalonar** e **como** (sem depender de sklearn).
- `aula03_01_scalers_comparacao.png` — a variável `income` (limpa/imputada) em 4 réguas:
  bruta, **min-max** `(x-min)/(max-min)`, **z-score** `(x-média)/desvio`,
  **robust** `(x-mediana)/IQR`.
- `aula03_02_efeito_escala_distancia.png` — escolher 2 países (um rico, um pobre) e mostrar
  a **contribuição de cada indicador** `(a−b)²` para a distância **antes** e **depois** do
  z-score (evidencia que, sem escala, `income` domina; depois, fica equilibrado).
- `aula03_03_assimetria_log.png` — histogramas de `child_mort` e `income` **antes × depois**
  do `log1p`, com o valor de `skew` anotado em cada painel.
- Imprimir na tela a assimetria (`skew`) antes/depois do log.

### `aula04_eda_correlacao.py`
EDA da base limpa + correlação (a base da escolha dos 4 indicadores).
- `aula04_01_heatmap_correlacao.png` — heatmap da matriz de **Pearson** (base limpa),
  `cmap="coolwarm"`, `vmin=-1`, `vmax=1`.
- `aula04_02_dispersao_features.png` — grade de `scatter` entre os pares das 4 `FEATURES`
  (ou `child_mort × income` e `acesso_agua × alfabetizacao`), para ver as relações.
- `aula04_03_histogramas_limpo.png` — histogramas das 4 `FEATURES` na base limpa (mostra a forma).
- Imprimir os **pares mais correlacionados** (top 15), que justificam rejeitar redundâncias.

### `aula05_kmeans.py`
Roda o K-Means e escolhe o *k*.
- `aula05_01_cotovelo_silhueta.png` — 2 painéis: **inércia** por k (cotovelo) e **silhueta** por k.
- `aula05_02_ch_db.png` — Calinski-Harabasz (maior=melhor) e Davies-Bouldin (menor=melhor) por k.
- `aula05_03_clusters_dispersao.png` — dispersão dos países coloridos pelos **3 clusters**
  (ex.: `child_mort × income` e `acesso_agua × alfabetizacao`), com a tabela de perfis
  (mediana por cluster) impressa.
- Usar `k = 3`. Imprimir silhueta/CH/DB finais e o tamanho de cada cluster.

### `aula06_metricas.py`
Avaliação do agrupamento (Aula 06).
- `aula06_01_metricas_por_k.png` — silhueta, CH e DB **lado a lado** por k (evidência da escolha).
- `aula06_02_estabilidade_sementes.png` — roda o K-Means (k=3) com 10 sementes e mostra o
  **ARI** de cada uma vs a solução base (barra), evidenciando estabilidade.
- Imprimir a matriz de confusão/crosstab K-Means × Ward e o ARI (0 a 1).

### `aula07_hierarquico.py`
Agrupamento hierárquico.
- `aula07_01_dendrograma.png` — `linkage(X, method="ward", metric="euclidean")` +
  `dendrogram(...)` com os nomes dos países nas folhas. **Altura ≠ k** (deixar claro no título).
- `aula07_02_comparacao_ward.png` — heatmap da tabela de concordância **K-Means × Ward**
  (crosstab) + o **ARI** anotado, para a comparação entre métodos.
- Usar `fcluster(Z, t=3, criterion="maxclust")` (**número de grupos**, nunca `"distance"`).

## 6. Regras do enunciado/aulas que o código TEM de respeitar

- Ordem canônica: **limpar → tratar outliers → log (assimetria) → escalonar → modelar**.
- **Outliers: manter** (são países genuinamente ricos/pobres) → por isso **`RobustScaler`**
  (não `StandardScaler`).
- **Distância: Euclidiana (L2)**.
- **4 indicadores** exatamente: `child_mort, income, acesso_agua_pct, alfabetizacao_pct`.
- No `fcluster`, usar `criterion="maxclust"`; no `linkage` de Ward, passar `metric="euclidean"`.
- Não usar `df_raw` (com `99999`) em nenhuma figura "depois" — só a base limpa.

## 7. Resumo das figuras (nomes finais)

```
outputs/figuras/
  aula02_01_faltantes_bruto.png
  aula02_02_antes_depois_hist.png
  aula02_03_antes_depois_box.png
  aula02_04_efeito_sentinela.png
  aula03_01_scalers_comparacao.png
  aula03_02_efeito_escala_distancia.png
  aula03_03_assimetria_log.png
  aula04_01_heatmap_correlacao.png
  aula04_02_dispersao_features.png
  aula04_03_histogramas_limpo.png
  aula05_01_cotovelo_silhueta.png
  aula05_02_ch_db.png
  aula05_03_clusters_dispersao.png
  aula06_01_metricas_por_k.png
  aula06_02_estabilidade_sementes.png
  aula07_01_dendrograma.png
  aula07_02_comparacao_ward.png
```

## 8. Como rodar (na ordem)

```bash
python src/aula02_limpeza.py
python src/aula03_escalonamento.py
python src/aula04_eda_correlacao.py
python src/aula05_kmeans.py
python src/aula06_metricas.py
python src/aula07_hierarquico.py
```

Cada script deve terminar **sem erro** e imprimir o caminho de cada figura salva.

---

### Prompt curto para colar no Continue

> "Crie, na pasta `src/`, um arquivo `.py` por aula — `_comum.py`, `aula02_limpeza.py`,
> `aula03_escalonamento.py`, `aula04_eda_correlacao.py`, `aula05_kmeans.py`,
> `aula06_metricas.py`, `aula07_hierarquico.py` — seguindo exatamente a SPEC_Scripts_Por_Aula.md
> que está na raiz. Cada script roda sozinho (`python src/aula0X_....py`), usa `matplotlib`
> com backend `Agg` (sem `show()`), reaproveita a limpeza de `_comum` (sentinela 99999/100000/999 → NaN,
> DOMAIN_RULES, FEATURES, LOG_FEATURES) e salva todas as figuras listadas em `outputs/figuras/`
> (PNG, 150 dpi, títulos em português). Não altere o notebook nem os CSVs. Ao final, rode os
> seis scripts e me diga se algum deu erro."
