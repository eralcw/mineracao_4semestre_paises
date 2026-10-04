# Projeto 1 — Priorização de ajuda humanitária (Mineração de Dados)
### Roteiro passo a passo do trabalho — para entregar à IA do VS Code

> **Como usar este arquivo:** cole este documento como instrução no chat da IA do VS Code (Copilot / Cursor / continuação) e anexe/reference o arquivo `paises_help_international.csv`, que já está nesta mesma pasta. Peça que a IA execute os passos na ordem e produza (1) o notebook do pipeline e (2) o relatório. Tudo que está marcado com ⚠️ é decisão que precisa ser **escrita e justificada** no relatório — é o que o professor cobra.

---

## 1. O que o trabalho pede (contexto)

Uma ONG (Help International) tem um **fundo limitado** para ajuda humanitária. A pergunta motriz: com base em indicadores socioeconômicos e de saúde, **quais países priorizar** e **como agrupá-los** para orientar estratégias diferentes.

- **Papel de vocês:** a equipe de ciência de dados da ONG.
- **Público:** a diretoria, que decide a alocação dos recursos.
- **Produto:** uma **recomendação baseada em perfis de países com necessidades semelhantes — não um ranking simples.**
- **Regra do enunciado:** embasar a escolha dos países em **4 indicadores**.

O trabalho percorre o conteúdo das **Aulas 1 a 7 (com DBSCAN da Aula 8 como método extra)**: limpeza, escalonamento, análise exploratória/correlação, K-Means (escolha de *k*) e agrupamento hierárquico.

## 2. Entregáveis e prazo

1. **Notebook Jupyter** — pipeline reproduzível, do CSV bruto até os grupos.
2. **Relatório de até 10 páginas** com a recomendação para a diretoria.
3. **Slide-resumo** com o perfil dos grupos prioritários (mencionado na Aula 1 — vale entregar).
4. **Prazo: segunda-feira, 05/10/2026, 23h59.**

O relatório **obrigatoriamente** precisa responder, com justificativa de decisão:
- Quais indicadores foram usados?
- Quais métodos foram usados para preenchimento dos faltantes?
- Quais métodos foram usados para tratamento dos outliers?
- Foi necessário tratamento da assimetria dos dados?
- Houve escalonamento?
- Qual métrica de distância foi usada?

> Critério declarado no enunciado: *"Boas escolhas de dados e de método valem mais do que muitos gráficos."* Ou seja: **justificar cada decisão vale mais que encher o relatório de figura.**

---

## 3. Os dados

Arquivo: **`paises_help_international.csv`** — 167 países, 18 colunas (1 identificador + 17 numéricas).

### 3.1 Dicionário das colunas

| Coluna | Significado | Unidade |
|---|---|---|
| `country` | Nome do país | — (identificador, **não usar no modelo**) |
| `child_mort` | Mortalidade infantil | por 1.000 nascidos |
| `exports` | Exportações | % do PIB |
| `health` | Gasto em saúde | % do PIB |
| `imports` | Importações | % do PIB |
| `income` | Renda líquida per capita | US$ |
| `inflation` | Inflação (crescimento anual do PIB deflator) | % |
| `life_expec` | Expectativa de vida | anos |
| `total_fer` | Fertilidade total | filhos por mulher |
| `gdpp` | PIB per capita | US$ |
| `acesso_agua_pct` | Acesso à água potável | % da população |
| `acesso_esgoto_pct` | Acesso a esgoto | % da população |
| `acesso_energia_pct` | Acesso à energia elétrica | % da população |
| `alfabetizacao_pct` | Alfabetização | % da população |
| `internet_pct` | Acesso à internet | % da população |
| `medicos_por_1000` | Médicos | por 1.000 habitantes |
| `extrema_pobreza_pct` | População em extrema pobreza | % |
| `cesta_basica_salario_pct` | Custo da cesta básica | % do salário mínimo |

As 9 primeiras numéricas (`child_mort` … `gdpp`) são o conjunto clássico "Country Data" citado na Aula 1. As 8 últimas foram acrescentadas e **concentram a maior parte dos problemas de qualidade** — é nelas que o pré-processamento vai trabalhar.

### 3.2 Mapa de qualidade — o que está errado no arquivo bruto

Estes problemas **já foram identificados**; o pipeline precisa tratá-los explicitamente.

**a) Valores faltantes (vazios reais) — 15 linhas:**

| País | Coluna faltante |
|---|---|
| Afghanistan | `internet_pct` |
| Colombia | `cesta_basica_salario_pct` |
| Ecuador | `life_expec` |
| Haiti | `acesso_agua_pct` |
| Iraq | `acesso_agua_pct` |
| Jordan | `inflation` |
| Kazakhstan | `total_fer` |
| Mali | `extrema_pobreza_pct` |
| Mozambique | `alfabetizacao_pct` |
| Norway | `internet_pct` |
| Philippines | `exports` |
| Rwanda | `acesso_esgoto_pct` |
| Tanzania | `child_mort` |
| Vietnam | `acesso_energia_pct` |
| Yemen | `medicos_por_1000` |

**b) Faltantes disfarçados / sentinelas (código `99999` que significa "sem dado"):**

| País | Coluna | Valor |
|---|---|---|
| Bangladesh | `income` | 99999 |
| Guatemala | `health` | 99999 |
| Morocco | `imports` | 99999 |
| Uganda | `gdpp` | 99999 |
| Poland | `health` | **999** (também sentinela) |
| Angola | `acesso_energia_pct` | 99999 |
| Nepal | `acesso_esgoto_pct` | 99999 |
| Zambia | `alfabetizacao_pct` | 99999 |
| Pakistan | `internet_pct` | 99999 |
| Bolivia | `medicos_por_1000` | 99999 |
| Ghana | `cesta_basica_salario_pct` | 99999 |

→ Esses `99999` **precisam virar `NaN` antes de qualquer estatística** (aula 2: "faltantes disfarçados"). Se ficarem, arruínam média, desvio e escala.

**c) Valores impossíveis (fora da faixa válida):**

| País | Coluna | Valor | Problema |
|---|---|---|---|
| Brazil | `acesso_agua_pct` | 150 | % > 100 |
| Kenya | `alfabetizacao_pct` | 240 | % > 100 |
| Peru | `internet_pct` | -12 | % negativo |
| India | `extrema_pobreza_pct` | -4 | % negativo |
| Chile | `child_mort` | -8 | mortalidade negativa |
| Germany | `life_expec` | 205 | expectativa de vida impossível |

→ Tratar como faltante (virar `NaN`) ou corrigir — decidir e justificar.

**d) Outliers plausíveis mas extremos (NÃO são erro necessariamente — investigar):**
- `inflation`: Turkey = 100000 (provável sentinela), Nigeria = 104.
- `total_fer`: Argentina = 45 (implausível — provável erro).
- `gdpp` / `income`: cauda longa forte (Luxembourg 119000 vs Burundi 231) → assimetria à direita.
- ⚠️ **Cuidado:** em `cesta_basica_salario_pct`, valores **> 100 são legítimos** (significa que a cesta custa mais de um salário mínimo — típico de países pobres). **Não** tratar como erro; só o `99999` do Ghana é sentinela.

**e) Coluna identificadora:** `country` tem 100% de cardinalidade única → **não entra no agrupamento**, mas deve ser preservada para rotular os resultados.

**f) Assimetria:** várias variáveis (renda, `gdpp`, inflação, mortalidade infantil) têm cauda longa à direita → vai precisar de discussão de transformação log (aula 3).

---

## 4. Passo a passo do pipeline

> Ordem canônica das aulas (Aula 3, slide "A ordem certa do pré-processamento"):
> **1) limpar → 2) tratar outliers → 3) log (opcional, nas assimétricas) → 4) escalonar → 5) modelar.**
> Seguir essa ordem.

### Etapa 1 — Carregar e inspecionar
- Ler o CSV com pandas; conferir `shape`, `dtypes`, `head()`, `info()`.
- Rodar `isnull().sum()` (faltantes reais) e um perfilamento (`describe()`, `nunique()`, `value_counts()` nas colunas suspeitas) para **encontrar os `99999` e os valores impossíveis**.
- Registrar num "diário de bordo" o que foi encontrado (isso vira tabela no relatório).
- 🔎 *Gráficos de EDA (aula 4): histograma por variável (forma), boxplot (outliers), scatter (relações), heatmap de correlação.*

### Etapa 2 — Converter faltantes disfarçados em `NaN`
- Substituir `99999` (e o `999` do `health` da Poland) por `NaN` em todas as colunas numéricas.
- Substituir os valores impossíveis da seção 3.2.c por `NaN` (ou corrigir — decidir).
- **Justificativa a escrever:** sem isso, um `99999` de "sem dado" vira dado real e destrói média/desvio.

### Etapa 3 — Tratar os valores faltantes (decisão obrigatória)
- Como são **poucos** (15 faltantes reais + ~11 sentinelas, num total de ~167 linhas), avaliar:
  - **Imputação por mediana** (robusta a outliers) — opção simples e defensável; ou
  - **Imputação por KNN** (`KNNImputer`) — usa os registros parecidos, mais plausível; ou
  - **Remoção das linhas** — só se justificar que são poucas e aleatórias.
- ⚠️ **Decidir e justificar o método escolhido** (é uma das perguntas do relatório). Ideal: comparar 2 métodos e mostrar que o resultado do agrupamento não muda muito.
- Registrar quantas linhas/valores foram imputados.

### Etapa 4 — Duplicatas e identificadores
- Checar `df.duplicated().sum()` → remover duplicatas reais se houver (aula 2).
- Confirmar que `country` é ID (cardinalidade 100%) e **excluí-lo do modelo**, guardando-o à parte.

### Etapa 5 — Outliers (decisão obrigatória)
- Detectar com **IQR** (robusto: fora de `Q1 − 1.5·IQR` e `Q3 + 1.5·IQR`) e/ou **z-score** (|z| > 3).
- Para cada outlier, decidir entre **remover / corrigir / manter**. Nas variáveis socioeconômicas, extremos muitas vezes são **países genuinamente pobres ou ricos** — remover pode apagar justamente o que interessa à ONG.
- ⚠️ **Justificar a decisão** — e lembrar que ela "aponta qual scaler usar" (aula 3): se mantiver outliers, usar `RobustScaler`.

### Etapa 6 — Assimetria (decisão obrigatória: "foi necessário?")
- Calcular `skew()` por variável. Distribuições com cauda longa (renda, `gdpp`, inflação, `child_mort`) → **assimetria**.
- Se necessário, aplicar **transformação log** (`np.log1p`) **antes** de escalonar.
- ⚠️ **Escrever claramente:** escalonar **não** resolve assimetria — são etapas diferentes (log muda a **forma**; scaler muda a **escala**). Só aplicar log onde fizer sentido.

### Etapa 7 — Selecionar os 4 indicadores
- Montar o **heatmap de correlação (Pearson)** entre todas as variáveis numéricas.
- **Objetivo:** escolher **4 indicadores** para o agrupamento, evitando **redundância** (duas colunas muito correlacionadas dizem quase a mesma coisa — ex.: `income` ↔ `gdpp`, `acesso_agua_pct` ↔ `acesso_esgoto_pct`).
- Escolher indicadores que cubram **dimensões diferentes** da necessidade (ex.: uma de saúde, uma de renda/economia, uma de infraestrutura/acesso, uma de educação) — e que ajudem a ONG a definir **estratégias diferentes por grupo**.
- ⚠️ **Justificar por que esses 4** (é a primeira pergunta do relatório). Aula 4: correlação detecta redundância; *"correlação não é causa"*.

### Etapa 8 — Escalonar (decisão obrigatória: "houve escalonamento?")
- **Padronizar com z-score (`StandardScaler`)** — variáveis têm unidades muito diferentes (US$, %, anos), e todo método de distância sofre com isso.
- **Se mantiver outliers**, preferir `RobustScaler` (mediana + IQR).
- ⚠️ Justificar a escolha do scaler. Explicar o efeito: **sem escalonar, a variável de maior faixa domina a distância** (aula 3/4).
- *Obs.: aqui não há treino/teste (é aprendizado não supervisionado), então o `fit` é feito uma vez sobre a base completa — não há teste para vazar.*

### Etapa 9 — Métrica de distância (decisão obrigatória)
- Escolher **uma** métrica e justificar (aula 4):
  - **Euclidiana (L2)** — padrão para dados contínuos, poucas dimensões, **escalonados**. *Recomendada como escolha principal.*
  - **Manhattan (L1)** — menos sensível a diferenças grandes de uma única variável.
  - **Cosseno** — quando importa o "perfil/proporção", não o tamanho.
- ⚠️ Escrever a métrica no relatório e por que ela combina com o escalonamento e com o K-Means.

### Etapa 10 — K-Means: escolher *k* e interpretar
- Rodar `KMeans(init="k-means++", n_init=10, random_state=42)` para `k` de **2 a 10**.
- Escolher *k* por **duas evidências combinadas** (aula 5/6):
  - **Cotovelo (elbow)** — plotar a inércia (`inertia_`) por *k* e achar onde a curva dobra.
  - **Silhueta** (`silhouette_score`) — maior é melhor; plotar por *k*.
  - Reforçar com **Calinski-Harabasz** (maior = melhor) e **Davies-Bouldin** (menor = melhor).
- ⚠️ Se as métricas discordarem, decidir pela **maioria + estabilidade + interpretabilidade** (a aula diz: na dúvida, o **menor *k* útil**).
- **Interpretar os centróides** de cada grupo: o que caracteriza o grupo? Dar **nome** a cada grupo (ex.: "perfil de alta necessidade em saúde").
- Rodar com **várias sementes (`random_state`)** para verificar **estabilidade**.

### Etapa 11 — Agrupamento hierárquico (comparação)
- `Z = linkage(Xz, method="ward")` (Ward como padrão; a aula recomenda começar por ele).
- Desenhar o **dendrograma** (`dendrogram(Z)`) e cortar no **maior salto** de altura.
- Extrair os grupos com `fcluster(Z, t=k, criterion="maxclust")` — ⚠️ **usar `maxclust` (número de grupos), não `distance` (altura)**; confundir os dois é o erro mais silencioso da aula.
- Comparar a partição do hierárquico com a do K-Means (usando a mesma *k* e a silhueta de cada um).

### Etapa 12 — (Opcional / extra) DBSCAN
- Se quiser mostrar um método alternativo (aula 8): padronizar, `minPts ≈ 2 × nº de variáveis`, escolher `eps` pela **curva de k-distâncias** (cotovelo), rodar `DBSCAN`.
- Os pontos `-1` são **ruído/outliers** — o DBSCAN entrega isso de graça. Reportar quantos grupos e quantos `-1`.
- Comparar com K-Means: se os grupos "batem", a estrutura é mais ou menos esférica; se o DBSCAN acha formas que o K-Means quebra, use isso como argumento.

### Etapa 13 — Comparar, decidir e concluir
- Colocar os métodos **lado a lado** (tamanho dos grupos, silhueta, coerência).
- Escolher o agrupamento final e **justificar a escolha**.

### Etapa 14 — A recomendação à diretoria
- Traduzir os grupos em **estratégias diferentes de ajuda** (o ponto central do produto): cada grupo recebe uma estratégia conforme seu perfil de necessidade.
- Priorizar os grupos **não por um ranking de países, mas por perfil** — ex.: grupo "crise de saúde infantil + renda baixa" → intervenção em saúde; grupo "infraestrutura precária" → saneamento/energia.
- Indicar **quais países** caem em cada grupo (usando a coluna `country`).

---

## 5. Estrutura sugerida do notebook

1. **Imports e configuração** (pandas, numpy, matplotlib/seaborn, sklearn, scipy).
2. **Carregamento + EDA inicial** (shape, dtypes, faltantes, describe, gráficos).
3. **Limpeza** (sentinela→NaN, impossíveis→NaN, faltantes imputados, duplicatas).
4. **Outliers** (IQR/z-score + decisão).
5. **Assimetria** (skew + log se necessário).
6. **Seleção das 4 variáveis** (heatmap de correlação + justificativa).
7. **Escalonamento** (scaler escolhido).
8. **K-Means** (cotovelo, silhueta, CH, DB, escolha de *k*, centróides, rótulos por país).
9. **Hierárquico** (linkage, dendrograma, corte, comparação).
10. **(Opcional) DBSCAN.**
11. **Tabela final:** país × grupo × perfil → base do relatório e do slide.

**Reprodutibilidade:** fixar `random_state` em tudo; deixar o notebook rodar do zero sem erro (Célula → Run All).

## 6. Estrutura sugerida do relatório (até 10 páginas)

1. **Resumo executivo** (1 parágrafo + o *k* escolhido e os grupos).
2. **Dados e qualidade** — fonte, 167 países, problemas encontrados (usar as tabelas da seção 3.2).
3. **Pré-processamento** — faltantes (método + justificativa), outliers (decisão), assimetria (sim/não + por quê). *(Responde 3 das perguntas do enunciado.)*
4. **Seleção dos indicadores** — os 4 escolhidos, com o heatmap e a justificativa. *(Responde "quais indicadores".)*
5. **Escalonamento e distância** — scaler e métrica, com justificativa. *(Responde as 2 últimas perguntas.)*
6. **Agrupamento** — *k* (cotovelo + silhueta), perfil de cada grupo, comparação com o hierárquico.
7. **Recomendação à diretoria** — estratégia por perfil de grupo + países de cada grupo.
8. **Limitações** — o que o método não captura, o que ficou de fora.

## 7. Checklist final (antes de entregar)

- [ ] Sentinelas (`99999`/`999`) e valores impossíveis tratados.
- [ ] Faltantes imputados **com método justificado**.
- [ ] Outliers tratados **com decisão justificada**.
- [ ] Assimetria avaliada (log ou justificativa de não usar).
- [ ] **Exatamente 4 indicadores** escolhidos e justificados (heatmap de correlação).
- [ ] Escalonamento feito e justificado.
- [ ] Métrica de distância escolhida e justificada.
- [ ] *k* escolhido por cotovelo **e** silhueta (concordância ou desempate explicado).
- [ ] Grupos **interpretados e nomeados**; países listados por grupo.
- [ ] Hierárquico rodado e comparado com o K-Means.
- [ ] Notebook roda do zero (Run All) sem erro.
- [ ] Relatório responde **todas** as 6 perguntas do enunciado.
- [ ] Slide-resumo com o perfil dos grupos.

## 8. Erros comuns a evitar (das aulas)

- Escolher o *k* pelo menor SSE (inércia sozinha **sempre** cai → use cotovelo + silhueta).
- **Esquecer de escalonar** antes de qualquer método de distância.
- Deixar `99999` no dado como se fosse valor real.
- Confundir **altura** do dendrograma com o número de grupos *k*.
- Usar `criterion="distance"` pensando que é `="maxclust"` no `fcluster`.
- Reportar um único número "certo" de grupos sem mostrar a evidência.
- Encher o relatório de gráficos e **não justificar as decisões** (é o oposto do que o professor valoriza).

---

### Prompt curto para colar no VS Code (se quiser um resumo)

> "Nesta pasta está `paises_help_international.csv` (167 países, 17 indicadores). Siga o roteiro de `Projeto1_Passo_a_Passo.md` para construir um pipeline de mineração de dados não supervisionada: limpar (sentinela 99999→NaN, imputar faltantes), tratar outliers e assimetria, escolher 4 indicadores por correlação, padronizar (z-score), rodar K-Means (escolher *k* por cotovelo + silhueta; também Calinski-Harabasz e Davies-Bouldin) e agrupamento hierárquico (Ward + dendrograma), interpretar e nomear os grupos, e gerar a tabela país→grupo. Ao final, produza o notebook reproduzível e o relatório de até 10 páginas respondendo às 6 perguntas do enunciado (indicadores, faltantes, outliers, assimetria, escalonamento, distância)."
