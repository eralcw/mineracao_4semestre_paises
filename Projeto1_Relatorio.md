# Relatório — Priorização de ajuda humanitária (ONG Help International)

**Disciplina:** Mineração de Dados — Projeto 1
**Público:** Diretoria da ONG Help International
**Entregável:** recomendação de priorização baseada em **perfis de países**, com pipeline reproduzível

---

## 1. Resumo executivo

Construímos um pipeline de **aprendizado não supervisionado** (K-Means + agrupamento
hierárquico) sobre **167 países** e **4 indicadores** socioeconômicos, para agrupar
países com **necessidades parecidas** e orientar a alocação de um fundo limitado.

Em vez de um *ranking* linear de países, o resultado é uma **tipologia de 3 perfis**:

| Perfil | n | Mortalidade infantil (por mil) | Renda per capita (US$) | Acesso à água (%) | Alfabetização (%) |
|---|---|---|---|---|---|
| **1 – Alta necessidade** | 48 | 88,8 | 1.860 | 55,4 | 61,0 |
| **2 – Necessidade intermediária** | 59 | 21,5 | 9.720 | 73,1 | 77,6 |
| **3 – Baixa necessidade / maior capacidade** | 60 | 5,8 | 31.350 | 90,9 | 90,8 |

**Decisão recomendada:** priorizar o **Perfil 1** (48 países), que concentra a maior
necessidade em todas as dimensões. O Perfil 2 recebe apoio intermediário; o Perfil 3
não é prioridade de ajuda emergencial. Detalhes e listas de países nas seções 6 e 7.

**Escolha de *k*:** 3. A escolha é defensável por cotovelo, Calinski-Harabasz e
interpretabilidade; o trade-off está discutido na seção 6.3 (é o ponto mais sensível
do trabalho e está explicitado em vez de escondido).

**Método de distância:** Euclidiana (L2). **Escalonamento:** RobustScaler.
**Assimetria:** tratada com log em renda e mortalidade infantil.

---

## 2. Dados e qualidade

- **Fonte:** `paises_help_international.csv` — 167 países × 18 colunas (1 ID + 17 numéricas).
- **A coluna `country`** tem 100% de cardinalidade única: é um **identificador**, não
  entra no modelo, mas é preservada para rotular os resultados.
- Não há **linhas duplicadas**.

O arquivo bruto concentra os problemas de qualidade nas 8 colunas adicionais
(`acesso_agua_pct` … `cesta_basica_salario_pct`). Encontramos **três tipos** de problema:

### 2.1 Faltantes reais (vazios explícitos) — 15 valores
Afghanistan (`internet_pct`), Colombia (`cesta_basica_salario_pct`), Ecuador
(`life_expec`), Haiti (`acesso_agua_pct`), Iraq (`acesso_agua_pct`), Jordan
(`inflation`), Kazakhstan (`total_fer`), Mali (`extrema_pobreza_pct`), Mozambique
(`alfabetizacao_pct`), Norway (`internet_pct`), Philippines (`exports`), Rwanda
(`acesso_esgoto_pct`), Tanzania (`child_mort`), Vietnam (`acesso_energia_pct`),
Yemen (`medicos_por_1000`).

### 2.2 Faltantes disfarçados (sentinelas) — `99999`, `100000` e `999`
Bangladesh (`income`), Guatemala (`health`), Morocco (`imports`), Uganda (`gdpp`),
**Poland (`health` = 999)**, Angola (`acesso_energia_pct`), Nepal (`acesso_esgoto_pct`),
Zambia (`alfabetizacao_pct`), Pakistan (`internet_pct`), Bolivia (`medicos_por_1000`),
Ghana (`cesta_basica_salario_pct`) e **Turkey (`inflation` = 100000)** — 12 sentinelas no total.

> Esses códigos significam "sem informação" (`99999` e `100000` como "sem dado"; `999`
> no `health` da Poland). Se ficassem no dado, um único `99999` destruiria a média, o
> desvio e a escala da variável. **Transformamos todos em `NaN`.**

### 2.3 Valores impossíveis (fora da faixa válida)
Brazil (`acesso_agua_pct` = 150), Kenya (`alfabetizacao_pct` = 240), Peru
(`internet_pct` = −12), India (`extrema_pobreza_pct` = −4), Chile (`child_mort` = −8),
Germany (`life_expec` = 205), Argentina (`total_fer` = 45).

> São erros de digitação/regra de negócio. **Também viramos `NaN`** (em vez de corrigir
> chutando um valor), pois o valor correto é desconhecido e a imputação por mediana trata.

### 2.4 Cuidado de domínio (não é erro!)
Em `cesta_basica_salario_pct`, **valores > 100 são legítimos** — significam que a cesta
básica custa mais de um salário mínimo (típico de países pobres). **Não** tratamos como
erro; apenas o `99999` (Ghana) é sentinela.

**Resultado da limpeza:** 19 valores viraram `NaN` (sentinelas/impossíveis) **+** 15
faltantes reais **= 34 valores ausentes** a imputar, distribuídos em ~167 linhas.

---

## 3. Pré-processamento

Ordem canônica seguida (Aula 3): **limpar → tratar outliers → log (assimetria) →
escalonar → modelar**.

### 3.1 Preenchimento dos faltantes — *imputação por mediana*
**Pergunta do enunciado: "quais métodos foram usados para preenchimento dos faltantes?"**

- **Método escolhido: imputação por mediana.**
- **Justificativa:** são poucos faltantes (< 6% do total de células dos 4 indicadores) e
  a **mediana é robusta a outliers**, ao contrário da média (um único país rico com
  renda altíssima não a desloca).
- **Alternativa testada: `KNNImputer`.** Comparamos mediana × KNN em **pipelines
  idênticos** (mesmo log + mesmo escalonamento nos dois ramos, para que a diferença no
  ARI fosse atribuível **só** ao método de imputação). Resultado: **ARI = 0,96** — ou
  seja, o agrupamento é praticamente o mesmo. Mantivemos a mediana por ser mais simples,
  robusta e por **não criar dependência entre features** (o KNN usa as outras colunas
  para estimar, o que "vaza" informação entre indicadores correlacionados).

> *Nota de transparência:* uma versão anterior deste notebook reportava ARI ≈ 0,55 nessa
> comparação. O valor baixo vinha de um **erro de comparabilidade** (um ramo aplicava log
> e o outro não), não do método de imputação. Corrigida a comparação, ambos os ramos
> concordam fortemente (0,96).

### 3.2 Tratamento dos outliers — *decisão: manter*
**Pergunta do enunciado: "quais métodos foram usados para tratamento dos outliers?"**

- **Detecção:** método do **IQR** (valores fora de `Q1 − 1,5·IQR` e `Q3 + 1,5·IQR`).
- **Decisão: NÃO remover** os outliers.
- **Justificativa:** nas variáveis socioeconômicas, os extremos são, em sua maioria,
  **países genuinamente muito ricos ou muito pobres** — exatamente a população que
  interessa à ONG. Remover outliers aqui apagaria o alvo do estudo. Erros grosseiros
  (sentinelas e impossíveis) já haviam sido tratados na limpeza (seção 2).
- **Consequência importante:** como **mantemos os outliers**, o escalonamento deve ser
  feito com **`RobustScaler`** (mediana + IQR), que não se deixa arrastar por eles.

### 3.3 Assimetria — *sim, foi necessário tratar*
**Pergunta do enunciado: "foi necessário tratamento da assimetria dos dados?"**

- **Sim.** `income`, `gdpp`, `child_mort`, `total_fer`, `inflation` e `exports` têm
  cauda longa à direita (assimetria positiva forte).
- **Tratamento:** `log1p` aplicado **apenas** a `child_mort` e `income` (os dois
  indicadores assimétricos escolhidos). Após o log, a assimetria caiu para perto de 0
  (skew ≈ 0,07 e ≈ −0,25).
- **Ponto didático (cai na prova):** **escalonar não resolve assimetria.** O log muda a
  **forma** da distribuição; o scaler muda a **escala**. São etapas distintas e nesta
  ordem (log → scaler).

### 3.4 Escalonamento — *sim, houve escalonamento*
**Pergunta do enunciado: "houve escalonamento?"**

- **Sim. Scaler escolhido: `RobustScaler`** (subtrai a mediana e divide pelo IQR).
- **Justificativa:** os indicadores têm unidades incomparáveis (US$, %, por mil). Sem
  escalonar, `income` (que varia de ~300 a ~120.000) dominaria completamente a distância.
  Escolhemos o **RobustScaler** (e não o `StandardScaler`) justamente **porque decidimos
  manter os outliers** (seção 3.2) — a mediana e o IQR são resistentes a eles.
- Como é aprendizado **não supervisionado**, não há treino/teste: o `fit` é feito uma
  vez sobre a base completa (não há vazamento).

### 3.5 Métrica de distância
**Pergunta do enunciado: "qual métrica de distância foi usada?"**

- **Euclidiana (L2).**
- **Justificativa:** é a métrica padrão para **dados contínuos, poucas dimensões
  (4) e já escalonados** (Aula 4). É também a métrica natural do K-Means (que minimiza
  a soma dos quadrados das distâncias) e do Ward.

---

## 4. Seleção dos 4 indicadores

**Pergunta do enunciado: "quais indicadores foram usados?"**

Os 4 indicadores adotados, escolhidos pela análise de correlação (heatmap de Pearson):

| Indicador | Faceta de necessidade |
|---|---|
| `child_mort` | Saúde infantil |
| `income` | Capacidade econômica (renda) |
| `acesso_agua_pct` | Infraestrutura / saneamento |
| `alfabetizacao_pct` | Educação |

**Como chegamos neles (e a ressalva honesta sobre colinearidade):**

Descartamos colunas que medem **praticamente a mesma coisa** que uma já escolhida:
`gdpp` (r = 0,89 com `income`), `acesso_esgoto_pct` / `acesso_energia_pct` /
`internet_pct` (r ≥ 0,95 com `acesso_agua_pct`), `life_expec` (r = −0,89 com
`child_mort`) e `total_fer` (r = 0,85 com `child_mort`). `health` e `exports` foram
descartados por **não medirem necessidade humanitária** diretamente (`exports` é
abertura comercial; `health` discrimina pouco entre os perfis).

> **Ressalva importante:** neste dataset, **quase todos os indicadores socioeconômicos
> são fortemente correlacionados** — eles medem, no fundo, **um mesmo fator latente de
> "desenvolvimento/pobreza"**. Os próprios 4 escolhidos são colineares entre si:

| Par | Correlação (r) |
|---|---|
| `acesso_agua_pct` × `alfabetizacao_pct` | 0,951 |
| `child_mort` × `acesso_agua_pct` | −0,883 |
| `child_mort` × `alfabetizacao_pct` | −0,879 |

Quantificamos essa multicolinearidade com o **VIF**:

| Indicador | VIF |
|---|---|
| `child_mort` | 4,67 |
| `income` | 2,71 |
| `acesso_agua_pct` | 7,25 |
| `alfabetizacao_pct` | 8,75 |

(VIF > 5 já indica colinearidade.) O **1º componente principal explica 82,7%** da
variância dos 4 indicadores — confirmando o fator latente único.

**Conclusão honesta:** *não são 4 dimensões independentes*, e sim **4 facetas
complementares do mesmo eixo de desenvolvimento**. Mantivemos as 4 porque cada uma
responde a uma **estratégia de ajuda distinta** (saúde, renda, saneamento, educação) e
porque a escolha **minimiza a redundância** dentro do que os dados permitem — com este
conjunto de colunas, qualquer subconjunto de 4 colapsa na mesma direção. Reportar isso
abertamente é mais correto do que vender "4 dimensões independentes".

---

## 5. Agrupamento

### 5.1 K-Means — escolha de *k*
Rodamos o K-Means para *k* de 2 a 10 e levantamos 4 evidências:

| k | Inércia | Silhueta | Calinski-Harabasz | Davies-Bouldin |
|---|---|---|---|---|
| 2 | 87,40 | **0,533** | 338,6 | **0,646** |
| **3** | **48,82** | 0,465 | **366,0** | 0,700 |
| 4 | 36,43 | 0,400 | 343,5 | 0,800 |
| 5 | 29,44 | 0,375 | 326,5 | 0,912 |
| … | … | … | … | … |
| 10 | 18,38 | 0,274 | 235,8 | 1,090 |

- **Cotovelo (inércia):** a maior queda ocorre de k=2→3 (−38,6), depois a curva achata
  (−12,4; −7,0; …). Sugere **k = 3**.
- **Silhueta:** máxima em **k = 2** (0,533), caindo para 0,465 em k=3.
- **Calinski-Harabasz:** máxima em **k = 3** (366,0).
- **Davies-Bouldin:** melhor em **k = 2**.

**Há conflito entre as métricas** (silhueta/DB → 2; cotovelo/CH → 3). **Desempate:**

1. O cotovelo e o CH apontam **3**; a silhueta/DB apontam 2, mas a diferença de silhueta
   entre 2 e 3 é pequena (**0,07**).
2. **Interpretabilidade / acionabilidade:** em **k = 2** o grupo "ricos" fica muito
   heterogêneo (junta rendas de ~4.240 a ~119.000), o que é **pouco útil** para a ONG
   decidir. Em **k = 3** obtemos **3 níveis de necessidade** claramente interpretáveis.
3. Aula 5/6: na dúvida, o **menor *k* útil** — e o menor *k* que gera perfis
   **acionáveis** aqui é 3.

**Escolhemos k = 3.**

**Estabilidade:** rodando o K-Means com 10 sementes diferentes, o **ARI médio foi 0,962**
(mínimo 0,962) — a solução é **estável**.

### 5.2 Perfil dos grupos (K-Means, k = 3)

| Perfil | n | Mortal. infantil | Renda (US$) | Água (%) | Alfabetização (%) |
|---|---|---|---|---|---|
| 1 – Alta necessidade | 48 | 88,8 | 1.860 | 55,4 | 61,0 |
| 2 – Intermediária | 59 | 21,5 | 9.720 | 73,1 | 77,6 |
| 3 – Baixa necessidade | 60 | 5,8 | 31.350 | 90,9 | 90,8 |

Os **centróides** (espaço padronizado) confirmam a ordenação monotônica: o Perfil 1 é
"ruim em tudo" (`child_mort` +0,74; `income` −0,89; água −0,82; alfabetização −0,87); o
Perfil 3 é "bom em tudo"; o Perfil 2 fica no meio. Ou seja, os grupos formam um **eixo
único e ordenado de necessidade**.

### 5.3 Agrupamento hierárquico (Ward) e comparação

- `linkage(method="ward")`, corte com `fcluster(..., criterion="maxclust")` (**número de
  grupos**, e **não** `distance`).
- **K-Means (k=3):** silhueta 0,465 | **Ward (k=3):** silhueta 0,432 → o K-Means é
  levemente melhor e foi o escolhido.
- **Concordância K-Means × Ward em k=3: ARI = 0,42.**

**Como interpretar o ARI baixo (ponto para a banca):** em **k=2** a concordância era
altíssima (ARI ≈ 0,95). Isso mostra que os dois métodos **concordam no eixo principal
(pobre × rico)**, mas **divergem em onde cortar o nível intermediário**. O **terceiro
grupo não é robusto entre métodos** — ele deve ser tratado como **zona de transição**,
não como fronteira nítida. Recomendamos usar os perfis como **priorização**, não como
classificação binária.

---

## 6. Recomendação à diretoria

Cada perfil recebe uma **estratégia diferente** (é o produto central do trabalho):

### Perfil 1 — Alta necessidade (48 países) — **PRIORIDADE MÁXIMA**
Mortalidade infantil altíssima (≈ 89‰), renda ≈ US$ 1.860, água e alfabetização baixas.
- **Estratégia:** maior parcela do fundo. Foco em **saúde infantil**, **água/saneamento**
  e **transferência de renda**. Intervenções estruturais, de longo prazo.
- **Países (amostra):** Haiti, Serra Leoa, Chade, República Centro-Africana, Mali,
  Nigéria, Níger, Angola, Burkina Faso, Moçambique, Burundi, Paquistão, Malawi,
  Afeganistão, Zâmbia, Uganda, Sudão, Índia, Iêmen, Nepal, Camboja, Tanzânia…

### Perfil 2 — Necessidade intermediária (59 países) — **PRIORIDADE MÉDIA**
- **Estratégia:** parcela intermediária. Foco em **infraestrutura de acesso (água/esgoto)**
  e **educação**, com apoio a programas de saúde pública. Acompanhamento de transição.
- **Países (amostra):** Bangladesh, Bolívia, Iraque, Indonésia, Filipinas, Egito,
  Marrocos, Vietnã, Peru, Brasil, Colômbia, China, Argentina, Venezuela…

### Perfil 3 — Baixa necessidade / maior capacidade (60 países) — **PRIORIDADE BAIXA**
- **Estratégia:** menor parcela. Apoio **pontual / fortalecimento institucional**, em
  parceria com governos e agências locais. Sem ajuda emergencial.
- **Países (amostra):** Estados Unidos, Alemanha, Japão, Canadá, Austrália, França,
  Reino Unido, Suécia, Noruega, Luxemburgo, Cingapura, Portugal, Espanha, Itália…

> Lista completa dos 167 países com seu cluster está em `outputs/07_paises_clusters.csv`.

---

## 7. Limitações

1. **Colinearidade:** os 4 indicadores medem, em grande parte, o mesmo fator de
   desenvolvimento (1º componente = 82,7%). Os grupos são, essencialmente, **níveis de
   desenvolvimento**, não perfis ortogonais.
2. **Fronteira do Perfil 2:** o ARI de 0,42 com o Ward mostra que o limite do nível
   intermediário não é robusto — é uma **zona de transição**.
3. **Dados de um único período (cross-sectional):** não capturam **tendência/evolução**;
   um país em melhora rápida pode parecer igual a outro estagnado.
4. **Imputação:** 34 valores foram imputados por mediana; onde há mais de um faltante no
   mesmo país, a incerteza aumenta.
5. **Não há variável de "custo de intervenção"** — a priorização é por **necessidade**,
   não por custo-efetividade.

---

## 8. Reprodutibilidade

- Todo o pipeline está em `notebooks/Projeto1_Mineracao_Pipeline_Final.ipynb`.
- Semente fixa (`SEED = 42`) em **todo** o pipeline.
- O notebook roda do zero (*Run All*) **sem erros**.
- Resultados exportados automaticamente para `outputs/` (CSVs).

## 9. Respostas diretas às 6 perguntas do enunciado

| Pergunta | Resposta |
|---|---|
| **Quais indicadores?** | `child_mort`, `income`, `acesso_agua_pct`, `alfabetizacao_pct` — escolhidos pelo heatmap de correlação para minimizar redundância; são facetas colineares de um mesmo fator de desenvolvimento (VIF até 8,75). |
| **Preenchimento dos faltantes?** | Imputação por **mediana**. KNN comparado em pipeline idêntico (ARI 0,96 ⇒ resultado praticamente igual). |
| **Tratamento dos outliers?** | Detecção por **IQR**; decisão de **MANTER** (são países genuinamente ricos/pobres). Por isso usamos `RobustScaler`. |
| **Tratamento da assimetria?** | **Sim.** `log1p` em `child_mort` e `income` (skew → ≈ 0). |
| **Houve escalonamento?** | **Sim.** `RobustScaler` (mediana + IQR). |
| **Métrica de distância?** | **Euclidiana (L2).** |
