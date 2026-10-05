# Como cada figura responde às perguntas do professor

Este documento explica, **figura por figura**, **qual pergunta do enunciado** ela responde e
**como** (o que olhar e que conclusão ela sustenta). Serve de **legenda/roteiro do relatório**:
cada figura deve entrar logo depois do parágrafo que responde à pergunta correspondente.

> As figuras estão em `outputs/figuras/` e são geradas por `src/figuras_relatorio.py`.
> Convenção dos nomes: `Q1..Q9` = pergunta/etapa a que a figura responde.

---

## Resumo (mapa pergunta → figura)

| Pergunta do enunciado | Figura(s) |
|---|---|
| 1. Quais indicadores usou? | `Q1_indicadores_heatmap.png`, `Q1_indicadores_escolhidos.png` |
| 2. Quais métodos para preenchimento dos faltantes? | `Q2_faltantes.png`, `Q2_imputacao_antes_depois.png` |
| 3. Quais métodos para tratamento dos outliers? | `Q3_outliers_iqr.png` |
| 4. Foi necessário tratamento da assimetria? | `Q4_assimetria_log.png` |
| 5. Houve escalonamento? | `Q5_escalonamento_antes_depois.png` |
| 6. Qual métrica de distância foi usada? | `Q6_distancia.png` |
| (Aulas 5–6) Escolher o *k* | `Q7_cotovelo_silhueta.png`, `Q7_metricas_k.png` |
| (Aulas 5–6) Interpretar os grupos | `Q8_perfis.png` |
| (Aula 7) Agrupamento hierárquico | `Q9_dendrograma.png`, `Q9_ward_vs_kmeans.png` |

---

## Q1 — "Quais indicadores usou?"

**Figura `Q1_indicadores_heatmap.png`** — heatmap da correlação de Pearson (base limpa), com
as 17 colunas.
**Como responde:** mostra o **mapa de redundância** entre todas as variáveis. Dá para ver os
blocos de colunas que medem quase a mesma coisa (o bloco de acesso/infraestrutura no canto
inferior direito, por exemplo, todo vermelho-escuro). É a base da decisão dos 4 indicadores.

**Figura `Q1_indicadores_escolhidos.png`** — à esquerda, barras dos **pares rejeitados por
redundância** com o `|r|` ao lado e a linha de limiar `|r| = 0,8`; à direita, a lista dos
**4 escolhidos** e a ressalva de colinearidade.
**Como responde:** fecha a justificativa. Mostra que descartamos pares quase idênticos
(`acesso_agua × acesso_esgoto` = 0,96; `income × gdpp` = 0,89; `child_mort × life_expec` = −0,89)
e que sobraram **4 facetas** (saúde, renda, infraestrutura, educação). E é honesto: avisa que os
4 são **colineares entre si** (VIF até 8,8; 1º componente = 82,7%) — são facetas de um mesmo
fator de desenvolvimento, não dimensões independentes.

> **Conclusão que a figura sustenta:** usamos `child_mort`, `income`, `acesso_agua_pct` e
> `alfabetizacao_pct`, escolhidos pelo heatmap para **reduzir redundância**, reconhecendo a
> colinearidade do dataset.

---

## Q2 — "Quais métodos foram usados para preenchimento dos faltantes?"

**Figura `Q2_faltantes.png`** — barras da quantidade de faltantes por coluna.
**Como responde:** quantifica o **problema**: quais colunas tinham buracos e quantos. O título
já resume o total: **34 valores** (15 faltantes reais + 19 sentinelas/impossíveis).

**Figura `Q2_imputacao_antes_depois.png`** — para as colunas com faltantes, histograma
**ANTES (com NaN) × DEPOIS (imputação pela mediana)**, com a **mesma escala** nos dois painéis
e a linha da mediana.
**Como responde:** mostra o **método** (mediana) e o **efeito**: a forma da distribuição é
preservada e o valor imputado cai no centro. Justifica por que a mediana é adequada — ela é
**robusta a outliers** (ao contrário da média), e são poucos valores.

> **Conclusão:** preenchimento por **imputação pela mediana** (alternativa KNN testada, com
> ARI ≈ 0,96 — o agrupamento praticamente não muda).

---

## Q3 — "Quais métodos foram usados para tratamento dos outliers?"

**Figura `Q3_outliers_iqr.png`** — boxplots das 4 variáveis com as **cercas do IQR**
(`Q1 − 1,5·IQR` e `Q3 + 1,5·IQR`) desenhadas e os pontos fora marcados; o `%` de outliers por
variável aparece no título de cada painel (`income` em escala log).
**Como responde:** mostra **o método de detecção** (IQR) e a **decisão** — os extremos são
**mantidos**. A leitura visual: os pontos fora das cercas são poucos e são **países
genuinamente muito ricos ou muito pobres**, que são justamente o público-alvo da ONG; removê-los
apagaria o objeto do estudo. Erros grosseiros (sentinela/impossíveis) já foram tratados na limpeza.

> **Conclusão:** detecção por **IQR**; decisão de **manter** os outliers — o que amarra com a
> escolha do `RobustScaler` (Q5).

---

## Q4 — "Foi necessário um tratamento da assimetria dos dados?"

**Figura `Q4_assimetria_log.png`** — 2×2: histogramas de `child_mort` e `income`
**ANTES (com o `skew` anotado) × DEPOIS do `log1p`** (com o novo `skew`).
**Como responde:** mostra que **sim, foi necessário**: antes os dados têm cauda longa à direita
(`income` skew ≈ 2,2; `child_mort` ≈ 1,5); depois do `log1p`, a cauda encolhe e o `skew` cai
para perto de 0 (≈ −0,25 e ≈ 0,07). A mensagem da aula fica visível: **o log muda a FORMA** da
distribuição.

> **Conclusão:** **sim**, tratamos a assimetria com `log1p` em `child_mort` e `income`. E o
> ponto didático: **escalonar não resolve assimetria** — log (forma) e scaler (escala) são etapas
> diferentes, nesta ordem.

---

## Q5 — "Houve escalonamento?"

**Figura `Q5_escalonamento_antes_depois.png`** — grade 2×4: cada variável **ANTES (unidade
bruta)** × **DEPOIS do `RobustScaler`** (linha tracejada em 0).
**Como responde:** mostra que **sim**: depois do escalonamento, todas as 4 variáveis passam a
ter **mediana 0 e IQR 1** — a "régua" fica igual para todas. Sem isso, `income` (que varia de
centenas a ~120.000) dominaria a distância só por causa da grandeza dos números. Justifica
também o **RobustScaler** e não o z-score: como decidimos **manter os outliers** (Q3), o
`RobustScaler` (mediana + IQR) é o coerente, porque não se deixa arrastar por eles.

> **Conclusão:** **sim**, escalonamento com **`RobustScaler`**, coerente com a decisão de manter
> os outliers.

---

## Q6 — "Qual métrica de distância foi usada?"

**Figura `Q6_distancia.png`** — contribuição de cada variável para a distância **em %**,
**SEM escalonar × COM z-score**, no par país mais rico × mais pobre.
**Como responde:** mostra **qual métrica** (Euclidiana, L2) e **por que** o escalonamento
importa para ela. No painel sem escalonar, a renda concentra quase **100%** da distância (as
outras variáveis somem); com z-score, a contribuição se distribui entre as 4. É a prova visual
de que a **Euclidiana só faz sentido com variáveis na mesma escala**.

> **Conclusão:** métrica **Euclidiana (L2)** — padrão para dados contínuos, poucas dimensões e
> escalonados; combina com o K-Means e com o Ward.

---

## Aulas 5–6 — escolher o *k* e interpretar os grupos

**Figura `Q7_cotovelo_silhueta.png`** — inércia por *k* (cotovelo) e silhueta por *k*, com a
linha em `k = 3`.
**Como responde:** mostra a **evidência da escolha de `k`**: o cotovelo (maior queda de inércia
em 2→3) e a silhueta (máxima em k=2, mas só +0,07 em relação a k=3). Título afirma a decisão.

**Figura `Q7_metricas_k.png`** — silhueta, Calinski-Harabasz e Davies-Bouldin por *k*.
**Como responde:** reforça a escolha — **CH maximiza em k=3** (evidência de que 3 grupos se
separam bem), completando o quadro.

**Figura `Q8_perfis.png`** — à esquerda, o perfil padronizado dos 3 grupos; à direita, os
países coloridos pelos 3 clusters (`child_mort × income`).
**Como responde:** **interpreta os grupos**: o Perfil 1 é "ruim em tudo" (alta mortalidade,
renda baixa), o Perfil 3 é "bom em tudo" e o Perfil 2 fica no meio — um **eixo único e ordenado
de necessidade**. É o que vira a recomendação à diretoria.

> **Conclusão:** **k = 3**, estável (ARI entre sementes ≈ 0,96), interpretável como 3 níveis de
> necessidade.

---

## Aula 7 — agrupamento hierárquico

**Figura `Q9_dendrograma.png`** — dendrograma do Ward com a **linha de corte em `k = 3`**.
**Como responde:** mostra a **leitura do dendrograma** — e o cuidado central da aula: **a
altura NÃO é o número de grupos**; o `k` sai de onde a linha de corte cruza os galhos.

**Figura `Q9_ward_vs_kmeans.png`** — heatmap da concordância **K-Means × Ward** com o **ARI**
anotado.
**Como responde:** **compara os métodos**. O ARI mostra que os dois concordam no eixo principal
(pobre × rico) e divergem no corte do nível intermediário — ou seja, a fronteira do Perfil 2 é
uma **zona de transição**, não uma linha rígida.

> **Conclusão:** o hierárquico (Ward, corte em k=3) **confirma** a estrutura encontrada pelo
> K-Means, com a ressalva honesta sobre a fronteira do grupo intermediário.

---

## Como usar no relatório

1. Na resposta de cada pergunta, **cole a figura correspondente logo abaixo** do parágrafo.
2. Use a frase do **título da figura** como **legenda** — ela já é a resposta.
3. Para a Recomendação à diretoria, use a `Q8_perfis.png` (perfil dos 3 grupos).

> Observação: a figura `Q6_distancia.png` fala de distância **com z-score** apenas para
> **ilustrar** o efeito da escala; no pipeline final usamos `RobustScaler` (Q5). Mantenha essa
> nota no relatório para não parecer incoerência.
