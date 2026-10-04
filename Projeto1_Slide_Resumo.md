---
marp: true
theme: default
paginate: true
title: Priorização de ajuda humanitária — Help International
---

# Priorização de ajuda humanitária
## Perfis de países para alocar um fundo limitado

**Mineração de Dados — Projeto 1**
Equipe de Ciência de Dados da ONG *Help International*

---

# O problema

- Fundo **limitado**, demanda ilimitada.
- **Não** queremos um ranking: queremos **perfis de necessidade** para orientar **estratégias diferentes**.
- Base: **167 países**, indicadores socioeconômicos e de saúde.

> "Boas escolhas de dados e de método valem mais do que muitos gráficos."

---

# Os 4 indicadores (e por quê)

| Indicador | Faceta |
|---|---|
| `child_mort` | Saúde infantil |
| `income` | Renda |
| `acesso_agua_pct` | Saneamento/infra |
| `alfabetizacao_pct` | Educação |

**Escolha por correlação** para evitar redundância.
**Ressalva honesta:** os 4 são **colineares** (VIF até 8,8; 1º componente = 82,7%) —
são **facetas de um mesmo fator de desenvolvimento**, não dimensões independentes.

---

# Pré-processamento (as 6 decisões)

| Etapa | Decisão |
|---|---|
| **Faltantes** | Sentinela/impossíveis → NaN; imputação por **mediana** (KNN testado: ARI 0,96) |
| **Outliers** | Detectados por IQR; **mantidos** (são países ricos/pobres reais) |
| **Assimetria** | **Sim**: `log1p` em renda e mortalidade infantil |
| **Escalonamento** | **Sim**: `RobustScaler` (coerente com manter outliers) |
| **Distância** | **Euclidiana (L2)** |
| **Duplicatas** | Nenhuma |

---

# Onde os dados brutos enganam

- **Sentinelas** `99999` (e `999` da Poland) fingem ser valores reais.
- **Impossíveis:** Brasil com 150% de acesso à água, Quênia com 240% de alfabetização.
- **Cuidado de domínio:** em `cesta_basica`, **> 100 é legítimo** (cesta custa mais de 1 salário mínimo).

> Se o `99999` ficasse, destruiria média, desvio e escala.

---

# Escolha de k: 3 perfis

- Cotovelo e **Calinski-Harabasz** → **k = 3**
- Silhueta e Davies-Bouldin → k = 2 (diferença pequena, 0,07)
- **Desempate:** k=2 junta países de renda 4.240 a 119.000 (**pouco acionável**);
  k=3 dá **3 níveis de necessidade** úteis para a ONG.
- **Estável:** ARI médio 0,96 entre sementes.

---

# Os 3 perfis

| Perfil | n | Mort. inf. (‰) | Renda (US$) | Água (%) | Alfab. (%) |
|---|---|---|---|---|---|
| **1 · Alta necessidade** | 48 | 88,8 | 1.860 | 55 | 61 |
| **2 · Intermediária** | 59 | 21,5 | 9.720 | 73 | 78 |
| **3 · Baixa necessidade** | 60 | 5,8 | 31.350 | 91 | 91 |

Eixo **único e ordenado** de necessidade (do mais crítico ao mais capaz).

---

# Perfil 1 — PRIORIDADE MÁXIMA (48 países)

Saúde, renda, água e educação **todas baixas**.

**Estratégia:** maior parcela do fundo → **saúde infantil + água/saneamento + renda**. Longo prazo.

**Exemplos:** Haiti, Serra Leoa, Chade, Mali, Níger, Moçambique, Afeganistão, Iêmen, Nepal, Tanzânia.

---

# Perfil 2 — PRIORIDADE MÉDIA (59 países)

**Estratégia:** infraestrutura de acesso (água/esgoto) + educação; apoio a saúde pública.

**Exemplos:** Bangladesh, Bolívia, Iraque, Indonésia, Filipinas, Egito, Vietnã, Brasil, Colômbia.

---

# Perfil 3 — PRIORIDADE BAIXA (60 países)

**Estratégia:** apoio pontual / institucional, em parceria local. Sem ajuda emergencial.

**Exemplos:** EUA, Alemanha, Japão, Canadá, França, Reino Unido, Noruega, Espanha.

---

# Comparação com Ward (rigor)

- K-Means (k=3): silhueta **0,465** | Ward (k=3): silhueta 0,432 → **K-Means vence**.
- Concordância K-Means × Ward: **ARI = 0,42** (em k=2 era 0,95).

**Leitura:** os métodos concordam no eixo pobre×rico, mas divergem no corte do
nível intermediário → **Perfil 2 é zona de transição**.

> Transparência: o 3º grupo **não é robusto entre métodos** — por isso usamos os
> perfis como **priorização**, não como classificação rígida.

---

# Limitações

1. Indicadores **colineares** → grupos = níveis de desenvolvimento.
2. Fronteira do Perfil 2 é **difusa** (ARI 0,42).
3. Dados de **um único período** (sem tendência).
4. Priorização por **necessidade**, não por **custo-efetividade**.

---

# Recomendação final

1. **Priorizar o Perfil 1** (48 países) — concentra toda a vulnerabilidade.
2. **Perfil 2** (59) — apoio intermediário em infraestrutura/educação.
3. **Perfil 3** (60) — fora da ajuda emergencial.

**Produto:** tipologia de perfis (não ranking), pronta para a alocação do fundo.

---

# Notas de metodologia (reprodutibilidade)

- Notebook roda do zero (*Run All*), **sem erros**, semente fixa em todo o pipeline.
- Arquivos: `notebooks/Projeto1_Mineracao_Pipeline_Final.ipynb`
- Saídas: `outputs/07_paises_clusters.csv` (167 países × cluster × perfil)

**Obrigado!**
