# Projeto 1 — Agrupamento de países (ONG *Help International*)

Mineração de Dados — pipeline que agrupa **167 países** em **3 perfis de necessidade**
(alta / intermediária / baixa) para orientar a alocação de um fundo humanitário limitado.

> **Não é um ranking.** É uma **tipologia de perfis**: cada grupo aponta uma **estratégia
> diferente** de ajuda, não uma ordem de prioridade país a país.

---

## Visão geral

A partir de indicadores socioeconômicos e de saúde, escolhemos **4 indicadores** e aplicamos
**K-Means (k=3)** + **agrupamento hierárquico (Ward)** para produzir os perfis.

**Os 4 indicadores escolhidos** (por correlação, evitando redundância):

| Indicador | Faceta |
|---|---|
| `child_mort` | Saúde infantil |
| `income` | Renda / capacidade econômica |
| `acesso_agua_pct` | Infraestrutura / saneamento |
| `alfabetizacao_pct` | Educação |

**Ressalva honesta:** os 4 são **colineares entre si** (VIF até 8,8; 1º componente = 82,7%
da variância) — são **facetas de um mesmo fator de desenvolvimento**, não dimensões
independentes.

**Resultado:**

| Perfil | n | Mort. inf. (‰) | Renda (US$) | Água (%) | Alfab. (%) |
|---|---|---|---|---|---|
| 1 · Alta necessidade | 48 | 88,8 | 1.860 | 55 | 61 |
| 2 · Intermediária | 59 | 21,5 | 9.720 | 73 | 78 |
| 3 · Baixa necessidade | 60 | 5,8 | 31.350 | 91 | 91 |

---

## Estrutura do repositório

```
.
├── data/
│   └── paises_help_international.csv      # base bruta (167 × 18)
├── notebooks/
│   └── Projeto1_Mineracao_Pipeline_Final.ipynb   # pipeline completo, do zero ao fim
├── src/
│   ├── _comum.py                          # limpeza + pipeline compartilhados
│   └── figuras_relatorio.py               # gera as figuras Q1..Q9 (respostas)
├── outputs/
│   ├── 0X_*.csv                           # resultados do notebook (métricas, clusters, etc.)
│   ├── dados_limpos.csv                   # base limpa
│   └── figuras/
│       └── Q1..Q9_*.png                   # figuras que RESPONDEM às perguntas
├── Projeto1_Relatorio.md                  # relatório (≤ 10 páginas)
├── COMO_AS_FIGURAS_RESPONDEM.md           # legenda: figura → pergunta respondida
├── requirements.txt
└── README.md                              # este arquivo
```

---

## Pré-requisitos

- **Python 3.10+**
- As dependências em `requirements.txt` (`pandas`, `numpy`, `matplotlib`, `scikit-learn`,
  `scipy`, `jupyterlab`, `ipykernel`).

---

## Como usar

### 1) Criar o ambiente virtual e instalar dependências

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Linux / macOS:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Rodar o pipeline completo (notebook)

```powershell
.\.venv\Scripts\python.exe -m jupyter lab notebooks/Projeto1_Mineracao_Pipeline_Final.ipynb
```

Abra o notebook e use **Run All** (*Kernel → Restart Kernel and Run All Cells*).
Ele roda do zero, sem erros, com semente fixa (`SEED = 42`), e grava os resultados em
`outputs/`.

> Também é possível rodar sem abrir a interface (executa e regrava o notebook):
> ```powershell
> .\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace `
>   notebooks/Projeto1_Mineracao_Pipeline_Final.ipynb
> ```

### 3) Gerar as figuras do relatório (Q1..Q9)

As figuras que **respondem diretamente às 6 perguntas** (+ escolha de k, perfis e hierárquico)
são geradas por um script único e autossuficiente:

```powershell
.\.venv\Scripts\python.exe src/figuras_relatorio.py
```

Ele grava as **13 figuras** em `outputs/figuras/` (PNG, 150 dpi) e imprime o caminho de cada uma.
Para entender **qual pergunta cada figura responde**, veja **`COMO_AS_FIGURAS_RESPONDEM.md`**.

---

## As 6 perguntas do enunciado → onde estão respondidas

| # | Pergunta | Figuras | Seções |
|---|---|---|---|
| 1 | Quais indicadores usou? | `Q1_indicadores_heatmap`, `Q1_indicadores_escolhidos` | Relatório §4 |
| 2 | Preenchimento dos faltantes | `Q2_faltantes`, `Q2_imputacao_antes_depois` | Relatório §3 |
| 3 | Tratamento dos outliers | `Q3_outliers_iqr` | Relatório §3 |
| 4 | Tratamento da assimetria | `Q4_assimetria_log` | Relatório §3 |
| 5 | Houve escalonamento? | `Q5_escalonamento_antes_depois` | Relatório §3 |
| 6 | Métrica de distância | `Q6_distancia` | Relatório §3 |
| — | Escolha de *k* | `Q7_cotovelo_silhueta`, `Q7_metricas_k` | Relatório §5 |
| — | Interpretar os grupos | `Q8_perfis` | Relatório §6 |
| — | Hierárquico (Ward) | `Q9_dendrograma`, `Q9_ward_vs_kmeans` | Relatório §5 |

O resumo consolidado das 6 perguntas está no **§9 do relatório**. A legenda figura → pergunta
(a explicação de **o que olhar** em cada figura) está em **`COMO_AS_FIGURAS_RESPONDEM.md`**.

---

## Notas de método (o que o código garante)

- **Ordem canônica:** limpar → tratar outliers → log (assimetria) → escalonar → modelar.
- **Limpeza:** sentinelas (`99999`, `100000`, `999`) e valores impossíveis → `NaN`;
  faltantes → imputação pela **mediana**.
- **Assimetria:** `log1p` em `child_mort` e `income`.
- **Escalonamento:** **`RobustScaler`** (coerente com **manter** os outliers).
- **Distância:** **Euclidiana (L2)**.
- **K-Means:** `n_init=50`, `random_state=42`; **hierárquico:** Ward, corte por
  `fcluster(..., criterion="maxclust")`.
- **Reprodutibilidade:** roda do zero, sem erros, com semente fixa em todo o pipeline.

---

## Limitações (transparência)

1. Indicadores **colineares** → os grupos são níveis de um mesmo fator de desenvolvimento.
2. A fronteira do **Perfil 2** é **difusa** (ARI K-Means × Ward = 0,42) — é uma **zona de
   transição**, não uma linha rígida.
3. Dados de **um único período** (sem tendência temporal).
4. Priorização por **necessidade**, não por **custo-efetividade**.

---

## Exportar o relatório para PDF (opcional)

No VS Code com a extensão **Markdown PDF**:

- `Projeto1_Relatorio.md` → PDF via *Markdown PDF: Export (pdf)*.

As figuras citadas no relatório estão em `outputs/figuras/` (`Q1..Q9`).
