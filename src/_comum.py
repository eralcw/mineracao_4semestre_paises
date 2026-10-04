"""Utilidades compartilhadas -- Projeto 1 (Mineracao de Dados).

Reproduz EXATAMENTE a logica do notebook
`notebooks/Projeto1_Mineracao_Pipeline_Final.ipynb`:
limpar (sentinela/regras de dominio -> NaN) -> imputar mediana ->
log1p nas assimetricas -> RobustScaler.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent          # raiz do repositorio
FIG_DIR = RAIZ / "outputs" / "figuras"
OUT_DIR = RAIZ / "outputs"

SEED = 42

# ---------------------------------------------------------------------------
# Codigos de "sem dado" (faltantes disfarcados)
# ---------------------------------------------------------------------------
SENTINELAS = [99999, 100000, 999]

# ---------------------------------------------------------------------------
# Valores fora da faixa valida (viram NaN)
# ---------------------------------------------------------------------------
DOMAIN_RULES = {
    "health":                   lambda s: (s < 0) | (s > 100),
    "child_mort":               lambda s: s < 0,
    "life_expec":               lambda s: (s <= 0) | (s > 120),
    "total_fer":                lambda s: (s < 0) | (s > 20),   # Argentina=45 fica de fora
    "acesso_agua_pct":          lambda s: (s < 0) | (s > 100),   # Brazil=150
    "acesso_esgoto_pct":        lambda s: (s < 0) | (s > 100),
    "acesso_energia_pct":       lambda s: (s < 0) | (s > 100),
    "alfabetizacao_pct":        lambda s: (s < 0) | (s > 100),   # Kenya=240
    "internet_pct":             lambda s: (s < 0) | (s > 100),   # Peru=-12
    "medicos_por_1000":         lambda s: s < 0,
    "extrema_pobreza_pct":      lambda s: (s < 0) | (s > 100),   # India=-4
    "cesta_basica_salario_pct": lambda s: s < 0,                 # >100 e LEGITIMO
}

# Os 4 indicadores escolhidos (4 facetas de necessidade)
FEATURES = ["child_mort", "income", "acesso_agua_pct", "alfabetizacao_pct"]
# Assimetricos que recebem log1p
LOG_FEATURES = ["child_mort", "income"]


# ---------------------------------------------------------------------------
# Funcoes
# ---------------------------------------------------------------------------
def carregar_bruto():
    """Le o CSV bruto, procurando em caminhos comuns. Devolve `df_raw`."""
    candidatos = [
        RAIZ / "data" / "paises_help_international.csv",
        Path("data/paises_help_international.csv"),
        Path("../data/paises_help_international.csv"),
        Path("./data/paises_help_international.csv"),
        Path("/content/paises_help_international.csv"),
    ]
    caminho = next((p for p in candidatos if p.exists()), None)
    if caminho is None:
        raise FileNotFoundError(
            "Coloque 'paises_help_international.csv' na pasta data/ do projeto."
        )
    return pd.read_csv(caminho)


def colunas_numericas(df):
    """Lista as colunas numericas do DataFrame."""
    return df.select_dtypes(include=np.number).columns.tolist()


def limpar(df_raw):
    """Copia, sentinelas -> NaN, aplica DOMAIN_RULES (mask -> NaN). Mesma logica do notebook."""
    df = df_raw.copy()
    num = colunas_numericas(df)

    # 1) sentinelas -> NaN
    df[num] = df[num].replace(SENTINELAS, np.nan)

    # 2) valores impossiveis -> NaN
    for col, regra in DOMAIN_RULES.items():
        if col in df.columns:
            mask = regra(df[col]) & df[col].notna()
            df.loc[mask, col] = np.nan

    return df


def matriz_modelo(df):
    """Sobre FEATURES: imputa por mediana -> log1p nas assimetricas -> RobustScaler.

    Devolve um DataFrame padronizado, pronto para o K-Means.
    """
    X_before = df[FEATURES].copy()

    # 1) imputacao por mediana
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X_before), columns=FEATURES, index=df.index
    )

    # 2) log1p nas assimetricas (clip em 0 por seguranca)
    X_transformed = X_imputed.copy()
    for col in LOG_FEATURES:
        X_transformed[col] = np.log1p(X_transformed[col].clip(lower=0))

    # 3) RobustScaler (mediana + IQR)
    X_scaled = RobustScaler().fit_transform(X_transformed)
    return pd.DataFrame(X_scaled, columns=FEATURES, index=df.index)


def salvar(fig, nome):
    """Cria FIG_DIR se preciso e salva a figura em PNG (150 dpi). Devolve o caminho."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    caminho = FIG_DIR / nome
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    print(f"[figura] {caminho}")
    return caminho
