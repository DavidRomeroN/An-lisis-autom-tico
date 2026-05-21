"""
cleaning.py
-----------
Limpieza universal antes del modelado (Bank Marketing, crédito, churn, etc.).
"""

import pandas as pd
import numpy as np

MISSING_TOKENS = {
    "", "?", "na", "n/a", "null", "none", "nan",
    "unknown", "desconocido", "missing", "-", ".",
}

# Columnas típicas que no aportan al modelo
DROP_NAME_PATTERNS = ("unnamed", "index", "id_cliente", "customer_id")


def normalize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte marcadores de faltante (p. ej. 'unknown' en Bank Marketing) a NaN."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            s = df[col].astype(str).str.strip()
            mask = s.str.lower().isin(MISSING_TOKENS)
            df.loc[mask, col] = np.nan
    return df


def strip_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"nan": np.nan, "None": np.nan})
    return df


def coerce_numeric_columns(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Intenta convertir columnas object con números (edad, balance, etc.)."""
    df = df.copy()
    for col in df.columns:
        if col == target_col:
            continue
        if df[col].dtype != object:
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() >= 0.5 * len(df):
            df[col] = converted
    return df


def drop_noise_columns(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Elimina columnas ID, unnamed y constantes."""
    df = df.copy()
    drop_cols = []
    n = len(df)

    for col in df.columns:
        if col == target_col:
            continue
        low = str(col).lower()
        if any(p in low for p in DROP_NAME_PATTERNS):
            drop_cols.append(col)
            continue
        nunique = df[col].nunique(dropna=False)
        if nunique <= 1:
            drop_cols.append(col)
        elif nunique >= 0.95 * n:
            drop_cols.append(col)

    if drop_cols:
        df = df.drop(columns=list(set(drop_cols)))
    return df


def full_clean(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, dict]:
    """
    Pipeline de limpieza completo.
    Retorna (df_limpio, reporte de acciones).
    """
    report = {
        "filas_inicial": len(df),
        "nulos_antes": int(df.isnull().sum().sum()),
        "columnas_inicial": df.shape[1],
        "duplicados_eliminados": 0,
        "columnas_eliminadas": [],
    }

    df = strip_string_columns(df)
    df = normalize_missing_values(df)
    df = coerce_numeric_columns(df, target_col)

    n_dup = df.duplicated().sum()
    if n_dup:
        df = df.drop_duplicates()
        report["duplicados_eliminados"] = int(n_dup)

    before_cols = set(df.columns)
    df = drop_noise_columns(df, target_col)
    report["columnas_eliminadas"] = sorted(before_cols - set(df.columns))
    report["nulos_despues_marcado"] = int(df.isnull().sum().sum())
    report["filas_final"] = len(df)
    report["columnas_final"] = df.shape[1]

    return df, report
