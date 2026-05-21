"""
detector.py
-----------
Detección automática de separador, columna objetivo y clase positiva.
Corregido: 'y' solo coincide con columna exacta (no studytime/higher).
Student Performance: prioriza G3 y binariza aprobado (>= 10).
"""

import pandas as pd
import numpy as np

# Coincidencia exacta del nombre de columna (minúsculas)
TARGET_EXACT = {
    "y", "target", "label", "class", "outcome", "response",
    "churn", "deposit", "resultado", "respuesta", "objetivo",
    "converted", "subscribed", "default", "default_payment_next_month",
    "exited",
}

# Subcadena solo para palabras largas (evita 'y' en studytime)
TARGET_CONTAINS = [
    "target", "label", "outcome", "response", "churn", "deposit",
    "resultado", "respuesta", "converted", "subscribed", "default",
]

GRADE_COLUMNS = ("g3", "g2", "g1")
GRADE_PASS_THRESHOLD = 10

POSITIVE_KEYWORDS = {
    "yes", "si", "sí", "true", "1", "positive", "success",
    "subscribed", "converted", "bueno", "aceptado",
}

YES_NO_COLUMNS = (
    "schoolsup", "famsup", "paid", "activities", "nursery",
    "higher", "internet", "romantic",
)


def detect_separator(filepath: str, n_lines: int = 5) -> str:
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            with open(filepath, "r", encoding=encoding, errors="replace") as f:
                sample = "".join([f.readline() for _ in range(n_lines)])
            break
        except OSError:
            sample = ""
    counts = {sep: sample.count(sep) for sep in [",", ";", "\t", "|"]}
    return max(counts, key=counts.get) if any(counts.values()) else ","


def _is_binary_column(series: pd.Series) -> bool:
    vals = series.dropna().unique()
    if len(vals) != 2:
        return False
    if series.dtype == object:
        lowered = {str(v).strip().lower() for v in vals}
        return lowered <= {"yes", "no"} or lowered <= {"0", "1"} or lowered <= {"true", "false"}
    try:
        return set(vals).issubset({0, 1}) or set(vals).issubset({0.0, 1.0})
    except TypeError:
        return False


def _is_grade_column(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() < len(series) * 0.8:
        return False
    vals = numeric.dropna()
    return vals.min() >= 0 and vals.max() <= 20 and vals.nunique() > 2


def infer_target_column(df: pd.DataFrame) -> str:
    """
    Prioridad:
      1. Nombre exacto (y, g3, target, default, ...)
      2. Student: G3 > G2 > G1
      3. Columnas yes/no conocidas al final
      4. Subcadena (palabras largas, nunca 'y' suelto)
      5. Última columna binaria yes/no
      6. Última columna del archivo
    """
    cols_lower = {c: c.lower() for c in df.columns}

    # Notas finales: G3 > G2 > G1 (Student Performance)
    for grade in GRADE_COLUMNS:
        if grade in cols_lower.values():
            col = next(c for c, l in cols_lower.items() if l == grade)
            if _is_grade_column(df[col]):
                return col

    for col, low in cols_lower.items():
        if low in TARGET_EXACT:
            return col

    for name in YES_NO_COLUMNS:
        if name in cols_lower.values():
            col = next(c for c, l in cols_lower.items() if l == name)
            if _is_binary_column(df[col]):
                return col

    for col, low in cols_lower.items():
        for kw in TARGET_CONTAINS:
            if kw in low:
                return col

    for col in reversed(df.columns):
        if _is_binary_column(df[col]):
            return col

    return df.columns[-1]


def infer_positive_class(series: pd.Series, target_col: str):
    """Clase positiva para columnas yes/no o 0/1."""
    vals = series.value_counts()

    if series.dtype == object:
        for v in vals.index:
            if str(v).strip().lower() in POSITIVE_KEYWORDS:
                return v

    numeric_vals = pd.to_numeric(series, errors="coerce")
    if numeric_vals.notna().all() and set(numeric_vals.unique()).issubset({0, 1, 0.0, 1.0}):
        return 1

    return vals.index[-1]


def build_binary_target(
    df: pd.DataFrame, target_col: str, positive_class=None
) -> tuple[pd.Series, str, str]:
    """
    Construye y binaria (0/1) y descripción legible.
    Retorna: (y, positive_label, target_description)
    """
    series = df[target_col]
    low = target_col.lower()

    if low in GRADE_COLUMNS or _is_grade_column(series):
        numeric = pd.to_numeric(series, errors="coerce")
        y = (numeric >= GRADE_PASS_THRESHOLD).astype(int)
        desc = (
            f"{target_col}: aprobado si nota >= {GRADE_PASS_THRESHOLD} "
            f"({int(y.sum())} aprobados de {len(y)})"
        )
        return y, "aprobado (1)", desc

    if series.dtype == object:
        pos = positive_class if positive_class is not None else infer_positive_class(series, target_col)
        pos_norm = str(pos).strip().lower()
        y = series.astype(str).str.strip().str.lower().eq(pos_norm).astype(int)
        desc = f"{target_col}: positivo = '{pos}'"
        return y, str(pos), desc

    numeric = pd.to_numeric(series, errors="coerce")
    if set(numeric.dropna().unique()).issubset({0, 1, 0.0, 1.0}):
        y = (numeric == 1).astype(int)
        return y, "1", f"{target_col}: positivo = 1"

    pos = positive_class if positive_class is not None else infer_positive_class(series, target_col)
    y = (series == pos).astype(int)
    if y.nunique() < 2:
        minority = series.value_counts().index[-1]
        y = (series == minority).astype(int)
        pos = minority
    desc = f"{target_col}: positivo = {pos}"
    return y, str(pos), desc


def load_dataset(filepath: str) -> tuple[pd.DataFrame, str, str, str, str]:
    """
    Retorna (df, separator, target_column, positive_class, target_description).
    """
    sep = detect_separator(filepath)
    df = None
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(filepath, sep=sep, encoding=encoding)
            break
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    if df is None:
        df = pd.read_csv(filepath, sep=sep, encoding="latin-1", on_bad_lines="skip")

    df.columns = [
        str(c).strip().lower().replace(" ", "_").replace(".", "_")
        for c in df.columns
    ]

    target = infer_target_column(df)
    _, pos_label, desc = build_binary_target(df, target)
    pos = pos_label

    return df, sep, target, pos, desc
