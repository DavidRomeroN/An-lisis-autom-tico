"""
detector.py
-----------
Detección automática de separador, columna objetivo y clase positiva.
"""

import pandas as pd


TARGET_KEYWORDS = [
    "target", "label", "class", "outcome", "response",
    "churn", "deposit", "y", "resultado", "respuesta",
    "objetivo", "converted", "subscribed", "default", "exited",
]

POSITIVE_KEYWORDS = [
    "yes", "si", "sí", "true", "1", "positive", "success",
    "subscribed", "converted", "bueno", "aceptado",
]


def detect_separator(filepath: str, n_lines: int = 5) -> str:
    """Detecta el separador del CSV probando ',' ';' tabulación y pipe."""
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
        return True
    try:
        return set(vals).issubset({0, 1}) or set(vals).issubset({0.0, 1.0})
    except TypeError:
        return False


def infer_target_column(df: pd.DataFrame) -> str:
    """
    Infiere la columna objetivo:
      1. Nombre en TARGET_KEYWORDS
      2. Nombre que contiene keyword
      3. Última columna binaria (categórica o numérica 0/1)
      4. Última columna del dataframe
    """
    cols_lower = {c: c.lower() for c in df.columns}

    for col, low in cols_lower.items():
        if low in TARGET_KEYWORDS:
            return col

    for col, low in cols_lower.items():
        for kw in TARGET_KEYWORDS:
            if kw in low:
                return col

    for col in reversed(df.columns):
        if _is_binary_column(df[col]):
            return col

    return df.columns[-1]


def infer_positive_class(series: pd.Series):
    """Clase positiva: keywords conocidos o clase minoritaria."""
    vals = series.value_counts()

    if series.dtype == object:
        for v in vals.index:
            if str(v).strip().lower() in POSITIVE_KEYWORDS:
                return v

    numeric_vals = pd.to_numeric(series, errors="coerce")
    if numeric_vals.notna().all() and set(numeric_vals.unique()).issubset({0, 1, 0.0, 1.0}):
        return 1

    return vals.index[-1]


def load_dataset(filepath: str) -> tuple[pd.DataFrame, str, str, str]:
    """Carga CSV y retorna (df, separator, target_column, positive_class)."""
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
    pos = infer_positive_class(df[target])

    return df, sep, target, pos
