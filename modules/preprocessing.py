"""
preprocessing.py
----------------
Preprocesamiento sin data leakage:
  - Limpieza (duplicados, IDs, columnas constantes)
  - Partición Train / Val / Test (80% desarrollo · 20% prueba por defecto)
  - Imputación y One-Hot ajustados SOLO en train
  - StandardScaler ajustado SOLO en train
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.20
DEFAULT_VAL_FRAC_OF_TRAIN = 0.15


def clean_dataframe(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Limpieza básica aplicada antes de particionar."""
    df = df.copy()
    df = df.drop_duplicates()

    drop_cols = []
    for col in df.columns:
        if col == target_col:
            continue
        nunique = df[col].nunique(dropna=False)
        if nunique <= 1:
            drop_cols.append(col)
        elif nunique >= 0.95 * len(df):
            drop_cols.append(col)

    if drop_cols:
        df = df.drop(columns=drop_cols)

    return df


def _column_types(df: pd.DataFrame, target_col: str) -> tuple[list, list]:
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    if target_col in num_cols:
        num_cols.remove(target_col)
    if target_col in cat_cols:
        cat_cols.remove(target_col)

    return num_cols, cat_cols


def _build_y(df: pd.DataFrame, target_col: str, positive_class) -> pd.Series:
    series = df[target_col]
    if series.dtype == object:
        pos_norm = str(positive_class).strip().lower()
        y = series.astype(str).str.strip().str.lower().eq(pos_norm).astype(int)
    else:
        y = (series == positive_class).astype(int)

    if y.nunique() < 2:
        vals = series.dropna().unique()
        if len(vals) >= 2:
            minority = series.value_counts().index[-1]
            y = (series == minority).astype(int)

    return y


def _impute_frame(df: pd.DataFrame, num_cols: list, cat_cols: list,
                  num_stats: dict, cat_stats: dict) -> pd.DataFrame:
    df = df.copy()
    for col in num_cols:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(num_stats.get(col, 0))
    for col in cat_cols:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(cat_stats.get(col, "missing"))
    return df


def _fit_imputation_stats(train_df: pd.DataFrame,
                          num_cols: list, cat_cols: list) -> tuple[dict, dict]:
    num_stats = {}
    for col in num_cols:
        if col in train_df.columns:
            num_stats[col] = train_df[col].median()
    cat_stats = {}
    for col in cat_cols:
        if col in train_df.columns:
            mode = train_df[col].mode()
            cat_stats[col] = mode.iloc[0] if len(mode) else "missing"
    return num_stats, cat_stats


def _encode_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                  cat_cols: list) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if cat_cols:
        X_train = pd.get_dummies(train_df, columns=cat_cols, drop_first=True)
        X_val   = pd.get_dummies(val_df,   columns=cat_cols, drop_first=True)
        X_test  = pd.get_dummies(test_df,  columns=cat_cols, drop_first=True)

        all_cols = sorted(set(X_train.columns) | set(X_val.columns) | set(X_test.columns))
        X_train = X_train.reindex(columns=all_cols, fill_value=0)
        X_val   = X_val.reindex(columns=all_cols, fill_value=0)
        X_test  = X_test.reindex(columns=all_cols, fill_value=0)
    else:
        X_train, X_val, X_test = train_df, val_df, test_df

    return (
        X_train.fillna(0),
        X_val.fillna(0),
        X_test.fillna(0),
    )


def prepare_dataset(df: pd.DataFrame, target_col: str, positive_class,
                    test_size: float = DEFAULT_TEST_SIZE,
                    val_frac_of_train: float = DEFAULT_VAL_FRAC_OF_TRAIN) -> dict:
    """
    Pipeline completo: limpieza → partición → transformaciones sin leakage.

    Por defecto: 20% test; validación = val_frac_of_train del bloque de desarrollo.
    """
    df = clean_dataframe(df, target_col)
    num_cols, cat_cols = _column_types(df, target_col)
    y = _build_y(df, target_col, positive_class)

    feature_df = df.drop(columns=[target_col])
    min_class = y.value_counts().min()
    use_stratify = min_class >= 6

    def do_split(stratify: bool):
        strat = y if stratify else None
        X_dev, X_test, y_dev, y_test = train_test_split(
            feature_df, y,
            test_size=test_size,
            stratify=strat,
            random_state=RANDOM_STATE,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_dev, y_dev,
            test_size=val_frac_of_train,
            stratify=y_dev if stratify else None,
            random_state=RANDOM_STATE,
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    try:
        parts = do_split(use_stratify)
    except ValueError:
        use_stratify = False
        parts = do_split(False)

    X_train, X_val, X_test, y_train, y_val, y_test = parts

    if y_test.nunique() < 2:
        use_stratify = False
        X_train, X_val, X_test, y_train, y_val, y_test = do_split(False)

    num_stats, cat_stats = _fit_imputation_stats(X_train, num_cols, cat_cols)

    X_train = _impute_frame(X_train, num_cols, cat_cols, num_stats, cat_stats)
    X_val   = _impute_frame(X_val,   num_cols, cat_cols, num_stats, cat_stats)
    X_test  = _impute_frame(X_test,  num_cols, cat_cols, num_stats, cat_stats)

    X_train, X_val, X_test = _encode_split(X_train, X_val, X_test, cat_cols)

    n_total = len(df)
    pct = lambda n: round(100 * n / n_total, 1)

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "num_cols": num_cols, "cat_cols": cat_cols,
        "stratified": use_stratify,
        "n_total": n_total,
        "pct_train": pct(len(X_train)),
        "pct_val": pct(len(X_val)),
        "pct_test": pct(len(X_test)),
        "pct_dev": pct(len(X_train) + len(X_val)),
    }


def encode_features(df: pd.DataFrame, target_col: str, positive_class) -> tuple:
    """
    Compatibilidad: prepara sin particionar (solo para vista previa).
    La app debe usar prepare_dataset para el pipeline real.
    """
    df = clean_dataframe(df, target_col)
    num_cols, cat_cols = _column_types(df, target_col)
    y = _build_y(df, target_col, positive_class)
    X_raw = df.drop(columns=[target_col])
    num_stats, cat_stats = _fit_imputation_stats(X_raw, num_cols, cat_cols)
    X_raw = _impute_frame(X_raw, num_cols, cat_cols, num_stats, cat_stats)
    if cat_cols:
        X_encoded = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True)
    else:
        X_encoded = X_raw
    return X_encoded.fillna(0), y, num_cols, cat_cols


def split_data(X: pd.DataFrame, y: pd.Series,
               test_size: float = DEFAULT_TEST_SIZE,
               val_size: float = DEFAULT_VAL_FRAC_OF_TRAIN) -> dict:
    """Partición sobre matrices ya codificadas (legacy). Preferir prepare_dataset."""
    min_class = y.value_counts().min()
    use_stratify = min_class >= 6

    def do_split(stratify: bool):
        strat = y if stratify else None
        X_dev, X_test, y_dev, y_test = train_test_split(
            X, y, test_size=test_size, stratify=strat, random_state=RANDOM_STATE,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_dev, y_dev, test_size=val_size, stratify=y_dev if stratify else None,
            random_state=RANDOM_STATE,
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    try:
        parts = do_split(use_stratify)
    except ValueError:
        use_stratify = False
        parts = do_split(False)

    X_train, X_val, X_test, y_train, y_val, y_test = parts
    if y_test.nunique() < 2:
        use_stratify = False
        X_train, X_val, X_test, y_train, y_val, y_test = do_split(False)

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "stratified": use_stratify,
    }


def scale_data(splits: dict) -> tuple:
    """StandardScaler — fit() solo sobre X_train."""
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(splits["X_train"])
    X_val_sc   = scaler.transform(splits["X_val"])
    X_test_sc  = scaler.transform(splits["X_test"])

    scaled = {
        "X_train": X_train_sc, "X_val": X_val_sc, "X_test": X_test_sc,
        "y_train": splits["y_train"],
        "y_val":   splits["y_val"],
        "y_test":  splits["y_test"],
    }
    return scaled, scaler
