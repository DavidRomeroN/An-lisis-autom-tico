"""
clustering.py
-------------
Segmentación no supervisada:
  - K-Means con selección automática de k por índice Silhouette
  - Clustering jerárquico aglomerativo (Ward)
  - Cálculo de Silhouette Score global y por muestra
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.cluster.hierarchy import linkage

RANDOM_STATE = 42


def build_cluster_matrix(X_train_sc: np.ndarray,
                          X_train: pd.DataFrame,
                          num_cols: list) -> tuple[np.ndarray, list]:
    """
    Extrae la submatriz de variables numéricas para clustering.
    Si no hay numéricas, usa las primeras 5 columnas disponibles.
    """
    valid_cols = [c for c in X_train.columns if c in num_cols]
    if not valid_cols:
        valid_cols = X_train.columns[:min(5, len(X_train.columns))].tolist()

    X_cl = pd.DataFrame(X_train_sc, columns=X_train.columns)[valid_cols].values
    return X_cl, valid_cols


def run_kmeans_sweep(X_cl: np.ndarray, k_max: int = 10) -> dict:
    """
    Ejecuta K-Means para k=2..k_max y calcula inercia + Silhouette.
    Selecciona automáticamente el k con mayor Silhouette Score.

    Retorna dict con:
      - k_range    : lista de valores de k evaluados
      - inertias   : lista de inercias
      - sil_scores : lista de Silhouette Scores
      - k_opt      : k óptimo seleccionado
    """
    k_range    = list(range(2, k_max + 1))
    inertias   = []
    sil_scores = []

    for k in k_range:
        km   = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=RANDOM_STATE)
        labs = km.fit_predict(X_cl)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_cl, labs))

    k_opt = k_range[int(np.argmax(sil_scores))]

    return {
        "k_range":    k_range,
        "inertias":   inertias,
        "sil_scores": sil_scores,
        "k_opt":      k_opt,
    }


def fit_kmeans_final(X_cl: np.ndarray, k: int) -> dict:
    """
    Entrena K-Means final con el k óptimo.

    Retorna dict con:
      - labels     : etiquetas de clúster por observación
      - sil_global : Silhouette Score promedio
      - sil_samples: Silhouette por muestra (para el plot)
      - centers    : centroides
    """
    km     = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=RANDOM_STATE)
    labels = km.fit_predict(X_cl)

    return {
        "labels":      labels,
        "sil_global":  silhouette_score(X_cl, labels),
        "sil_samples": silhouette_samples(X_cl, labels),
        "centers":     km.cluster_centers_,
    }


def fit_hierarchical(X_cl: np.ndarray, k: int) -> dict:
    """
    Clustering jerárquico aglomerativo (Ward).

    Retorna dict con:
      - labels      : etiquetas de clúster
      - sil_global  : Silhouette Score
      - linkage_matrix: matriz Z para dendrograma
    """
    n_sample = min(500, X_cl.shape[0])
    idx      = np.random.RandomState(RANDOM_STATE).choice(X_cl.shape[0], n_sample, replace=False)
    Z        = linkage(X_cl[idx], method="ward")

    hier   = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = hier.fit_predict(X_cl)

    return {
        "labels":         labels,
        "sil_global":     silhouette_score(X_cl, labels),
        "linkage_matrix": Z,
        "n_sample":       n_sample,
    }


def build_cluster_profiles(X_cl: np.ndarray, labels: np.ndarray,
                            col_names: list, y_train: pd.Series,
                            raw_means=None) -> pd.DataFrame:
    """
    Construye tabla de perfiles de clústeres con estadísticas descriptivas.
    Si raw_means está disponible, muestra medias en escala original (más interpretable).
    """
    stats_df = raw_means if raw_means is not None else pd.DataFrame(X_cl, columns=col_names)
    stats_df = stats_df.copy()
    stats_df["Clúster"] = [f"C{i+1}" for i in labels]
    stats_df["target"] = y_train.values

    perfil = stats_df.groupby("Clúster").agg(
        N=("target", "count"),
        **{"% Positivos": ("target", lambda x: round(x.mean() * 100, 1))},
        **{c: (c, "mean") for c in col_names if c in stats_df.columns},
    ).round(3)

    return perfil


def cluster_profile_labels(col_names: list) -> list[str]:
    """Sugerencias de perfiles según nombres de columnas (interpretación de negocio)."""
    hints = []
    joined = " ".join(col_names).lower()
    if any(k in joined for k in ("age", "edad")):
        hints.append("Segmentos por rango etario")
    if any(k in joined for k in ("balance", "saldo", "income", "ingreso")):
        hints.append("Perfiles por nivel de ingresos o saldo")
    if any(k in joined for k in ("duration", "duracion", "campaign")):
        hints.append("Clientes según intensidad de contacto o campaña")
    if not hints:
        hints.append("Grupos con comportamiento numérico similar — revisar medias por clúster")
    return hints