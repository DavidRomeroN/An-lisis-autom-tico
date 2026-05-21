"""
evaluation.py
-------------
Evaluación comparativa de modelos con manejo robusto de edge cases:
  - Datasets con una sola clase en test
  - AUC indefinido (nan)
  - Matriz de confusión con clases faltantes
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
)


def safe_auc(y_test, y_prob) -> float:
    """AUC-ROC seguro — retorna 0.5 si solo hay una clase en y_test."""
    if len(np.unique(y_test)) < 2:
        return 0.5
    try:
        return float(roc_auc_score(y_test, y_prob))
    except Exception:
        return 0.5


def build_results_table(results: dict) -> pd.DataFrame:
    df = pd.DataFrame(results).T
    df = df.sort_values("AUC-ROC", ascending=False)
    return df.round(4)


def get_best_model_name(results_df: pd.DataFrame) -> str:
    return results_df["AUC-ROC"].idxmax()


def compute_roc_curves(models_probs: list, y_test) -> list:
    """
    Calcula curvas ROC. Si solo hay una clase en y_test,
    retorna la diagonal (modelo aleatorio) para todos.
    """
    curves = []
    only_one_class = len(np.unique(y_test)) < 2

    for m in models_probs:
        if only_one_class:
            fpr = np.array([0.0, 1.0])
            tpr = np.array([0.0, 1.0])
            auc = 0.5
        else:
            try:
                fpr, tpr, _ = roc_curve(y_test, m["y_prob"])
                auc = roc_auc_score(y_test, m["y_prob"])
            except Exception:
                fpr = np.array([0.0, 1.0])
                tpr = np.array([0.0, 1.0])
                auc = 0.5
        curves.append({**m, "fpr": fpr, "tpr": tpr, "auc": auc})
    return curves


def compute_confusion_metrics(y_test, y_pred) -> dict:
    """
    Calcula TP, TN, FP, FN y métricas derivadas.
    Robusto ante test sets con una sola clase.
    """
    # Asegurar que la matriz tenga siempre forma 2x2
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy  = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    especif   = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "cm":        cm,
        "tp":        int(tp),
        "tn":        int(tn),
        "fp":        int(fp),
        "fn":        int(fn),
        "accuracy":  accuracy,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "especif":   especif,
    }


def get_classification_report(y_test, y_pred) -> str:
    try:
        return classification_report(
            y_test, y_pred,
            target_names=["No respondió", "Respondió"],
            zero_division=0
        )
    except Exception as e:
        return f"No se pudo generar el reporte: {e}"