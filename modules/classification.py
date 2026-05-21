"""
classification.py
-----------------
Modelos de clasificación con manejo robusto de edge cases.
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from .evaluation import safe_auc

RANDOM_STATE = 42


def _safe_proba(model, X) -> np.ndarray:
    """predict_proba seguro — maneja modelos con una sola clase en train."""
    proba = model.predict_proba(X)
    if proba.shape[1] == 1:
        return np.zeros(len(X), dtype=float)
    return proba[:, 1]


def tune_decision_tree(X_train_sc, y_train,
                        X_val_sc, y_val,
                        depth_range: range = range(2, 20)) -> dict:
    val_f1_list = []
    for d in depth_range:
        dt = DecisionTreeClassifier(
            max_depth=d, class_weight="balanced", random_state=RANDOM_STATE
        )
        dt.fit(X_train_sc, y_train)
        val_f1_list.append(f1_score(y_val, dt.predict(X_val_sc), zero_division=0))

    best_depth = depth_range.start + int(np.argmax(val_f1_list))

    return {
        "best_depth":  best_depth,
        "depth_range": list(depth_range),
        "val_f1_list": val_f1_list,
    }


def train_decision_tree(X_train_sc, y_train, best_depth: int) -> DecisionTreeClassifier:
    dt = DecisionTreeClassifier(
        max_depth=best_depth,
        min_samples_leaf=max(1, int(len(y_train) * 0.01)),
        class_weight="balanced",
        random_state=RANDOM_STATE
    )
    dt.fit(X_train_sc, y_train)
    return dt


def train_random_forest(X_train_sc, y_train, n_train: int) -> RandomForestClassifier:
    n_estimators = 100 if n_train < 5000 else 200
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features="sqrt",
        min_samples_leaf=max(1, int(n_train * 0.005)),
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    rf.fit(X_train_sc, y_train)
    return rf


def evaluate_model(model, X_test_sc, y_test) -> dict:
    y_pred = model.predict(X_test_sc)
    y_prob = _safe_proba(model, X_test_sc)

    return {
        "y_pred":   y_pred,
        "y_prob":   y_prob,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1":       float(f1_score(y_test, y_pred, zero_division=0)),
        "auc":      safe_auc(y_test, y_prob),
    }


def get_feature_importance(model: RandomForestClassifier,
                            feature_names: list,
                            top_n: int = 15) -> pd.Series:
    fi = pd.Series(model.feature_importances_, index=feature_names)
    return fi.sort_values(ascending=True).tail(top_n)