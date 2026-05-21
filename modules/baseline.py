"""
baseline.py
-----------
Modelo baseline ZeroR con métricas seguras.
"""

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score
from .evaluation import safe_auc

RANDOM_STATE = 42


def train_baseline(X_train, y_train) -> DummyClassifier:
    model = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def evaluate_baseline(model: DummyClassifier, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    proba  = model.predict_proba(X_test)

    # Si solo hay una clase en train, predict_proba retorna 1 columna
    if proba.shape[1] == 1:
        y_prob = np.zeros(len(y_pred), dtype=float)
    else:
        y_prob = proba[:, 1]

    return {
        "y_pred":   y_pred,
        "y_prob":   y_prob,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1":       float(f1_score(y_test, y_pred, zero_division=0)),
        "auc":      safe_auc(y_test, y_prob),
    }