"""
pipeline.py
-----------
Ejecuta el flujo completo de ML para un dataset (sin UI).
"""

from .preprocessing import prepare_dataset, scale_data
from .baseline import train_baseline, evaluate_baseline
from .clustering import (
    build_cluster_matrix, run_kmeans_sweep, fit_kmeans_final,
    fit_hierarchical, build_cluster_profiles,
)
from .classification import (
    tune_decision_tree, train_decision_tree, train_random_forest, evaluate_model,
)
from .evaluation import build_results_table, get_best_model_name


def run_full_pipeline(df, target: str, positive, test_size: float, val_frac: float,
                      k_max: int) -> dict:
    """Entrena y evalúa todos los modelos; retorna artefactos para la UI."""
    prep = prepare_dataset(df, target, positive, test_size=test_size,
                           val_frac_of_train=val_frac)
    splits = {
        "X_train": prep["X_train"], "X_val": prep["X_val"], "X_test": prep["X_test"],
        "y_train": prep["y_train"], "y_val": prep["y_val"], "y_test": prep["y_test"],
    }
    scaled, _ = scale_data(splits)

    baseline_model = train_baseline(scaled["X_train"], scaled["y_train"])
    baseline_result = evaluate_baseline(
        baseline_model, scaled["X_test"], scaled["y_test"]
    )

    X_cl, cl_cols = build_cluster_matrix(
        scaled["X_train"], splits["X_train"], prep["num_cols"]
    )
    sweep = run_kmeans_sweep(X_cl, k_max=k_max)
    K_OPT = sweep["k_opt"]
    km_result = fit_kmeans_final(X_cl, K_OPT)
    hier_result = fit_hierarchical(X_cl, K_OPT)

    present_num = [c for c in cl_cols if c in splits["X_train"].columns]
    raw_profile = (
        splits["X_train"][present_num].reset_index(drop=True)
        if present_num else None
    )
    perfil = build_cluster_profiles(
        X_cl, km_result["labels"], cl_cols, splits["y_train"], raw_means=raw_profile
    )

    tuning = tune_decision_tree(
        scaled["X_train"], scaled["y_train"],
        scaled["X_val"], scaled["y_val"],
    )
    dt_model = train_decision_tree(
        scaled["X_train"], scaled["y_train"], tuning["best_depth"]
    )
    dt_result = evaluate_model(dt_model, scaled["X_test"], scaled["y_test"])

    rf_model = train_random_forest(
        scaled["X_train"], scaled["y_train"], n_train=len(splits["X_train"])
    )
    rf_result = evaluate_model(rf_model, scaled["X_test"], scaled["y_test"])

    results = {
        "Baseline (ZeroR)": {
            "Accuracy": baseline_result["accuracy"],
            "F1-Score": baseline_result["f1"],
            "AUC-ROC": baseline_result["auc"],
        },
        "Árbol de Decisión": {
            "Accuracy": dt_result["accuracy"],
            "F1-Score": dt_result["f1"],
            "AUC-ROC": dt_result["auc"],
        },
        "Random Forest": {
            "Accuracy": rf_result["accuracy"],
            "F1-Score": rf_result["f1"],
            "AUC-ROC": rf_result["auc"],
        },
    }

    df_res = build_results_table(results)
    mejor = get_best_model_name(df_res)

    return {
        "prep": prep,
        "splits": splits,
        "scaled": scaled,
        "baseline_result": baseline_result,
        "sweep": sweep,
        "K_OPT": K_OPT,
        "km_result": km_result,
        "hier_result": hier_result,
        "cl_cols": cl_cols,
        "X_cl": X_cl,
        "perfil": perfil,
        "tuning": tuning,
        "dt_result": dt_result,
        "rf_result": rf_result,
        "rf_model": rf_model,
        "results": results,
        "df_res": df_res,
        "mejor": mejor,
        "N_EST": rf_model.n_estimators,
        "BEST_DEPTH": tuning["best_depth"],
        "sil_km": km_result["sil_global"],
        "sil_hier": hier_result["sil_global"],
    }
