"""
ui_analysis.py
--------------
Renderizado Streamlit del análisis completo.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram

from .clustering import cluster_profile_labels
from .classification import get_feature_importance
from .evaluation import (
    compute_confusion_metrics,
    compute_roc_curves,
    get_classification_report,
)

SEP_NAMES = {
    ",": "coma",
    ";": "punto y coma",
    "\t": "tabulación",
    "|": "pipe",
}

PLOT_COLORS = {
    "primary": "#1e3a5f",
    "secondary": "#2d5a87",
    "accent": "#c45c26",
    "neutral": "#9aa5b1",
    "grid": "#e8ecf0",
}


def configure_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["IBM Plex Sans", "Segoe UI", "DejaVu Sans"],
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.edgecolor": "#d0d5dc",
        "axes.labelcolor": "#1a1a2e",
        "text.color": "#1a1a2e",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "grid.color": PLOT_COLORS["grid"],
        "grid.alpha": 0.8,
    })


def _style_axes(ax, title: str) -> None:
    ax.set_title(title, fontweight=600, pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y", linestyle="-", alpha=0.4)
    ax.set_axisbelow(True)


def render_dataset_analysis(
    st, out: dict, ds: dict, palette: list,
    section, status, warn, info_card, plot_close,
) -> None:
    df = ds["df"]
    TARGET = ds["target"]
    POS = ds["positive"]
    filename = ds["name"]
    prep = out["prep"]
    splits = out["splits"]
    scaled = out["scaled"]
    clean = prep.get("clean_report", {})

    # 1. Dataset
    section("1. Dataset y limpieza")
    sources = ds.get("sources", [filename])
    if len(sources) > 1:
        info_card(
            "<strong>Datos unificados:</strong> "
            + ", ".join(f"<code>{s}</code>" for s in sources)
            + f" — <strong>{df.shape[0]:,}</strong> registros en un solo análisis."
        )
        if ds.get("rows_per_source"):
            st.markdown("**Registros por origen**")
            st.dataframe(
                pd.DataFrame([
                    {"Origen": k, "Registros": v}
                    for k, v in ds["rows_per_source"].items()
                ]),
                use_container_width=True,
                hide_index=True,
            )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filas totales", f"{df.shape[0]:,}")
    c2.metric("Columnas", df.shape[1])
    c3.metric("Archivos fusionados", len(sources))
    c4.metric("Nulos tras limpieza", f"{clean.get('nulos_despues_marcado', df.isnull().sum().sum()):,}")

    target_desc = prep.get("target_description") or ds.get("target_desc", "")
    status(
        f"<strong>Variable objetivo:</strong> {TARGET} · "
        f"<strong>Clase positiva:</strong> {POS}<br>{target_desc}"
    )
    if clean.get("notas_previas_excluidas"):
        info_card(
            "Se excluyeron G1 y G2 del conjunto de predictores para evitar data leakage "
            "(las notas de periodos anteriores predicen G3)."
        )

    if clean:
        st.markdown("**Reporte de limpieza e imputación**")
        cr1, cr2, cr3 = st.columns(3)
        cr1.metric("Duplicados eliminados", clean.get("duplicados_eliminados", 0))
        cr2.metric(
            "Marcadores a NaN",
            f"{clean.get('nulos_antes', 0):,} / {clean.get('nulos_despues_marcado', 0):,}",
        )
        dropped = clean.get("columnas_eliminadas", [])
        cr3.metric("Columnas eliminadas", len(dropped))
        if dropped:
            st.caption("Columnas: " + ", ".join(dropped[:12]) + ("…" if len(dropped) > 12 else ""))

    col_data, col_chart = st.columns([2, 1])
    with col_data:
        st.dataframe(df.head(8), use_container_width=True)
    with col_chart:
        target_dist = df[TARGET].value_counts()
        fig, ax = plt.subplots(figsize=(4, 3))
        bars = ax.bar(
            target_dist.index.astype(str),
            target_dist.values,
            color=[PLOT_COLORS["primary"], PLOT_COLORS["secondary"]][: len(target_dist)],
            edgecolor="white",
            linewidth=1,
        )
        for b in bars:
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + target_dist.max() * 0.02,
                f"{int(b.get_height()):,}",
                ha="center",
                fontsize=9,
                fontweight="600",
            )
        _style_axes(ax, f"Distribución de {TARGET}")
        plt.tight_layout()
        plot_close(fig)

    # 2. Preprocesamiento
    section("2. Preprocesamiento y partición")
    n_features = splits["X_train"].shape[1]
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Train", f"{len(splits['X_train']):,}", f"{prep['pct_train']}%")
    p2.metric("Validación", f"{len(splits['X_val']):,}", f"{prep['pct_val']}%")
    p3.metric("Test", f"{len(splits['X_test']):,}", f"{prep['pct_test']}%")
    p4.metric("Desarrollo", f"{prep['pct_dev']}%", f"Test {prep['pct_test']}%")

    status(
        f"Imputación (mediana/moda) y codificación One-Hot ajustadas solo en train. "
        f"{len(prep['cat_cols'])} variables categóricas, {n_features} features finales. "
        "Sin data leakage en escalado ni imputación."
    )

    # 3. Baseline
    section("3. Modelo baseline (ZeroR)")
    br = out["baseline_result"]
    b1, b2, b3 = st.columns(3)
    b1.metric("Accuracy", f"{br['accuracy']:.3f}")
    b2.metric("F1-Score", f"{br['f1']:.3f}")
    b3.metric("AUC-ROC", f"{br['auc']:.3f}")
    info_card(
        "El baseline predice siempre la clase mayoritaria. Sirve como referencia mínima: "
        "AUC cercano a 0.50 indica capacidad discriminativa nula."
    )

    # 4. Clustering
    section("4. Segmentación")
    sweep = out["sweep"]
    K_OPT = out["K_OPT"]
    hier_result = out["hier_result"]

    cl1, cl2 = st.columns(2)
    with cl1:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(sweep["k_range"], sweep["inertias"], "o-", color=PLOT_COLORS["primary"], lw=2, ms=6)
        ax.axvline(K_OPT, color=PLOT_COLORS["accent"], ls="--", lw=1.5, label=f"k óptimo = {K_OPT}")
        _style_axes(ax, "Método del codo — K-Means")
        ax.set_xlabel("Número de clústeres (k)")
        ax.set_ylabel("Inercia")
        ax.legend(fontsize=8, frameon=False)
        plt.tight_layout()
        plot_close(fig)
    with cl2:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(sweep["k_range"], sweep["sil_scores"], "s-", color=PLOT_COLORS["secondary"], lw=2, ms=6)
        ax.axvline(K_OPT, color=PLOT_COLORS["accent"], ls="--", lw=1.5, label=f"k óptimo = {K_OPT}")
        _style_axes(ax, "Índice Silhouette por k")
        ax.set_xlabel("k")
        ax.set_ylabel("Coeficiente Silhouette")
        ax.legend(fontsize=8, frameon=False)
        plt.tight_layout()
        plot_close(fig)

    st.markdown("**Perfiles por clúster**")
    st.dataframe(out["perfil"], use_container_width=True)
    for hint in cluster_profile_labels(out["cl_cols"]):
        info_card(hint)

    Z = hier_result["linkage_matrix"]
    fig, ax = plt.subplots(figsize=(12, 4))
    dendrogram(
        Z, ax=ax, truncate_mode="lastp", p=20, leaf_rotation=90,
        leaf_font_size=8, color_threshold=Z[-K_OPT + 1, 2],
        above_threshold_color=PLOT_COLORS["neutral"],
    )
    ax.axhline(Z[-K_OPT + 1, 2], color=PLOT_COLORS["accent"], ls="--", lw=1.5, label=f"Corte k = {K_OPT}")
    _style_axes(ax, f"Clustering jerárquico (Ward) — muestra n = {hier_result['n_sample']}")
    ax.set_xlabel("Índice de observación")
    ax.set_ylabel("Distancia")
    ax.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    plot_close(fig)

    # 5. Clasificación
    section("5. Clasificación")
    tuning = out["tuning"]
    dt_result = out["dt_result"]
    rf_result = out["rf_result"]

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(tuning["depth_range"], tuning["val_f1_list"], "o-", color=PLOT_COLORS["primary"], lw=2, ms=5)
    ax.axvline(out["BEST_DEPTH"], color=PLOT_COLORS["accent"], ls="--", label=f"depth = {out['BEST_DEPTH']}")
    _style_axes(ax, "Afinación del árbol — F1 en validación")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("F1-Score")
    ax.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    plot_close(fig)

    m1, m2 = st.columns(2)
    with m1:
        st.markdown(f"**Árbol de decisión** (max_depth = {out['BEST_DEPTH']})")
        a1, a2, a3 = st.columns(3)
        a1.metric("Accuracy", f"{dt_result['accuracy']:.3f}")
        a2.metric("F1-Score", f"{dt_result['f1']:.3f}")
        a3.metric("AUC-ROC", f"{dt_result['auc']:.3f}")
    with m2:
        st.markdown(f"**Random Forest** ({out['N_EST']} estimadores)")
        r1, r2, r3 = st.columns(3)
        r1.metric("Accuracy", f"{rf_result['accuracy']:.3f}")
        r2.metric("F1-Score", f"{rf_result['f1']:.3f}")
        r3.metric("AUC-ROC", f"{rf_result['auc']:.3f}")

    fi = get_feature_importance(out["rf_model"], splits["X_train"].columns.tolist())
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(fi.index, fi.values, color=PLOT_COLORS["secondary"])
    _style_axes(ax, "Variables más importantes — Random Forest")
    ax.set_xlabel("Importancia relativa")
    ax.grid(True, axis="x", linestyle="-", alpha=0.35)
    plt.tight_layout()
    plot_close(fig)

    # 6. Evaluación comparativa
    section("6. Evaluación comparativa de modelos")
    y_test = scaled["y_test"]
    df_res = out["df_res"]
    mejor = out["mejor"]
    models_probs = [
        {"name": "Baseline (ZeroR)", "y_prob": br["y_prob"], "color": PLOT_COLORS["neutral"], "linestyle": "--"},
        {"name": "Árbol de decisión", "y_prob": dt_result["y_prob"], "color": PLOT_COLORS["accent"], "linestyle": "-"},
        {"name": "Random Forest", "y_prob": rf_result["y_prob"], "color": PLOT_COLORS["primary"], "linestyle": "-"},
    ]
    curves = compute_roc_curves(models_probs, y_test)
    ev1, ev2 = st.columns([1.3, 1])
    with ev1:
        fig, ax = plt.subplots(figsize=(6, 5))
        for c in curves:
            ax.plot(
                c["fpr"], c["tpr"], color=c["color"], lw=2.2, ls=c["linestyle"],
                label=f"{c['name']} (AUC = {c['auc']:.3f})",
            )
        ax.plot([0, 1], [0, 1], color=PLOT_COLORS["neutral"], ls=":", lw=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        _style_axes(ax, "Curva ROC — conjunto de prueba")
        ax.set_xlabel("Tasa de falsos positivos")
        ax.set_ylabel("Tasa de verdaderos positivos")
        ax.legend(fontsize=8, loc="lower right", frameon=True, fancybox=True)
        plt.tight_layout()
        plot_close(fig)
    with ev2:
        st.markdown("**Comparación de métricas**")
        st.dataframe(
            df_res.style.highlight_max(axis=0, color="#e8f4ea"),
            use_container_width=True,
        )
        status(
            f"<strong>Mejor modelo:</strong> {mejor} — "
            f"AUC-ROC {df_res.loc[mejor, 'AUC-ROC']:.4f}, "
            f"F1-Score {df_res.loc[mejor, 'F1-Score']:.4f}"
        )

    # 7. Matriz de confusión
    section("7. Matriz de confusión — Random Forest")
    cm_data = compute_confusion_metrics(y_test, rf_result["y_pred"])
    tp, tn, fp, fn = cm_data["tp"], cm_data["tn"], cm_data["fp"], cm_data["fn"]

    mc1, mc2 = st.columns([1, 1.2])
    with mc1:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm_data["cm"], annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred: No", "Pred: Sí"],
            yticklabels=["Real: No", "Real: Sí"],
            ax=ax, linewidths=1, linecolor="white",
            cbar_kws={"shrink": 0.8},
            annot_kws={"size": 14, "weight": "600"},
        )
        ax.set_title("Matriz de confusión", fontweight=600, fontsize=11)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Clase real")
        plt.tight_layout()
        plot_close(fig)
    with mc2:
        mm = st.columns(2)
        mm[0].metric("Verdaderos positivos (TP)", f"{tp:,}")
        mm[1].metric("Verdaderos negativos (TN)", f"{tn:,}")
        m2 = st.columns(2)
        m2[0].metric("Falsos positivos (FP)", f"{fp:,}")
        m2[1].metric("Falsos negativos (FN)", f"{fn:,}")
        m3 = st.columns(3)
        m3[0].metric("Precisión", f"{cm_data['precision']:.3f}")
        m3[1].metric("Recall", f"{cm_data['recall']:.3f}")
        m3[2].metric("F1-Score", f"{cm_data['f1']:.3f}")
        m4 = st.columns(2)
        m4[0].metric("Especificidad", f"{cm_data['especif']:.3f}")
        m4[1].metric("Accuracy", f"{cm_data['accuracy']:.3f}")

    with st.expander("Reporte de clasificación detallado"):
        st.code(get_classification_report(y_test, rf_result["y_pred"]))

    # Resumen ejecutivo
    section("Resumen ejecutivo")
    src_line = ", ".join(ds.get("sources", [filename]))
    st.markdown(
        f"""
        <table class="summary-table">
            <tr><td>Fuentes de datos</td><td>{src_line}</td></tr>
            <tr><td>Registros analizados</td><td>{df.shape[0]:,}</td></tr>
            <tr><td>Variable objetivo</td><td>{TARGET} — {target_desc}</td></tr>
            <tr><td>Mejor modelo</td><td><strong>{mejor}</strong></td></tr>
            <tr><td>AUC-ROC (test)</td><td>{df_res.loc[mejor, 'AUC-ROC']:.4f}</td></tr>
            <tr><td>F1-Score (test)</td><td>{df_res.loc[mejor, 'F1-Score']:.4f}</td></tr>
            <tr><td>Clústeres (K-Means)</td><td>k = {K_OPT}</td></tr>
        </table>
        """,
        unsafe_allow_html=True,
    )
    st.success(f"Análisis completado sobre {df.shape[0]:,} registros unificados.")
