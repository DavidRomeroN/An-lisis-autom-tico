"""
app.py
------
Entry point de la aplicación Streamlit.
Solo contiene la UI y la navegación entre secciones.
Toda la lógica está en modules/.

Ejecutar con:
    python -m streamlit run app.py
"""

import os
import warnings
import tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import streamlit as st

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

import modules as md

# ─── CONFIGURACIÓN DE PÁGINA ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Minería de Datos — Evaluación Final",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = ["#1A3A5C", "#2E86C1", "#E74C3C", "#27AE60", "#F39C12", "#8E44AD"]


# ─── CARGAR CSS ───────────────────────────────────────────────────────────────
def load_css(path: str) -> None:
    with open(path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css("assets/style.css")


# ─── HELPERS UI ───────────────────────────────────────────────────────────────
def section(icon: str, title: str) -> None:
    st.markdown(
        f'<div class="section-header">{icon} {title}</div>',
        unsafe_allow_html=True,
    )


def status(text: str) -> None:
    st.markdown(f'<div class="status-box">{text}</div>', unsafe_allow_html=True)


def warn(text: str) -> None:
    st.markdown(f'<div class="warn-box">{text}</div>', unsafe_allow_html=True)


def info_card(text: str) -> None:
    st.markdown(f'<div class="info-card">{text}</div>', unsafe_allow_html=True)


def plot_close(fig) -> None:
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 Evaluación Final")
    st.markdown("**Minería de Datos — 8vo Ciclo**")
    st.divider()

    st.markdown("### 📂 Dataset")
    uploaded = st.file_uploader(
        "Sube tu CSV aquí",
        type=["csv", "tsv"],
        help="El sistema detecta automáticamente el separador y la columna objetivo.",
    )

    st.divider()
    st.markdown("### ⚙️ Parámetros")
    test_size = st.slider(
        "Proporción Test (prueba final)",
        0.15, 0.25, 0.20, 0.05,
        help="Por defecto 20% reservado solo para evaluar modelos. El 80% restante es para entrenamiento.",
    )
    val_size = st.slider(
        "Validación (% del bloque de entrenamiento)",
        0.10, 0.25, 0.15, 0.05,
        help="Subconjunto del 80% de desarrollo para afinar el árbol (sin tocar el test).",
    )
    k_max = st.slider("k máximo (clustering)", 5, 15, 10)

    st.divider()
    run_btn = st.button(
        "🚀 EJECUTAR ANÁLISIS",
        type="primary",
        use_container_width=True,
        disabled=(uploaded is None),
    )

    if uploaded is None:
        st.info("⬆️ Sube un CSV para comenzar")


# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown(
    '<p class="main-title">🎯 Evaluación Final — Minería de Datos</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="subtitle">Partición · Baseline · Segmentación · '
    "Clasificación · Evaluación Comparativa · Matriz de Confusión</p>",
    unsafe_allow_html=True,
)

# ─── PANTALLA DE BIENVENIDA ────────────────────────────────────────────────────
if uploaded is None:
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        **📊 Carga y Preprocesamiento**
        - Detección automática de separador y target
        - Partición 80% desarrollo / 20% prueba (sin data leakage)
        - StandardScaler sin data leakage
        - Baseline ZeroR como referencia
        """)
    with c2:
        st.markdown("""
        **🔵 Segmentación**
        - K-Means con k óptimo automático
        - Método del codo + Silhouette Score
        - Silhouette Plot por clúster
        - Clustering jerárquico + dendrograma
        """)
    with c3:
        st.markdown("""
        **🌳 Clasificación y Evaluación**
        - Árbol de Decisión (tunning automático)
        - Random Forest (estimadores adaptativos)
        - Curva ROC multi-modelo
        - Matriz de confusión con interpretación
        """)
    st.stop()


# ─── TRIGGER: guardar CSV y ejecutar ─────────────────────────────────────────
if uploaded and run_btn:
    with st.spinner("Cargando y analizando dataset..."):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tmp.write(uploaded.getbuffer())
        tmp.close()

        df, sep, TARGET, POS = md.load_dataset(tmp.name)
        os.unlink(tmp.name)

    st.session_state.update({
        "df": df, "TARGET": TARGET, "POS": POS, "sep": sep,
        "test_size": test_size, "val_size": val_size, "k_max": k_max,
        "filename": uploaded.name, "run": True,
    })


# ─── ANÁLISIS PRINCIPAL ───────────────────────────────────────────────────────
if not st.session_state.get("run"):
    st.stop()

df       = st.session_state["df"]
TARGET   = st.session_state["TARGET"]
POS      = st.session_state["POS"]
sep      = st.session_state["sep"]
test_sz  = st.session_state["test_size"]
val_sz   = st.session_state["val_size"]
k_max    = st.session_state["k_max"]
filename = st.session_state["filename"]

SEP_NAMES = {",": "coma ','", ";": "punto y coma ';'", "\t": "tabulación", "|": "pipe '|'"}

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — DATASET
# ══════════════════════════════════════════════════════════════════════════════
section("📊", "1. Dataset Detectado")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Filas",      f"{df.shape[0]:,}")
c2.metric("Columnas",   df.shape[1])
c3.metric("Separador",  SEP_NAMES.get(sep, sep))
c4.metric("Nulos",      df.isnull().sum().sum())

status(
    f"🎯 Columna objetivo: <strong>{TARGET}</strong> &nbsp;|&nbsp; "
    f"Clase positiva: <strong>{POS}</strong>"
)

col_data, col_chart = st.columns([2, 1])
with col_data:
    st.dataframe(df.head(8), use_container_width=True)
with col_chart:
    target_dist = df[TARGET].value_counts()
    fig, ax = plt.subplots(figsize=(4, 3))
    bars = ax.bar(
        target_dist.index.astype(str), target_dist.values,
        color=PALETTE[:2], edgecolor="white", linewidth=1.5,
    )
    for b in bars:
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + target_dist.max() * 0.02,
            f"{int(b.get_height()):,}",
            ha="center", fontsize=10, fontweight="bold",
        )
    ax.set_title(f"Distribución de '{TARGET}'", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plot_close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — PREPROCESAMIENTO
# ══════════════════════════════════════════════════════════════════════════════
section("🔧", "2. Preprocesamiento y Partición")

prep = md.prepare_dataset(df, TARGET, POS, test_size=test_sz, val_frac_of_train=val_sz)
splits = {
    "X_train": prep["X_train"], "X_val": prep["X_val"], "X_test": prep["X_test"],
    "y_train": prep["y_train"], "y_val": prep["y_val"], "y_test": prep["y_test"],
}
num_cols, cat_cols = prep["num_cols"], prep["cat_cols"]
scaled, scaler = md.scale_data(splits)
n_features = splits["X_train"].shape[1]

p1, p2, p3, p4 = st.columns(4)
p1.metric("Train", f"{len(splits['X_train']):,}", f"{prep['pct_train']}% · Pos {splits['y_train'].mean()*100:.1f}%")
p2.metric("Validación", f"{len(splits['X_val']):,}", f"{prep['pct_val']}% · Pos {splits['y_val'].mean()*100:.1f}%")
p3.metric("Test", f"{len(splits['X_test']):,}", f"{prep['pct_test']}% · Pos {splits['y_test'].mean()*100:.1f}%")
p4.metric("Desarrollo (Train+Val)", f"{len(splits['X_train'])+len(splits['X_val']):,}", f"{prep['pct_dev']}%")

status(
    "✅ <strong>Sin data leakage:</strong> partición primero; imputación y One-Hot solo con train; "
    f"StandardScaler ajustado solo en train. "
    f"{len(cat_cols)} categóricas codificadas → {n_features} features."
)

with st.expander("📖 Partición, data leakage y baseline (explicación)"):
    st.markdown(f"""
**Partición aplicada**
1. Se reserva **{prep['pct_test']}%** como conjunto de **prueba** (nunca usado para entrenar ni afinar).
2. El **{prep['pct_dev']}%** restante es **desarrollo**: **{prep['pct_train']}%** entrenamiento + **{prep['pct_val']}%** validación para elegir `max_depth` del árbol.
3. La partición es **estratificada** cuando hay suficientes ejemplos por clase (mantiene la proporción de respuesta positiva).

**Data leakage (fuga de información)**  
Ocurre si información del conjunto de prueba “contamina” el entrenamiento. Aquí se evita porque:
- La imputación (mediana/moda) se calcula **solo en train** y se aplica a val/test.
- El One-Hot Encoding aprende categorías **solo en train**.
- El escalado (`StandardScaler`) se ajusta **solo en train**.

**Modelo baseline (ZeroR)**  
Predice siempre la clase mayoritaria. Sirve de **piso mínimo**: cualquier modelo útil debe superar su accuracy y, sobre todo, su **F1** y **AUC** cuando la campaña busca detectar la clase positiva (respuesta sí).
    """)

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — BASELINE
# ══════════════════════════════════════════════════════════════════════════════
section("📏", "3. Modelo Baseline (ZeroR)")

baseline_model  = md.train_baseline(scaled["X_train"], scaled["y_train"])
baseline_result = md.evaluate_baseline(baseline_model, scaled["X_test"], scaled["y_test"])

b1, b2, b3 = st.columns(3)
b1.metric("Accuracy (baseline)", f"{baseline_result['accuracy']:.3f}")
b2.metric("F1-Score (baseline)", f"{baseline_result['f1']:.3f}", "← siempre 0.0")
b3.metric("AUC-ROC  (baseline)", f"{baseline_result['auc']:.3f}", "← aleatorio")

warn(
    "⚠️ El baseline predice siempre la clase mayoritaria. "
    "AUC = 0.50 y F1 = 0.0 representan el <strong>piso mínimo</strong> "
    "que deben superar los modelos reales."
)

results = {
    "Baseline (ZeroR)": {
        "Accuracy": baseline_result["accuracy"],
        "F1-Score": baseline_result["f1"],
        "AUC-ROC":  baseline_result["auc"],
    }
}

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
section("🔵", "4. Segmentación (Clustering)")

X_cl, cl_cols = md.build_cluster_matrix(
    scaled["X_train"], splits["X_train"], num_cols
)

with st.spinner(f"Ejecutando K-Means para k=2..{k_max}..."):
    sweep = md.run_kmeans_sweep(X_cl, k_max=k_max)

K_OPT = sweep["k_opt"]
status(
    f"✅ k óptimo detectado automáticamente: <strong>{K_OPT}</strong> &nbsp;|&nbsp; "
    f"Silhouette = <strong>{max(sweep['sil_scores']):.4f}</strong>"
)

# Codo + Silhouette sweep
cl1, cl2 = st.columns(2)
with cl1:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(sweep["k_range"], sweep["inertias"], "o-", color=PALETTE[0], lw=2, ms=7)
    ax.axvline(K_OPT, color=PALETTE[2], ls="--", lw=1.5, label=f"k óptimo = {K_OPT}")
    ax.set_title("Método del Codo — K-Means", fontsize=11)
    ax.set_xlabel("k"); ax.set_ylabel("Inercia")
    ax.spines[["top", "right"]].set_visible(False); ax.legend(fontsize=9)
    plt.tight_layout(); plot_close(fig)

with cl2:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(sweep["k_range"], sweep["sil_scores"], "s-", color=PALETTE[1], lw=2, ms=7)
    ax.axvline(K_OPT, color=PALETTE[2], ls="--", lw=1.5, label=f"k óptimo = {K_OPT}")
    ax.set_title("Índice Silhouette por k", fontsize=11)
    ax.set_xlabel("k"); ax.set_ylabel("Silhouette Score")
    ax.spines[["top", "right"]].set_visible(False); ax.legend(fontsize=9)
    plt.tight_layout(); plot_close(fig)

# K-Means final
km_result = md.fit_kmeans_final(X_cl, K_OPT)

# Silhouette Plot
sil_vals = km_result["sil_samples"]
y_lower  = 10
fig, ax  = plt.subplots(figsize=(10, 4))
cmap_sil = plt.cm.get_cmap("tab10")(np.linspace(0, 1, K_OPT))
for i in range(K_OPT):
    ith     = np.sort(sil_vals[km_result["labels"] == i])
    y_upper = y_lower + len(ith)
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith,
                     facecolor=cmap_sil[i], alpha=0.8)
    ax.text(-0.05, y_lower + 0.5 * len(ith), f"C{i+1}", fontsize=9)
    y_lower = y_upper + 10
ax.axvline(km_result["sil_global"], color="red", ls="--", lw=1.5,
           label=f"Silhouette = {km_result['sil_global']:.3f}")
ax.set_title(f"Silhouette Plot — K-Means (k={K_OPT})", fontsize=11)
ax.set_xlabel("Coeficiente Silhouette")
ax.spines[["top", "right"]].set_visible(False); ax.legend(fontsize=9)
plt.tight_layout(); plot_close(fig)

# Perfiles (escala original cuando hay columnas numéricas)
present_num = [c for c in cl_cols if c in splits["X_train"].columns]
raw_profile = splits["X_train"][present_num].reset_index(drop=True) if present_num else None
perfil = md.build_cluster_profiles(
    X_cl, km_result["labels"], cl_cols, splits["y_train"], raw_means=raw_profile
)
st.markdown("**Perfiles de clústeres (medias por segmento):**")
st.dataframe(perfil, use_container_width=True)
for hint in md.cluster_profile_labels(cl_cols):
    info_card(f"💡 Posible perfil: {hint}")

with st.expander("📖 Segmentación: K-Means, jerárquico y Silhouette"):
    st.markdown("""
**K-Means:** agrupa observaciones minimizando la distancia al centroide. El **método del codo** (inercia vs. k) sugiere dónde deja de bajar mucho el error; el **índice Silhouette** (entre -1 y 1) mide qué tan bien separados están los clústeres — valores cercanos a 1 indican agrupación clara.

**Clustering jerárquico (Ward):** fusiona gradualmente los grupos más parecidos; el **dendrograma** muestra esa fusión y permite cortar en k grupos.

**Interpretación:** compare medias por clúster y **% positivos** (tasa de respuesta a campaña). Un clúster con alto % positivo y alto saldo, por ejemplo, podría ser *“clientes premium con alta propensión”*; uno con bajo % positivo, *“bajo interés — evitar contacto masivo”*.
    """)

# Clustering Jerárquico
hier_result = md.fit_hierarchical(X_cl, K_OPT)
Z           = hier_result["linkage_matrix"]
n_sample    = hier_result["n_sample"]

from scipy.cluster.hierarchy import dendrogram
fig, ax = plt.subplots(figsize=(12, 4))
dendrogram(Z, ax=ax, truncate_mode="lastp", p=20, leaf_rotation=90,
           leaf_font_size=8, color_threshold=Z[-K_OPT + 1, 2],
           above_threshold_color="#aaa")
ax.axhline(Z[-K_OPT + 1, 2], color=PALETTE[2], ls="--", lw=1.5,
           label=f"Corte k={K_OPT}")
ax.set_title(f"Dendrograma — Clustering Jerárquico Ward (muestra n={n_sample})",
             fontsize=11)
ax.set_xlabel("Observaciones (truncado)"); ax.set_ylabel("Distancia Ward")
ax.spines[["top", "right"]].set_visible(False); ax.legend(fontsize=9)
plt.tight_layout(); plot_close(fig)

sil_km   = km_result["sil_global"]
sil_hier = hier_result["sil_global"]
mejor_cl = "K-Means" if sil_km >= sil_hier else "Jerárquico"
status(
    f"K-Means Silhouette: <strong>{sil_km:.4f}</strong> &nbsp;|&nbsp; "
    f"Jerárquico Silhouette: <strong>{sil_hier:.4f}</strong> &nbsp;|&nbsp; "
    f"✅ Mejor agrupación: <strong>{mejor_cl}</strong>"
)

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5 — CLASIFICACIÓN
# ══════════════════════════════════════════════════════════════════════════════
section("🌳", "5. Clasificación — Respuesta a campaña")

with st.expander("📖 Árbol de decisión vs. Random Forest"):
    st.markdown("""
**Problema de negocio:** predecir si un cliente **responderá positivamente** a una campaña (clase 1 = sí).

**Árbol de decisión:** reglas interpretables (si edad > X y saldo < Y → no contactar). Se afinó `max_depth` maximizando **F1 en validación** (equilibrio precisión/recall en clase positiva).

**Random Forest:** combina muchos árboles con votación; suele generalizar mejor y reduce sobreajuste. Se evalúan ambos en el **mismo test (20%)** con **Accuracy**, **F1-Score** y **AUC-ROC**.
    """)

# ── Árbol de Decisión ─────────────────────────────────────────────────────────
with st.spinner("Buscando max_depth óptimo (Árbol de Decisión)..."):
    tuning = md.tune_decision_tree(
        scaled["X_train"], scaled["y_train"],
        scaled["X_val"],   scaled["y_val"],
    )

BEST_DEPTH = tuning["best_depth"]

fig, ax = plt.subplots(figsize=(9, 3))
ax.plot(tuning["depth_range"], tuning["val_f1_list"],
        "o-", color=PALETTE[0], lw=2, ms=6)
ax.axvline(BEST_DEPTH, color=PALETTE[2], ls="--", lw=1.5,
           label=f"Óptimo = {BEST_DEPTH}")
ax.set_title("Tunning — F1 en Validación vs. max_depth", fontsize=11)
ax.set_xlabel("max_depth"); ax.set_ylabel("F1-Score")
ax.spines[["top", "right"]].set_visible(False); ax.legend(fontsize=9)
plt.tight_layout(); plot_close(fig)

dt_model  = md.train_decision_tree(scaled["X_train"], scaled["y_train"], BEST_DEPTH)
dt_result = md.evaluate_model(dt_model, scaled["X_test"], scaled["y_test"])

# ── Random Forest ─────────────────────────────────────────────────────────────
with st.spinner("Entrenando Random Forest..."):
    rf_model  = md.train_random_forest(
        scaled["X_train"], scaled["y_train"], n_train=len(splits["X_train"])
    )
rf_result = md.evaluate_model(rf_model, scaled["X_test"], scaled["y_test"])
N_EST     = rf_model.n_estimators

results["Árbol de Decisión"] = {
    "Accuracy": dt_result["accuracy"],
    "F1-Score": dt_result["f1"],
    "AUC-ROC":  dt_result["auc"],
}
results["Random Forest"] = {
    "Accuracy": rf_result["accuracy"],
    "F1-Score": rf_result["f1"],
    "AUC-ROC":  rf_result["auc"],
}

# Métricas comparadas
m1, m2 = st.columns(2)
with m1:
    st.markdown(f"**🌳 Árbol de Decisión** (max_depth = {BEST_DEPTH})")
    a1, a2, a3 = st.columns(3)
    a1.metric("Accuracy", f"{dt_result['accuracy']:.3f}")
    a2.metric("F1-Score", f"{dt_result['f1']:.3f}")
    a3.metric("AUC-ROC",  f"{dt_result['auc']:.3f}")

with m2:
    st.markdown(f"**🌲 Random Forest** ({N_EST} árboles)")
    r1, r2, r3 = st.columns(3)
    r1.metric("Accuracy", f"{rf_result['accuracy']:.3f}",
              f"{(rf_result['accuracy']-dt_result['accuracy'])*100:+.1f}%")
    r2.metric("F1-Score", f"{rf_result['f1']:.3f}",
              f"{(rf_result['f1']-dt_result['f1'])*100:+.1f}%")
    r3.metric("AUC-ROC",  f"{rf_result['auc']:.3f}",
              f"{(rf_result['auc']-dt_result['auc'])*100:+.1f}%")

# Feature Importance
fi = md.get_feature_importance(rf_model, splits["X_train"].columns.tolist())
fig, ax = plt.subplots(figsize=(9, 4))
ax.barh(fi.index, fi.values, color=PALETTE[1])
ax.set_title(f"Top {len(fi)} Variables más Importantes — Random Forest", fontsize=11)
ax.set_xlabel("Importancia (Mean Decrease Impurity)")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plot_close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 6 — EVALUACIÓN COMPARATIVA
# ══════════════════════════════════════════════════════════════════════════════
section("📈", "6. Evaluación Comparativa de Modelos")

y_test   = scaled["y_test"]
df_res   = md.build_results_table(results)
mejor    = md.get_best_model_name(df_res)

models_probs = [
    {"name": "Baseline (ZeroR)",  "y_prob": baseline_result["y_prob"],
     "color": PALETTE[4], "linestyle": "--"},
    {"name": "Árbol de Decisión", "y_prob": dt_result["y_prob"],
     "color": PALETTE[2], "linestyle": "-"},
    {"name": "Random Forest",     "y_prob": rf_result["y_prob"],
     "color": PALETTE[0], "linestyle": "-"},
]
curves = md.compute_roc_curves(models_probs, y_test)

ev1, ev2 = st.columns([1.3, 1])
with ev1:
    fig, ax = plt.subplots(figsize=(6, 5))
    for c in curves:
        ax.plot(c["fpr"], c["tpr"], color=c["color"],
                lw=2.5, ls=c["linestyle"],
                label=f"{c['name']}\nAUC = {c['auc']:.3f}")
    rf_curve = next(c for c in curves if "Forest" in c["name"])
    ax.fill_between(rf_curve["fpr"], rf_curve["tpr"],
                    alpha=0.08, color=PALETTE[0])
    ax.plot([0, 1], [0, 1], color="#aaa", ls=":", lw=1)
    ax.set_xlabel("Tasa de Falsos Positivos (1 - Especificidad)")
    ax.set_ylabel("Tasa de Verdaderos Positivos (Sensibilidad)")
    ax.set_title("Curva ROC — Todos los Modelos", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    plt.tight_layout(); plot_close(fig)

with ev2:
    st.markdown("**Tabla comparativa:**")
    st.dataframe(
        df_res.style.highlight_max(axis=0, color="#d4edda"),
        use_container_width=True,
    )
    status(
        f"🏆 Mejor modelo: <strong>{mejor}</strong><br>"
        f"AUC-ROC: {df_res.loc[mejor,'AUC-ROC']:.4f} &nbsp;|&nbsp; "
        f"F1-Score: {df_res.loc[mejor,'F1-Score']:.4f}"
    )

with st.expander("📋 Cómo comunicar resultados a gerencia (no técnica)"):
    st.markdown(f"""
| Mensaje para gerencia | Qué significa |
|---|---|
| **Mejor modelo: {mejor}** | Es el que mejor separa quién responderá sí vs. no (AUC más alto). |
| **Curva ROC** | Mientras más arqueada hacia la esquina superior izquierda, mejor discrimina sin depender de un umbral fijo. |
| **F1-Score** | Balance entre aciertos en contactos útiles y no perder oportunidades de venta. |
| **Matriz de confusión** | Cuántos contactos fueron acertados (TP/TN) y cuánto dinero se pierde en FP (contactos inútiles) y FN (ventas no captadas). |
| **Recomendación** | Usar el modelo para **priorizar** la campaña; el baseline muestra el rendimiento sin inteligencia (solo la clase más frecuente). |

**Frase ejecutiva sugerida:** *“Con {mejor}, identificamos mejor a los clientes con probabilidad de respuesta; en prueba, el AUC fue {df_res.loc[mejor,'AUC-ROC']:.2f}, superando al azar (0.50) y al baseline ({df_res.loc['Baseline (ZeroR)','AUC-ROC']:.2f}).”*
    """)

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 7 — MATRIZ DE CONFUSIÓN
# ══════════════════════════════════════════════════════════════════════════════
section("🔲", "7. Matriz de Confusión — Random Forest")

cm_data = md.compute_confusion_metrics(y_test, rf_result["y_pred"])
tp, tn, fp, fn = cm_data["tp"], cm_data["tn"], cm_data["fp"], cm_data["fn"]

mconf1, mconf2 = st.columns(2)
with mconf1:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm_data["cm"], annot=True, fmt="d", cmap="Blues",
        xticklabels=["Pred: No", "Pred: Sí"],
        yticklabels=["Real: No", "Real: Sí"],
        ax=ax, linewidths=2, linecolor="white",
        annot_kws={"size": 18, "weight": "bold"},
    )
    ax.set_title("Matriz de Confusión — Random Forest", fontsize=11)
    ax.set_xlabel("Predicción"); ax.set_ylabel("Clase real")
    plt.tight_layout(); plot_close(fig)

with mconf2:
    quad_labels = [
        ["TN\nNo contactados\ncorrectamente", "FP\nContactados\nsin retorno"],
        ["FN\nOportunidades\nperdidas",        "TP\nConversiones\ncapturadas"],
    ]
    quad_vals   = [[tn, fp], [fn, tp]]
    quad_colors = [["#D5F5E3", "#FDEBD0"], ["#FDEBD0", "#D5F5E3"]]

    fig, ax = plt.subplots(figsize=(5, 4))
    for r in range(2):
        for c in range(2):
            ax.add_patch(plt.Rectangle(
                (c, 1 - r), 1, 1,
                facecolor=quad_colors[r][c], edgecolor="white", lw=3,
            ))
            ax.text(
                c + 0.5, 1 - r + 0.5,
                f"{quad_labels[r][c]}\n\n{quad_vals[r][c]:,}",
                ha="center", va="center", fontsize=9, fontweight="bold",
            )
    ax.set_xlim(0, 2); ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(["Pred: No", "Pred: Sí"])
    ax.set_yticks([0.5, 1.5]); ax.set_yticklabels(["Real: Sí", "Real: No"])
    ax.set_title("Interpretación", fontsize=11); ax.tick_params(length=0)
    plt.tight_layout(); plot_close(fig)

# Métricas de la matriz
mm = st.columns(6)
mm[0].metric("TP", f"{tp:,}", "✅")
mm[1].metric("TN", f"{tn:,}", "✅")
mm[2].metric("FP", f"{fp:,}", "❌")
mm[3].metric("FN", f"{fn:,}", "❌")
mm[4].metric("Precisión", f"{cm_data['precision']:.3f}")
mm[5].metric("Recall",    f"{cm_data['recall']:.3f}")

mm2 = st.columns(4)
mm2[0].metric("F1-Score",      f"{cm_data['f1']:.3f}")
mm2[1].metric("Especificidad", f"{cm_data['especif']:.3f}")
mm2[2].metric("Accuracy",      f"{cm_data['accuracy']:.3f}")
mm2[3].metric("AUC-ROC",       f"{rf_result['auc']:.3f}")

# Interpretación textual
st.markdown("**Interpretación:**")
ci1, ci2 = st.columns(2)
recall_pct = cm_data["recall"] * 100
with ci1:
    status(
        f"✅ <strong>TP = {tp:,}</strong><br>"
        f"De {tp+fn} clientes que realmente responderían, "
        f"el modelo identificó correctamente {tp} ({recall_pct:.1f}%)."
    )
    status(
        f"✅ <strong>TN = {tn:,}</strong><br>"
        f"{tn} clientes descartados correctamente — "
        "ahorro en costo de contacto innecesario."
    )
with ci2:
    warn(
        f"❌ <strong>FP = {fp:,}</strong><br>"
        f"Se contactaron innecesariamente {fp} clientes que no habrían respondido "
        "(costo operativo sin retorno)."
    )
    warn(
        f"❌ <strong>FN = {fn:,}</strong><br>"
        f"{fn} oportunidades de venta perdidas — clientes que habrían respondido "
        "pero no fueron contactados."
    )

with st.expander("📋 Ver reporte completo de clasificación"):
    report = md.get_classification_report(y_test, rf_result["y_pred"])
    st.code(report)

# ══════════════════════════════════════════════════════════════════════════════
# RESUMEN EJECUTIVO
# ══════════════════════════════════════════════════════════════════════════════
section("✅", "Resumen Ejecutivo")

st.markdown(f"""
| Componente | Resultado |
|---|---|
| Dataset | `{filename}` — {df.shape[0]:,} registros × {df.shape[1]} columnas |
| Target detectado | `{TARGET}` → clase positiva: `{POS}` |
| Partición | {prep['pct_dev']}% desarrollo ({prep['pct_train']}% train + {prep['pct_val']}% val) / {prep['pct_test']}% test · sin data leakage |
| k clústeres | **{K_OPT}** — Silhouette = {sil_km:.4f} (detectado automáticamente) |
| Árbol max_depth | **{BEST_DEPTH}** (tunning automático en validación) |
| Random Forest | **{N_EST}** estimadores (adaptativo al tamaño del dataset) |
| Mejor modelo | **{mejor}** — AUC = {df_res.loc[mejor,'AUC-ROC']:.4f} · F1 = {df_res.loc[mejor,'F1-Score']:.4f} |
""")

st.success(
    f"✅ Análisis completado. "
    f"Mejor modelo: **{mejor}** con AUC-ROC = {df_res.loc[mejor,'AUC-ROC']:.4f}"
)