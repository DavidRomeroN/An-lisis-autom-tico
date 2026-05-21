"""
app.py — Evaluación Final Minería de Datos
Fusiona todos los CSV en un único resultado.
"""

import os
import warnings
import tempfile

import matplotlib
import matplotlib.pyplot as plt
import streamlit as st

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

from modules.datasets import list_bundled_csv, load_unified, DATA_DIR
from modules.pipeline import run_full_pipeline
from modules.ui_analysis import render_dataset_analysis, configure_plot_style

st.set_page_config(
    page_title="Minería de Datos | Evaluación Final",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = ["#1e3a5f", "#2d5a87", "#c45c26", "#2d6a4f", "#6b5b95", "#0d6e6e"]


def load_css(path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css("assets/style.css")
configure_plot_style()


def section(title: str) -> None:
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def status(text: str) -> None:
    st.markdown(f'<div class="status-box">{text}</div>', unsafe_allow_html=True)


def warn(text: str) -> None:
    st.markdown(f'<div class="warn-box">{text}</div>', unsafe_allow_html=True)


def info_card(text: str) -> None:
    st.markdown(f'<div class="info-card">{text}</div>', unsafe_allow_html=True)


def plot_close(fig) -> None:
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# --- Sidebar ---
with st.sidebar:
    st.markdown('<p class="sidebar-brand">Evaluación Final</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-meta">Minería de Datos · 8vo Ciclo</p>', unsafe_allow_html=True)
    st.divider()

    bundled = list_bundled_csv()
    if bundled:
        st.markdown('<p class="sidebar-section-title">Archivos en data/</p>', unsafe_allow_html=True)
        for p in bundled:
            st.markdown(f'<p class="file-list-item">{p.name}</p>', unsafe_allow_html=True)
        st.caption("Se fusionarán en un solo conjunto de datos.")
    else:
        st.warning(f"Coloca los CSV en:\n`{DATA_DIR}`")

    st.markdown('<p class="sidebar-section-title">Carga manual</p>', unsafe_allow_html=True)
    uploaded_list = st.file_uploader(
        "Archivos CSV o TSV",
        type=["csv", "tsv"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown('<p class="sidebar-section-title">Parámetros</p>', unsafe_allow_html=True)
    test_size = st.slider("Proporción test (prueba)", 0.15, 0.25, 0.20, 0.05)
    val_size = st.slider("Validación (% del train)", 0.10, 0.25, 0.15, 0.05)
    k_max = st.slider("k máximo (clustering)", 5, 15, 10)

    st.divider()
    can_run = bool(bundled) or bool(uploaded_list)
    run_btn = st.button(
        "Ejecutar análisis",
        type="primary",
        use_container_width=True,
        disabled=not can_run,
    )

    if not can_run:
        st.info("Copia los CSV a `data/` o súbelos arriba.")


# --- Header ---
st.markdown(
    """
    <div class="page-header">
        <p class="main-title">Evaluación Final — Minería de Datos</p>
        <p class="subtitle">
            Dataset unificado · Preprocesamiento 80/20 · Segmentación ·
            Clasificación · Evaluación ROC · Matriz de confusión
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not (bundled or uploaded_list):
    st.markdown(
        """
        <div class="welcome-panel">
            <h3>Inicio del análisis</h3>
            <ol>
                <li>Coloque <code>student-mat.csv</code> y <code>student-por.csv</code> en la carpeta <code>data/</code>.</li>
                <li>Configure los parámetros en el panel lateral si lo desea.</li>
                <li>Pulse <strong>Ejecutar análisis</strong> para fusionar ambos archivos y generar un único informe.</li>
            </ol>
            <p><strong>Variable objetivo:</strong> G3 binarizado (aprobado si nota &ge; 10).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


# --- Ejecutar ---
if run_btn:
    temp_paths = []
    try:
        with st.spinner("Cargando y fusionando datasets..."):
            upload_pairs = []
            if uploaded_list:
                for up in uploaded_list:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
                    tmp.write(up.getbuffer())
                    tmp.close()
                    temp_paths.append(tmp.name)
                    upload_pairs.append((up.name, tmp.name))

            unified = load_unified(upload_pairs)

        st.session_state.update({
            "unified": unified,
            "test_size": test_size,
            "val_size": val_size,
            "k_max": k_max,
            "run": True,
        })
    except Exception as e:
        st.error(f"No se pudo cargar o fusionar: {e}")
        st.stop()
    finally:
        for p in temp_paths:
            try:
                os.unlink(p)
            except OSError:
                pass


if not st.session_state.get("run"):
    st.stop()

unified = st.session_state["unified"]
test_sz = st.session_state["test_size"]
val_sz = st.session_state["val_size"]
k_max = st.session_state["k_max"]

st.markdown(
    f"""
    <div class="run-banner">
        <strong>Análisis unificado</strong> — {unified['df'].shape[0]:,} registros
        ({', '.join(unified.get('sources', []))})
    </div>
    """,
    unsafe_allow_html=True,
)

with st.spinner("Ejecutando pipeline completo..."):
    try:
        out = run_full_pipeline(
            unified["df"],
            unified["target"],
            unified["positive"],
            test_size=test_sz,
            val_frac=val_sz,
            k_max=k_max,
        )
        ds = {
            "name": unified.get("display_name", unified["name"]),
            "df": unified["df"],
            "target": unified["target"],
            "positive": unified["positive"],
            "target_desc": unified.get("target_desc", ""),
            "sep": unified["sep"],
            "sources": unified.get("sources", []),
            "rows_per_source": unified.get("rows_per_source", {}),
        }
        render_dataset_analysis(
            st, out, ds, PALETTE,
            section, status, warn, info_card, plot_close,
        )
    except Exception as e:
        st.error(f"Error en el análisis: {e}")
        with st.expander("Detalle técnico"):
            st.exception(e)
