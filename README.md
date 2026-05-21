# 🎯 Evaluación Final — Minería de Datos

App Streamlit que **fusiona** los CSV de `data/` (p. ej. student-mat + student-por) y produce **un solo informe** con todo el pipeline del examen.

---

## 🗂️ Estructura del proyecto

```
mineria_app/
│
├── app.py                  ← Entry point (UI y navegación)
├── requirements.txt
├── README.md
│
├── modules/
│   ├── __init__.py         ← Exportaciones del paquete
│   ├── detector.py         ← Detección automática de CSV, separador y target
│   ├── preprocessing.py    ← Partición, encoding y scaling sin data leakage
│   ├── baseline.py         ← Modelo ZeroR (piso mínimo de referencia)
│   ├── clustering.py       ← K-Means, clustering jerárquico, Silhouette
│   ├── classification.py   ← Árbol de Decisión y Random Forest
│   └── evaluation.py       ← Curva ROC, matriz de confusión y métricas
│
└── assets/
    └── style.css           ← Estilos CSS separados del código Python
```

---

## 🚀 Instalación y ejecución

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

Abre http://localhost:8501 en tu navegador.

---

## 📋 Flujo de uso

1. Copia los **2 CSV del ingeniero** en la carpeta `data/` (ver `data/README.md`)
2. Opcional: súbelos también desde el sidebar (acepta varios archivos)
3. Haz clic en **EJECUTAR ANÁLISIS**
4. Revisa el **informe único** (todos los registros fusionados)

**Partición por defecto:** 80% desarrollo (entrenamiento + validación) y 20% prueba final, sin fuga de información (imputación, encoding y escalado solo en train).

---

## 🤖 Detección automática

| Elemento | Método |
|---|---|
| Separador CSV | Cuenta `,` `;` `\t` `\|` en las primeras 5 líneas |
| Columna target | Keywords → última binaria categórica → última binaria numérica |
| Clase positiva | Keywords (`yes`, `si`, `true`...) → clase minoritaria |
| k óptimo | Máximo Silhouette Score entre k=2..k_max |
| max_depth árbol | Máximo F1 en validación entre depth=2..19 |