# Datasets del examen

Coloca aquí los **2 archivos CSV** que entregó el ingeniero.

Ejemplos habituales en minería de datos:

| Archivo típico | Separador | Columna objetivo | Positivo |
|----------------|-----------|------------------|----------|
| `student-mat.csv` / `student-por.csv` | `;` | **G3** (aprobado si nota ≥ 10) | `1` |
| `bank-full.csv` / `bank.csv` | `;` | `y` | `yes` |
| `credit_default.csv` / UCI default | `,` | `default` o última columna | `1` |

La app los detecta y ejecuta automáticamente:

- Limpieza (`unknown` → NaN, IDs, duplicados)
- Imputación (mediana numérica, moda categórica)
- Partición 80% desarrollo / 20% test
- Baseline, clustering, árbol, random forest, ROC y matriz de confusión

**No hace falta subirlos manualmente** si están en esta carpeta: al pulsar *EJECUTAR ANÁLISIS* se **fusionan** en un solo dataset y se genera **un único resultado** (partición, modelos, ROC, matriz).
