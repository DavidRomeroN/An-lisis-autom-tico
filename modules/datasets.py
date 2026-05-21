"""
datasets.py
-----------
Carga y fusión de varios CSV en un único dataset para un solo resultado.
"""

from pathlib import Path

import pandas as pd

from .detector import load_dataset, infer_target_column, build_binary_target

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ORIGIN_COLUMN = "_origen"


def list_bundled_csv() -> list[Path]:
    if not DATA_DIR.is_dir():
        return []
    return sorted(DATA_DIR.glob("*.csv")) + sorted(DATA_DIR.glob("*.tsv"))


def load_from_path(filepath: str | Path) -> dict:
    path = Path(filepath)
    df, sep, target, pos, target_desc = load_dataset(str(path))
    return {
        "name": path.name,
        "path": str(path),
        "df": df,
        "sep": sep,
        "target": target,
        "positive": pos,
        "target_desc": target_desc,
    }


def _origin_label(filename: str) -> str:
    stem = Path(filename).stem.lower()
    if "mat" in stem:
        return "mat"
    if "por" in stem:
        return "por"
    return stem


def merge_datasets(items: list[dict]) -> dict:
    """
    Concatena todos los CSV en un solo DataFrame.
    Añade columna _origen (mat/por) para trazabilidad.
    """
    if not items:
        raise ValueError("No hay datasets para fusionar.")

    if len(items) == 1:
        single = items[0].copy()
        single["df"] = single["df"].copy()
        single["df"][ORIGIN_COLUMN] = _origin_label(single["name"])
        single["sources"] = [single["name"]]
        single["display_name"] = single["name"]
        single["rows_per_source"] = {single["name"]: len(single["df"])}
        return single

    frames = []
    sources = []
    for item in items:
        df = item["df"].copy()
        df[ORIGIN_COLUMN] = _origin_label(item["name"])
        frames.append(df)
        sources.append(item["name"])

    merged = pd.concat(frames, ignore_index=True, sort=False)
    target = infer_target_column(merged)
    _, pos, desc = build_binary_target(merged, target)

    rows_per_source = (
        merged.groupby(ORIGIN_COLUMN, observed=True)
        .size()
        .to_dict()
    )

    names_short = " + ".join(_origin_label(s) for s in sources)

    return {
        "name": f"unificado ({names_short})",
        "display_name": f"Dataset unificado — {' + '.join(sources)}",
        "df": merged,
        "sep": items[0]["sep"],
        "target": target,
        "positive": pos,
        "target_desc": desc,
        "sources": sources,
        "rows_per_source": rows_per_source,
    }


def load_all_sources(uploaded_paths: list[tuple[str, str]] | None = None) -> list[dict]:
    """Carga subidos + bundled sin duplicar nombres."""
    uploaded_paths = uploaded_paths or []
    loaded = []
    names = set()

    for name, path in uploaded_paths:
        item = load_from_path(path)
        item["name"] = name
        loaded.append(item)
        names.add(name.lower())

    for p in list_bundled_csv():
        if p.name.lower() not in names:
            loaded.append(load_from_path(p))

    return loaded


def load_unified(uploaded_paths: list[tuple[str, str]] | None = None) -> dict:
    """Carga todos los CSV y devuelve un único dataset fusionado."""
    items = load_all_sources(uploaded_paths)
    if not items:
        raise FileNotFoundError("No se encontraron archivos CSV en data/ ni subidos.")
    return merge_datasets(items)
