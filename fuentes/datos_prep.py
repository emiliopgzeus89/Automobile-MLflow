"""
Funciones de carga y limpieza para el dataset Automobile de UCI.
"""
from pathlib import Path
import pandas as pd

UCI_COLUMNS = [
    "symboling", "normalized-losses", "make", "fuel-type", "aspiration",
    "num-of-doors", "body-style", "drive-wheels", "engine-location",
    "wheel-base", "length", "width", "height", "curb-weight", "engine-type",
    "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke",
    "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg",
    "price"
]

NUMERIC_COLUMNS = [
    "symboling", "normalized-losses", "wheel-base", "length", "width",
    "height", "curb-weight", "engine-size", "bore", "stroke",
    "compression-ratio", "horsepower", "peak-rpm", "city-mpg",
    "highway-mpg", "price"
]

def load_uci_automobile():
    """
    Descarga el dataset oficial con ucimlrepo y combina características y
    objetivo del repositorio para conservar todas las columnas originales.
    """
    from ucimlrepo import fetch_ucirepo
    dataset = fetch_ucirepo(id=10)
    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()
    df = pd.concat([features, targets], axis=1)
    
    # Reordena por los nombres oficiales
    if set(UCI_COLUMNS).issubset(df.columns):
        df = df[UCI_COLUMNS]
    return df

def clean_automobile(df):
    
    """Limpia valores '?', convierte variables numéricas e imputa faltantes."""
    df = df.copy()
    df.columns = [str(c).strip().lower().replace("_", "-") for c in df.columns]
    df = df.replace(r"^\s*\?\s*$", pd.NA, regex=True)

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "price" not in df.columns:
        raise ValueError("No se encontró la columna 'price' en el dataset.")
    df = df.dropna(subset=["price"]).copy()

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

    for col in numeric_cols:
        if col != "price":
            median = df[col].median()
            df[col] = df[col].fillna(median)

    for col in categorical_cols:
        mode = df[col].mode(dropna=True)
        replacement = mode.iloc[0] if not mode.empty else "desconocido"
        df[col] = df[col].fillna(replacement).astype(str).str.strip()

    # Elimina duplicados exactos, conservando la primera aparición.
    df = df.drop_duplicates().reset_index(drop=True)
    return df