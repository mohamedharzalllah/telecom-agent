"""Utilities to load the Excel data and expose a SQLite database for the SQL agent."""

from __future__ import annotations

import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src.config.setup import EXCEL_DIR, EXCEL_FILES, PROJECT_ROOT


# Persist the SQLite file so the SQL agent can reuse it across calls.
DB_DIR = PROJECT_ROOT / "data" / "xlsx"
DB_PATH = DB_DIR / "telecom.db"


def _read_excel_tables() -> Dict[str, pd.DataFrame]:
    """Load every Excel file in memory as a DataFrame keyed by table name."""
    frames: Dict[str, pd.DataFrame] = {}
    for table_name, filename in EXCEL_FILES.items():
        path = EXCEL_DIR / filename
        frames[table_name] = pd.read_excel(path)
    return frames


@lru_cache(maxsize=1)
def load_excel_frames() -> Dict[str, pd.DataFrame]:
    """
    Cached accessor for the Excel tables.

    Returns:
        Dict[str, pd.DataFrame]: mapping of table name -> dataframe.
    """
    return _read_excel_tables()


def build_sqlite_from_excel(force_rebuild: bool = False) -> Path:
    """
    Create (or refresh) a SQLite database populated from the Excel sources.

    Args:
        force_rebuild: if True, rewrites the SQLite file even if it exists.

    Returns:
        Path to the SQLite database file.
    """
    if DB_PATH.exists() and not force_rebuild:
        return DB_PATH

    frames = load_excel_frames()
    DB_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        for table_name, df in frames.items():
            df.to_sql(table_name, conn, if_exists="replace", index=False)

    return DB_PATH


def get_sqlite_uri() -> str:
    """Return the SQLAlchemy-style URI for the SQLite database."""
    build_sqlite_from_excel()
    return f"sqlite:///{DB_PATH}"


def table_schemas(sample_rows: int = 2) -> Dict[str, Dict[str, List]]:
    """
    Provide lightweight schema details for prompting.

    Args:
        sample_rows: number of sample rows to include for each table (0 to disable).
    """
    frames = load_excel_frames()
    meta: Dict[str, Dict[str, List]] = {}

    for table_name, df in frames.items():
        sample = df.head(sample_rows).to_dict(orient="records") if sample_rows > 0 else []
        meta[table_name] = {
            "columns": df.columns.tolist(),
            "sample_rows": sample,
        }
    return meta


TABLE_DESCRIPTIONS: Dict[str, str] = {
    "clients": "Identité des clients (nom, prénom, email, téléphone, adresse).",
    "abonnements": "Abonnements actifs des clients, avec forfait associé et dates.",
    "forfaits": "Catalogue des forfaits (nom, data mensuelle, minutes, sms, prix).",
    "consommation": "Suivi mensuel de consommation data / minutes / sms par client.",
    "factures": "Factures mensuelles des clients avec montants et statut de paiement.",
    "tickets_support": "Tickets de support client (catégorie, sujet, statut, priorité).",
}


def describe_tables() -> List[Tuple[str, str]]:
    """Return table names with short descriptions."""
    return list(TABLE_DESCRIPTIONS.items())
