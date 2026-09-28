"""Relecture des sorties de l'étape 1 (tables normalisées + journal) pour les étapes suivantes."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.load.canonical import TABLES
from src.load.events import journal_hash
from src.load.loader import LoadedData

# Colonnes produites au chargement mais lues par aucune étape suivante : pas relues (mémoire).
UNUSED_COLUMNS = {
    "payment": {"label_tokens"},
    "client_file": {"payment_reference_tokens", "payment_reference_numbers"},
    "debtor": {"name_tokens"},
    "assignor": {"name_tokens"},
}
# Tables dont les colonnes de listes restent en Arrow (les parties, petites, gardent des listes Python).
ARROW_LIST_TABLES = {"payment", "invoice", "client_file", "client_file_line"}


class InterimError(RuntimeError):
    pass


def read_table(path: Path, name: str, lean: bool = True) -> pd.DataFrame:
    """Une table de l'étape 1. `lean` : colonnes inutilisées écartées, colonnes de listes en Arrow
    (quelques octets par élément au lieu d'un objet Python par élément)."""
    if not lean:
        return pd.read_parquet(path)
    schema = pq.read_schema(path)
    columns = [f.name for f in schema if f.name not in UNUSED_COLUMNS.get(name, set())]
    lists = [f.name for f in schema if f.name in columns and name in ARROW_LIST_TABLES
             and (pa.types.is_list(f.type) or pa.types.is_large_list(f.type))]
    df = pd.read_parquet(path, columns=[c for c in columns if c not in lists])
    if lists:
        table = pq.read_table(path, columns=lists)
        for c in lists:
            df[c] = pd.Series(pd.arrays.ArrowExtensionArray(table.column(c)), index=df.index)
    return df[columns]


def load_interim(directory: str | Path, verify: bool = True, lean: bool = True) -> tuple[LoadedData, pd.DataFrame, dict]:
    """Tables, journal et métadonnées de l'étape 1. Vérifie l'empreinte du journal si `verify`."""
    d = Path(directory)
    meta_path = d / "journal_meta.json"
    if not meta_path.exists():
        raise InterimError(f"aucune sortie de l'étape 1 dans {d} : lancer d'abord le chargement")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    names = [*TABLES, "party_iban"]
    tables = {name: read_table(d / f"{name}.parquet", name, lean) for name in names if (d / f"{name}.parquet").exists()}
    journal = pd.read_parquet(d / "journal.parquet")
    if verify and journal_hash(journal) != meta["journal_sha256"]:
        raise InterimError("le journal ne correspond pas à son empreinte : relancer l'étape 1")
    mapped = meta.get("mapped_fields") or {
        name: [f.name for f in TABLES[name].fields if f.name in df.columns and df[f.name].notna().any()]
        for name, df in tables.items() if name in TABLES
    }
    return LoadedData(tables=tables, mapped_fields=mapped), journal, meta
