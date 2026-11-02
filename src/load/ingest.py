"""Ingestion assistée des sources réelles : lecture d'un fichier, profil des colonnes, contrôle du mapping.

Sert la page « Sources et ingestion » de l'interface. Rien n'est deviné : chaque colonne est choisie par
l'utilisateur ; ce module lit, montre et contrôle avec les convertisseurs du chargement (`convert_table`),
de sorte qu'un champ validé ici se charge à l'identique.

- `source_path` : chemin du fichier (absolu, ou relatif au répertoire des sources) ;
- `read_sample` : premières lignes, en texte comme au chargement ; `count_rows` : lignes du fichier ;
- `sniff_csv` : séparateur et encodage proposés, à confirmer ;
- `profile_columns` : remplissage, valeurs distinctes, exemples ;
- `check_fields` : par champ canonique, conversion d'un échantillon et statut ;
- `validate_sources` : fichiers complets, conversion de toutes les tables, contrôles qualité du chargement.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd

from src.load.canonical import IMPUTATION_STATUSES, TABLES, FieldType
from src.load.loader import Issue, LoadedData, _check_config, _read_source, convert_table

SAMPLE_ROWS = 20_000
SUPPORTED = (".csv", ".txt", ".tsv", ".parquet")
OK, INFO, WARNING, ERROR = "ok", "info", "attention", "erreur"


def source_path(base_dir: str | Path, source: str | None) -> Path | None:
    """Chemin du fichier source : absolu tel quel, sinon relatif au répertoire des sources."""
    return Path(base_dir) / source if source else None


def list_sources(base_dir: str | Path, limit: int = 500, depth: int = 3) -> list[str]:
    """Fichiers CSV / Parquet du répertoire des sources, sous-dossiers compris (jusqu'à `depth` niveaux, dossiers
    cachés exclus), chemins relatifs. Les entrées illisibles sont ignorées."""
    import os
    base = Path(base_dir)
    if not base.is_dir():
        return []
    found: list[str] = []
    for root, dirs, files in os.walk(base, onerror=lambda _: None):
        level = len(Path(root).relative_to(base).parts)
        dirs[:] = [] if level >= depth else sorted(d for d in dirs if not d.startswith((".", "__")))
        found += [str((Path(root) / f).relative_to(base)).replace("\\", "/") for f in sorted(files)
                  if Path(f).suffix.lower() in SUPPORTED]
        if len(found) >= limit:
            break
    return sorted(found)[:limit]


def sniff_csv(path: str | Path, size: int = 65_536) -> dict[str, Any]:
    """Options de lecture proposées pour un CSV (séparateur, encodage), à vérifier sur l'aperçu."""
    head = Path(path).open("rb").read(size)
    options: dict[str, Any] = {}
    if head.startswith(b"\xef\xbb\xbf"):
        options["encoding"] = "utf-8-sig"
        text = head[3:].decode("utf-8", errors="ignore")
    else:
        try:
            text = head.decode("utf-8")
        except UnicodeDecodeError as exc:
            if exc.start < len(head) - 4:           # pas seulement un caractère coupé en fin de bloc
                options["encoding"] = "latin-1"
                text = head.decode("latin-1")
            else:
                text = head[:exc.start].decode("utf-8")
    lines = text.splitlines()[:50]
    try:
        sep = csv.Sniffer().sniff("\n".join(lines), delimiters=";,\t|").delimiter
    except csv.Error:
        sep = ","
    if sep != ",":
        options["sep"] = sep
    return options


def read_sample(path: str | Path, read_options: dict[str, Any] | None = None, rows: int = SAMPLE_ROWS) -> pd.DataFrame:
    """Premières lignes du fichier, lues comme au chargement (texte pour les CSV, types natifs pour Parquet)."""
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq
        batch = next(pq.ParquetFile(path).iter_batches(batch_size=rows), None)
        return batch.to_pandas() if batch is not None else pd.read_parquet(path)
    opts = {"dtype": str, "keep_default_na": False, "na_values": [""], **(read_options or {})}
    return pd.read_csv(path, nrows=rows, **opts)


def count_rows(path: str | Path) -> int:
    """Lignes de données (Parquet : métadonnées ; CSV : sauts de ligne, en-tête exclu)."""
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq
        return pq.ParquetFile(path).metadata.num_rows
    n, last = 0, b"\n"
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 24):
            n += chunk.count(b"\n")
            last = chunk[-1:]
    return n - 1 + (last != b"\n")


def _filled(s: pd.Series) -> pd.Series:
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        return s.notna() & (s.astype("string").str.strip() != "")
    return s.notna()


def profile_columns(sample: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par colonne du fichier : type lu, remplissage, valeurs distinctes, exemples."""
    rows = []
    for col in sample.columns:
        s = sample[col]
        filled = _filled(s)
        values = s[filled].astype("string").unique()[:4]
        rows.append({"colonne": col, "type lu": str(s.dtype), "remplie": float(filled.mean()) if len(s) else 0.0,
                     "distinctes": int(s[filled].nunique()), "exemples": " · ".join(map(str, values))})
    return pd.DataFrame(rows, columns=["colonne", "type lu", "remplie", "distinctes", "exemples"])


def _overview(ftype: FieldType, conv: pd.Series) -> str:
    v = conv.dropna()
    if not len(v):
        return ""
    if ftype in (FieldType.DATE, FieldType.TIMESTAMP):
        fmt = "%d/%m/%Y" if ftype is FieldType.DATE else "%d/%m/%Y %H:%M"
        return f"{v.min():{fmt}} → {v.max():{fmt}}"
    if ftype is FieldType.AMOUNT:
        neg = int((v < 0).sum())
        return f"{v.min() / 100:,.2f} → {v.max() / 100:,.2f} €".replace(",", " ") + (f" · {neg} négatifs" if neg else "")
    distinct = v.nunique()
    return f"{distinct} distinctes · " + " · ".join(map(str, v.unique()[:3]))


def _convert(table_name: str, raw: pd.DataFrame, tcfg: dict[str, Any], schema_cfg: dict[str, Any],
             issues: list[Issue]) -> tuple[pd.DataFrame, list[str]]:
    """`convert_table` sur les seules colonnes présentes ; montants lus en centimes tant que l'unité manque."""
    present = {k: v for k, v in (tcfg.get("columns") or {}).items() if v and v in raw.columns}
    return convert_table(TABLES[table_name], raw, {**tcfg, "columns": present},
                         {**schema_cfg, "amount_unit": schema_cfg.get("amount_unit") or "cents"}, issues)


def check_fields(table_name: str, sample: pd.DataFrame, tcfg: dict[str, Any], schema_cfg: dict[str, Any],
                 converted: pd.DataFrame | None = None) -> pd.DataFrame:
    """Contrôle du mapping d'une table sur un échantillon : chaque champ canonique converti comme au chargement.

    Statuts : ok, info (optionnel non mappé, doublons de parties consolidés au chargement), attention (valeurs
    invalides, colonne vide, valeurs à traduire), erreur (champ requis non mappé, colonne absente du fichier,
    unité des montants non renseignée, identifiant en double). `converted` : conversion déjà faite."""
    table = TABLES[table_name]
    columns = {k: v for k, v in (tcfg.get("columns") or {}).items() if v}
    units_ok = schema_cfg.get("amount_unit") in ("cents", "units")
    if converted is None:
        converted, _ = _convert(table_name, sample, tcfg, schema_cfg, [])
    rows = []
    for f in table.fields:
        col = columns.get(f.name)
        row = {"champ": f.name, "type": f.type.value, "requis": f.required, "colonne": col or "",
               "remplis": None, "convertis": None, "invalides": 0, "exemples invalides": "", "aperçu": "",
               "statut": OK, "détail": ""}
        if not col:
            row["statut"], row["détail"] = (ERROR, "champ requis à mapper") if f.required else (INFO, "non mappé")
            rows.append(row)
            continue
        if col not in sample.columns:
            row["statut"], row["détail"] = ERROR, "colonne absente du fichier"
            rows.append(row)
            continue
        raw, conv = sample[col].reset_index(drop=True), converted[f.name]
        filled = _filled(raw)
        bad = filled & conv.isna()
        n_filled = int(filled.sum())
        row.update(remplis=float(filled.mean()) if len(raw) else 0.0, invalides=int(bad.sum()),
                   convertis=float(1 - bad.sum() / n_filled) if n_filled else None,
                   aperçu=_overview(f.type, conv))
        if bad.any():
            row["exemples invalides"] = " · ".join(map(str, raw[bad].astype("string").unique()[:5]))
        if f.type is FieldType.AMOUNT and not units_ok:
            row["statut"], row["détail"] = ERROR, "unité des montants (amount_unit) à renseigner"
        elif n_filled == 0:
            row["statut"], row["détail"] = (WARNING if f.required else INFO), "colonne mappée, toujours vide"
        elif bad.any():
            row["statut"], row["détail"] = WARNING, f"{int(bad.sum())} valeurs non convertibles (mises à vide)"
        if table_name == "imputation" and f.name == "status":
            unknown = conv.notna() & ~conv.isin(IMPUTATION_STATUSES)
            if unknown.any():
                row["statut"] = WARNING
                row["détail"] = ("valeurs à traduire en FULL / PARTIAL (value_maps) : "
                                 + ", ".join(map(str, conv[unknown].unique()[:8])))
        if f.name in table.primary_key and len(table.primary_key) == 1 and n_filled:
            dup = int(conv.dropna().duplicated().sum())
            if dup:
                party = table_name in ("assignor", "debtor")
                row["statut"] = INFO if party and row["statut"] == OK else (row["statut"] if party else ERROR)
                row["détail"] = (f"{dup} identifiants répétés : lignes regroupées par partie au chargement" if party
                                 else f"{dup} identifiants en double sur l'échantillon")
        rows.append(row)
    return pd.DataFrame(rows)


def table_status(fields: pd.DataFrame) -> str:
    for status in (ERROR, WARNING):
        if (fields["statut"] == status).any():
            return status
    return OK


# --- Validation complète ------------------------------------------------------------------------------------

def validate_sources(schema_cfg: dict[str, Any], base_dir: str | Path, log=print) -> dict[str, Any]:
    """Toutes les tables configurées, lues en entier et converties comme au chargement, puis les contrôles qualité
    du chargement (clés, intégrité référentielle, cohérence des imputations). Sans normalisation ni journal."""
    from src.load.events import derive_imputed_amounts
    from src.load.quality import check_quality

    base = Path(base_dir)
    problems = _check_config(schema_cfg, base)
    tables: dict[str, pd.DataFrame] = {}
    mapped: dict[str, list[str]] = {}
    issues: list[Issue] = []
    fields, volumes = [], []
    for name, table in TABLES.items():
        tcfg = (schema_cfg.get("tables") or {}).get(name) or {}
        if not tcfg.get("source") or not (base / tcfg["source"]).exists():
            continue
        log(f"  {name} : lecture de {tcfg['source']}")
        raw = _read_source(base / tcfg["source"], tcfg.get("read_options") or {})
        missing = sorted(v for v in (tcfg.get("columns") or {}).values() if v and v not in raw.columns)
        if missing:
            problems.append(f"tables.{name} : colonnes absentes du fichier {missing}")
            continue
        out, mapped[name] = _convert(name, raw, tcfg, schema_cfg, issues)
        fields.append(check_fields(name, raw, tcfg, schema_cfg, converted=out).assign(table=name))
        tables[name] = out
        volumes.append({"table": name, "lignes": len(raw), "colonnes du fichier": raw.shape[1],
                        "champs mappés": len(mapped[name])})
        del raw
    quality: list[Issue] = []
    required = [n for n, t in TABLES.items() if t.required]
    if all(n in tables for n in required):
        log("  contrôles qualité")
        inv = tables["invoice"]
        audit = {"invoice_current_amount": inv[["invoice_id", "current_amount"]].copy()} \
            if "current_amount" in mapped["invoice"] else {}
        tables["invoice"] = inv.drop(columns=["current_amount"])
        data = LoadedData(tables=tables, mapped_fields=mapped, issues=issues, audit=audit)
        derived = derive_imputed_amounts(tables["imputation"], tables["invoice"])
        quality = [i for i in check_quality(data, derived)
                   if not (i.table in ("assignor", "debtor") and i.kind == "cle_dupliquee")]
    else:
        problems.append("contrôles qualité non faits : tables requises manquantes ou illisibles "
                        f"({', '.join(n for n in required if n not in tables)})")
    issue_cols = ["table", "champ", "contrôle", "lignes", "détail"]
    as_rows = lambda found: pd.DataFrame([{"table": i.table, "champ": i.field, "contrôle": i.kind,  # noqa: E731
                                           "lignes": i.count, "détail": i.detail} for i in found], columns=issue_cols)
    field_table = pd.concat(fields, ignore_index=True) if fields else pd.DataFrame()
    if len(field_table):
        field_table = field_table[["table", *[c for c in field_table.columns if c != "table"]]]
    blocking = list(problems)
    if len(field_table):
        blocking += [f"{r.table}.{r.champ} : {r.détail}" for r in field_table.itertuples() if r.statut == ERROR]
    return {"problems": blocking, "fields": field_table, "conversion": as_rows(issues), "quality": as_rows(quality),
            "volumes": pd.DataFrame(volumes, columns=["table", "lignes", "colonnes du fichier", "champs mappés"])}
