import json
from datetime import datetime

import pandas as pd
import streamlit as st
import yaml

from src.config import resolve_path
from src.load.canonical import TABLES
from src.load.ingest import (
    ERROR, INFO, OK, SUPPORTED, WARNING, check_fields, count_rows, list_sources, profile_columns, read_sample,
    sniff_csv, source_path, table_status,
)
from src.load.loader import _check_config
from src.load.schema_io import FIELD_NOTES, GLOBAL_KEYS, TABLE_NOTES, VALUE_MAP_FIELDS, load_schema, save_schema
from src.ui.common import REAL, dataset, num, read_json, reports_dir, run_task, schema_path, step_header
from src.yaml_io import scalar

step_header("ingestion")
# Les saisies d'une table restent en mémoire quand on en affiche une autre (Streamlit efface sinon l'état des
# widgets non affichés).
for key in [k for k in st.session_state if str(k).startswith("ingest.")]:
    st.session_state[key] = st.session_state[key]

saved = load_schema(schema_path())
STATUS_LABEL = {OK: "✅ ok", INFO: "ℹ️ info", WARNING: "⚠️ attention", ERROR: "❌ erreur"}
CANONICAL_VALUES = ("FULL", "PARTIAL")
if dataset() != REAL:
    st.info("Jeu synthétique sélectionné : il est décrit par config/schema.synthetic.yaml. Cette page prépare les "
            f"sources réelles (`{schema_path()}`) ; le chargement se lance avec le jeu « réel » (barre latérale).",
            icon=":material/info:")
st.caption("Pour chaque table : le chemin du fichier, la lecture d'un échantillon, la correspondance colonne par "
           "colonne, contrôlée en direct avec les convertisseurs du chargement, puis la validation sur les fichiers "
           "complets avant de charger. Rien n'est deviné : chaque colonne se choisit dans l'en-tête du fichier.")


def k(table: str, part: str) -> str:
    return f"ingest.{table}.{part}"


def _init_globals() -> None:
    for key in GLOBAL_KEYS:
        choice = key in ("amount_unit", "decimal_separator")
        st.session_state.setdefault(f"ingest.{key}", saved[key] if choice else saved[key] or "")


def _init_table(name: str) -> None:
    t = saved["tables"][name]
    defaults = {k(name, "source"): t["source"] or "", k(name, "opts"): scalar(t["read_options"]) if t["read_options"]
                else "", k(name, "date_format"): t.get("date_format") or "",
                k(name, "timestamp_format"): t.get("timestamp_format") or ""}
    defaults |= {k(name, f"col.{f.name}"): t["columns"].get(f.name) for f in TABLES[name].fields}
    for fld in VALUE_MAP_FIELDS.get(name, []):
        mapping = (t.get("value_maps") or {}).get(fld) or {}
        for value in CANONICAL_VALUES:
            real = mapping.get(value, [])
            defaults[k(name, f"vm.{fld}.{value}")] = ", ".join(map(str, real if isinstance(real, list) else [real]))
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _options(text: str) -> dict | None:
    try:
        value = yaml.safe_load(text or "") or {}
    except yaml.YAMLError:
        return None
    return value if isinstance(value, dict) else None


def entry(name: str) -> dict:
    """Configuration de la table telle que saisie (ou telle qu'enregistrée si la table n'a pas été ouverte)."""
    s, t = st.session_state, saved["tables"][name]
    if k(name, "source") not in s:
        return t
    out = {"source": (s[k(name, "source")] or "").strip() or None,
           "read_options": _options(s[k(name, "opts")]) if _options(s[k(name, "opts")]) is not None
           else t["read_options"],
           "columns": {f.name: s.get(k(name, f"col.{f.name}")) for f in TABLES[name].fields}}
    for opt in ("date_format", "timestamp_format"):
        if s.get(k(name, opt)):
            out[opt] = s[k(name, opt)]
    if name in VALUE_MAP_FIELDS:
        out["value_maps"] = {fld: {v: [x.strip() for x in s.get(k(name, f"vm.{fld}.{v}"), "").split(",") if x.strip()]
                                   for v in CANONICAL_VALUES if s.get(k(name, f"vm.{fld}.{v}"), "").strip()}
                             for fld in VALUE_MAP_FIELDS[name]}
    return out


def _canonical(cfg: dict) -> str:
    """Forme comparable d'une configuration (valeurs traduites toujours en listes)."""
    out = {key: cfg.get(key) or None for key in GLOBAL_KEYS}
    out["tables"] = {}
    for name, t in cfg["tables"].items():
        t = {**t, "source": t.get("source") or None}
        if "value_maps" in t:
            t["value_maps"] = {f: {v: (r if isinstance(r, list) else [r]) for v, r in (m or {}).items() if r}
                               for f, m in t["value_maps"].items()}
        out["tables"][name] = t
    return json.dumps(out, sort_keys=True, default=str)


def current_config() -> dict:
    return {**{key: st.session_state.get(f"ingest.{key}", saved[key]) or None for key in GLOBAL_KEYS},
            "tables": {name: entry(name) for name in TABLES}}


@st.cache_data(show_spinner="Lecture du fichier…", max_entries=20)
def _sample(path: str, options: str, mtime: float) -> pd.DataFrame:
    return read_sample(path, json.loads(options))


@st.cache_data(show_spinner=False, max_entries=50)
def _rows(path: str, mtime: float) -> int:
    return count_rows(path)


def sample_of(path, options: dict) -> tuple[pd.DataFrame | None, str | None]:
    if path is None:
        return None, "Renseigner le chemin du fichier."
    if not path.exists():
        return None, f"Fichier introuvable : `{path}`"
    if path.suffix.lower() not in SUPPORTED:
        return None, f"Format non pris en charge ({path.suffix}) : CSV ou Parquet."
    try:
        return _sample(str(path), json.dumps(options, sort_keys=True, default=str), path.stat().st_mtime), None
    except Exception as exc:  # options de lecture inadaptées, fichier illisible
        return None, f"Lecture impossible avec ces options : {exc}"


def styled(fields: pd.DataFrame):
    colors = {OK: "", INFO: "color: #5b7fa3", WARNING: "background-color: rgba(235, 104, 52, 0.15)",
              ERROR: "background-color: rgba(214, 39, 40, 0.18)"}
    shown = fields.assign(statut=fields["statut"].map(STATUS_LABEL))
    return shown.style.apply(lambda row: [colors[fields.loc[row.name, "statut"]]] * len(row), axis=1)


FIELD_COLUMNS = {"remplis": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent"),
                 "convertis": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent"),
                 "requis": st.column_config.CheckboxColumn()}

# --- Conventions -------------------------------------------------------------------------------------------
_init_globals()
st.subheader("Conventions des sources")
c = st.columns(3)
c[0].selectbox("Unité des montants", [None, "cents", "units"], key="ingest.amount_unit",
               format_func=lambda v: "— à renseigner —" if v is None else v, help=GLOBAL_KEYS["amount_unit"])
c[1].selectbox("Séparateur décimal", [None, ".", ","], key="ingest.decimal_separator",
               format_func=lambda v: "— à renseigner —" if v is None else f"« {v} »",
               help=GLOBAL_KEYS["decimal_separator"])
c[2].text_input("Fuseau des horodatages", key="ingest.source_timezone", help=GLOBAL_KEYS["source_timezone"])
c[0].text_input("Format des dates", key="ingest.date_format", help=GLOBAL_KEYS["date_format"])
c[1].text_input("Format des horodatages", key="ingest.timestamp_format", help=GLOBAL_KEYS["timestamp_format"])
c[2].text_input("Répertoire des sources", key="ingest.base_dir", help=GLOBAL_KEYS["base_dir"])
base = resolve_path(st.session_state["ingest.base_dir"] or ".")
st.caption(f"Répertoire des sources : `{base}` — {'trouvé' if base.is_dir() else 'introuvable'}. Un chemin de "
           "fichier relatif part de ce répertoire ; un chemin absolu est pris tel quel.")

# --- Table ---------------------------------------------------------------------------------------------------


def _badge(name: str) -> str:
    e = entry(name)
    if not e["source"]:
        return "○" if not TABLES[name].required else "❌"
    path = source_path(base, e["source"])
    missing = [f for f in TABLES[name].fields if f.required and not e["columns"].get(f.name)]
    return "❌" if not path.exists() else ("⚠️" if missing else "✅")


st.subheader("Fichier et correspondance des colonnes")
# Libellés fixes : un libellé qui change pendant la saisie ferait perdre la sélection.
c = st.columns([2, 3], vertical_alignment="bottom")
name = c[0].selectbox("Table", list(TABLES), key="ingestion.table",
                      format_func=lambda n: f"{n} — {'requise' if TABLES[n].required else 'optionnelle'}")
_init_table(name)
c[1].caption("  ".join(f"{_badge(n)} {n}" for n in TABLES))
if name in TABLE_NOTES:
    st.caption(TABLE_NOTES[name])
files = list_sources(base)


def _pick(table: str) -> None:
    choice = st.session_state.get(f"ingestion.pick.{table}")
    if choice:
        st.session_state[k(table, "source")] = choice


def _sniff(table: str) -> None:
    path = source_path(base, (st.session_state[k(table, "source")] or "").strip() or None)
    if path is not None and path.exists() and path.suffix.lower() != ".parquet":
        found = sniff_csv(path)
        st.session_state[k(table, "opts")] = scalar(found) if found else ""


left, right = st.columns([3, 2], vertical_alignment="bottom")
left.text_input("Chemin du fichier (absolu, ou relatif au répertoire des sources)", key=k(name, "source"))
right.selectbox("ou le choisir dans le répertoire des sources", [None, *files], key=f"ingestion.pick.{name}",
                format_func=lambda v: f"— {len(files)} fichiers CSV / Parquet —" if v is None else v,
                on_change=_pick, args=(name,))
path = source_path(base, (st.session_state[k(name, "source")] or "").strip() or None)
left, right = st.columns([3, 2], vertical_alignment="bottom")
left.text_input("Options de lecture (YAML)", key=k(name, "opts"),
                help="Passées à pandas.read_csv, ex. {sep: ';', encoding: latin-1, decimal: ','}.")
right.button("Détecter séparateur et encodage", key=f"ingestion.sniff.{name}", on_click=_sniff, args=(name,),
             icon=":material/auto_fix_high:", disabled=path is None or not path.exists() or path.suffix == ".parquet",
             help="Proposition à vérifier sur l'aperçu.")
options = _options(st.session_state[k(name, "opts")])
if options is None:
    st.error("Options de lecture : YAML invalide (attendu un dictionnaire, ex. {sep: ';'}).")
    options = {}

sample, problem = sample_of(path, options)
header: list[str] = []
examples: dict[str, str] = {}
if problem:
    (st.info if path is None else st.warning)(problem)
else:
    header = list(map(str, sample.columns))
    profile = profile_columns(sample)
    examples = dict(zip(profile["colonne"], profile["exemples"]))
    st.caption(f"`{path}` · {num(_rows(str(path), path.stat().st_mtime))} lignes · {len(header)} colonnes · "
               f"contrôles sur les {num(len(sample))} premières lignes.")
    if len(header) == 1 and path.suffix.lower() != ".parquet":
        st.warning("Une seule colonne lue : le séparateur est probablement autre (« Détecter séparateur et "
                   "encodage »).", icon=":material/warning:")
    tab_profile, tab_preview = st.tabs(["Colonnes du fichier", "Aperçu"])
    with tab_profile:
        st.dataframe(profile, hide_index=True, width="stretch", height=300, column_config={
            "remplie": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent")})
    with tab_preview:
        st.dataframe(sample.head(50), hide_index=True, width="stretch", height=300)

st.markdown("**Correspondance** — champ du modèle canonique ← colonne du fichier (* : requis)")
grid = st.columns(3)
for i, f in enumerate(TABLES[name].fields):
    key = k(name, f"col.{f.name}")
    current = st.session_state.get(key)
    choices = [None, *header] + ([current] if current and current not in header else [])
    note = FIELD_NOTES.get((name, f.name))
    grid[i % 3].selectbox(
        f"{f.name}{' *' if f.required else ''} · {f.type.value}", choices, key=key,
        format_func=lambda v: "— non mappé —" if v is None else (
            f"{v} (absente du fichier)" if header and v not in header else
            f"{v}  ·  {examples.get(v, '')[:40]}" if examples.get(v) else v),
        help=f"{'Requis' if f.required else 'Optionnel'}. {note or ''}")

with st.expander("Formats propres à cette table et traduction des valeurs", icon=":material/tune:",
                 expanded=name in VALUE_MAP_FIELDS):
    c = st.columns(2)
    c[0].text_input("Format des dates (remplace la convention générale)", key=k(name, "date_format"))
    c[1].text_input("Format des horodatages (remplace la convention générale)", key=k(name, "timestamp_format"))
    for fld in VALUE_MAP_FIELDS.get(name, []):
        real = st.session_state.get(k(name, f"col.{fld}"))
        if sample is not None and real in sample.columns:
            counts = sample[real].value_counts(dropna=False).head(12)
            st.caption(f"Valeurs de `{real}` sur l'échantillon : "
                       + " · ".join(f"`{v}` ({n})" for v, n in counts.items()))
        mc = st.columns(2)
        for j, value in enumerate(CANONICAL_VALUES):
            mc[j].text_input(f"{fld} = {value} : valeurs réelles, séparées par des virgules",
                             key=k(name, f"vm.{fld}.{value}"))

config = current_config()
if sample is not None:
    fields = check_fields(name, sample, config["tables"][name], config)
    status = table_status(fields)
    {OK: st.success, WARNING: st.warning, ERROR: st.error}[status](
        {OK: "Mapping valide sur l'échantillon.",
         WARNING: "Mapping utilisable, avec des points à vérifier (lignes en orange).",
         ERROR: "Mapping incomplet ou incohérent (lignes en rouge) : le chargement serait refusé."}[status],
        icon={OK: ":material/check_circle:", WARNING: ":material/warning:", ERROR: ":material/error:"}[status])
    st.dataframe(styled(fields), hide_index=True, width="stretch", column_config=FIELD_COLUMNS)
    st.caption("« convertis » : part des valeurs non vides que le chargement sait lire (date, montant en centimes, "
               "identifiant) ; les autres deviendraient vides. « aperçu » : plage des dates et des montants convertis.")

changed = _canonical(config) != _canonical(saved)
c = st.columns([1, 3], vertical_alignment="center")
if c[0].button("Enregistrer schema.yaml", type="primary", icon=":material/save:", key="ingestion.save",
               disabled=not changed):
    save_schema(config, schema_path())
    st.toast("schema.yaml enregistré", icon=":material/check:")
    st.rerun()
c[1].caption(f"Modifications non enregistrées dans `{schema_path()}`." if changed else
             f"Saisie identique à `{schema_path()}`.")

# --- Validation ----------------------------------------------------------------------------------------------
st.divider()
st.subheader("Validation avant chargement")
rows = []
for table_name, table in TABLES.items():
    e = config["tables"][table_name]
    p = source_path(base, e["source"])
    required = [f for f in table.fields if f.required]
    mapped = sum(1 for f in required if e["columns"].get(f.name))
    row = {"table": table_name, "requise": table.required, "fichier": e["source"] or "", "lignes": None,
           "champs requis mappés": f"{mapped}/{len(required)}", "échantillon": ""}
    if p is not None and p.exists():
        s, err = sample_of(p, e["read_options"] or {})
        row["lignes"] = _rows(str(p), p.stat().st_mtime)
        row["échantillon"] = STATUS_LABEL[table_status(check_fields(table_name, s, e, config))] if s is not None \
            else f"❌ {err}"
    elif e["source"]:
        row["échantillon"] = "❌ fichier introuvable"
    elif table.required:
        row["échantillon"] = "❌ fichier à renseigner"
    rows.append(row)
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
    "lignes": st.column_config.NumberColumn(format="localized"), "requise": st.column_config.CheckboxColumn()})

problems = _check_config(config, base)
if problems:
    with st.expander(f"{len(problems)} point(s) bloquant le chargement", icon=":material/error:", expanded=False):
        st.markdown("\n".join(f"- {p}" for p in problems))
else:
    st.success("Configuration complète : tous les fichiers requis sont trouvés et leurs champs requis mappés.",
               icon=":material/check_circle:")

report = reports_dir() / "sources_validation.json"
c = st.columns(3)
if c[0].button("Valider les fichiers complets", icon=":material/fact_check:", key="ingestion.validate",
               disabled=changed, help="Lit chaque fichier en entier, le convertit comme au chargement et lance les "
                                      "contrôles qualité (clés, intégrité référentielle, imputations). Enregistrer "
                                      "d'abord la saisie."):
    code, lines = run_task("validate_sources", "Validation des sources")
    if code != 0:
        st.error("\n".join(lines[-6:]))
if dataset() == REAL:
    if c[1].button("Charger les données réelles (étape 1)", type="primary", icon=":material/play_arrow:",
                   key="ingestion.load", disabled=bool(problems) or changed):
        code, lines = run_task("load", "Étape 1 — chargement")
        if code != 0:
            st.error("Échec du chargement :\n\n" + "\n".join(lines[-8:]))
        else:
            st.success("Données chargées : résultats dans la page Chargement.", icon=":material/check_circle:")

result = read_json(report)
if result is None:
    st.caption("Aucune validation complète encore lancée.")
    st.stop()
when = datetime.fromtimestamp(report.stat().st_mtime)
st.markdown(f"**Dernière validation complète** — {when:%d/%m/%Y %H:%M}")
if schema_path().exists() and schema_path().stat().st_mtime > report.stat().st_mtime:
    st.warning("schema.yaml a été modifié depuis cette validation : la relancer.", icon=":material/sync_problem:")
if result["prêt_à_charger"]:
    st.success(f"Prêt à charger. {result['anomalies_qualité']} type(s) d'anomalie de qualité à examiner "
               "(non bloquants).", icon=":material/check_circle:")
else:
    st.error("Chargement impossible en l'état :\n\n" + "\n".join(f"- {p}" for p in result["bloquants"]),
             icon=":material/block:")
read = lambda n: pd.read_csv(reports_dir() / f"sources_validation_{n}.csv") \
    if (reports_dir() / f"sources_validation_{n}.csv").exists() else pd.DataFrame()  # noqa: E731
tabs = st.tabs(["Volumes", "Champs à vérifier", "Contrôles qualité", "Valeurs non convertibles", "Tous les champs"])
with tabs[0]:
    st.dataframe(read("volumes"), hide_index=True, width="stretch",
                 column_config={"lignes": st.column_config.NumberColumn(format="localized")})
fields_all = read("fields")
with tabs[1]:
    todo = fields_all[fields_all["statut"].isin([WARNING, ERROR])] if len(fields_all) else fields_all
    if todo.empty:
        st.success("Tous les champs mappés se convertissent sans perte.", icon=":material/check_circle:")
    else:
        st.dataframe(todo, hide_index=True, width="stretch", column_config=FIELD_COLUMNS)
with tabs[2]:
    quality = read("quality")
    if quality.empty:
        st.success("Aucune anomalie : clés uniques, références croisées résolues, imputations cohérentes.",
                   icon=":material/check_circle:")
    else:
        st.dataframe(quality, hide_index=True, width="stretch",
                     column_config={"lignes": st.column_config.NumberColumn(format="localized")})
        st.caption("Mêmes contrôles qu'au chargement (page Chargement, onglet Anomalies) : rien n'est corrigé, "
                   "chaque anomalie est comptée pour décision.")
with tabs[3]:
    conversion = read("conversion")
    if len(conversion):
        st.dataframe(conversion, hide_index=True, width="stretch")
    else:
        st.success("Toutes les valeurs non vides sont convertibles.", icon=":material/check_circle:")
with tabs[4]:
    st.dataframe(fields_all, hide_index=True, width="stretch", column_config=FIELD_COLUMNS)
