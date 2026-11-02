"""Interface de pilotage — version en un fichier.

    streamlit run reconciliation_ui.py

Exige reconciliation.py dans le même dossier. Écoute locale,
télémétrie désactivée par .streamlit/config.toml (créé par reconciliation.py).

Généré par tools/build_single_ui.py depuis src/ui.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import altair as alt
import json
import os
import pandas as pd
import pyarrow.parquet as pq
import streamlit as st
import subprocess
import time
import types
import typing
import yaml
from dataclasses import dataclass
from datetime import date, datetime
from pydantic import BaseModel, ValidationError
from typing import Any

from reconciliation import (
    AUTO_KO,
    AUTO_OK,
    DEFAULT_RULES_PATH,
    DEFAULT_SCHEMA_PATH,
    DEFAULT_SETTINGS_PATH,
    ERROR,
    ERROR_TYPES,
    FIELD_NOTES,
    GLOBAL_KEYS,
    INFO,
    IN_REVIEW,
    MODEL_DIRNAME,
    NO_DECISION,
    NO_TRUTH,
    OK,
    OUTSIDE,
    PERIODS,
    PURGE,
    REASONS,
    REPO_ROOT,
    RulesConfig,
    SEGMENTS,
    SUPPORTED,
    Settings,
    SplitError,
    SplitSettings,
    TABLES,
    TABLE_NOTES,
    VALUE_MAP_FIELDS,
    WARNING,
    _check_config,
    assign_period,
    check_fields,
    clopper_pearson_lower,
    compare_outcomes,
    compute_split,
    count_rows,
    diagnostics,
    error_table,
    find_pyspy,
    label_numbers,
    list_sources,
    load_rules,
    load_schema,
    load_settings,
    memory,
    normalize_text,
    profile_columns,
    profiling,
    rate_by,
    read_sample,
    reason_table,
    reference_keys,
    resolve_path,
    save_rules,
    save_schema,
    save_settings,
    scalar,
    sniff_csv,
    source_path,
    table_status,
    with_precision_bounds,
)


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# Éléments communs
# ════════════════════════════════════════════════════════════════════════════════════════════════════

# --- Registre des étapes ------------------------------------------------------------

DONE, PARTIAL, TODO = "done", "partial", "todo"
STATUS_BADGE = {
    DONE: ("Opérationnel", "green", ":material/check_circle:"),
    PARTIAL: ("Partiel", "orange", ":material/timelapse:"),
    TODO: ("À venir", "gray", ":material/schedule:"),
}


@dataclass(frozen=True)
class Step:
    key: str
    title: str
    icon: str
    status: str
    summary: str
    note: str = ""


STEPS: dict[str, Step] = {s.key: s for s in [
    Step("config", "Configuration", ":material/settings:", DONE,
         "Paramètres généraux, fichiers de configuration, environnement d'exécution."),
    Step("ingestion", "Sources et ingestion", ":material/upload_file:", DONE,
         "Fichiers sources réels : lecture, correspondance des colonnes contrôlée en direct, validation avant "
         "chargement."),
    Step("load", "1 · Chargement", ":material/database:", DONE,
         "Chargement, normalisation, journal d'événements, contrôles qualité."),
    Step("split", "2 · Découpage temporel", ":material/date_range:", DONE,
         "Périodes train / validation / test, état du grand livre à date et boucle quotidienne sans fuite."),
    Step("allocation", "3 · Allocation", ":material/account_tree:", DONE,
         "Rattachement de chaque paiement à une liste classée de débiteurs candidats, rappel mesuré."),
    Step("rules", "4 · Réconciliation algorithmique", ":material/rule:", DONE,
         "Règles déterministes, validation seulement si la solution est unique : la baseline à battre."),
    Step("ml", "5 · Réconciliation ML", ":material/model_training:", DONE,
         "Candidats, scoring LightGBM en deux passes, ensembles, décision par seuils sur le résiduel de l'étape 4."),
    Step("evaluation", "6 · Évaluation", ":material/monitoring:", DONE,
         "Backtest, taux d'automatisation à précision fixée, cascade, découpages par étape, règle, mois, groupe."),
    Step("analysis", "Évaluation approfondie", ":material/troubleshoot:", DONE,
         "Erreurs, résiduel manuel expliqué, segments, comparaison règles / ML : une ligne par paiement."),
    Step("runtime", "Temps de run", ":material/timer:", DONE,
         "D'où vient le temps des étapes : profil par étape et sous-étape, mémoire, profil par fonction."),
]}


def step_header(key: str) -> Step:
    step = STEPS[key]
    label, color, icon = STATUS_BADGE[step.status]
    st.title(step.title)
    st.badge(label, icon=icon, color=color)
    st.caption(step.summary)
    if step.note:
        (st.info if step.status == TODO else st.caption)(step.note)
    return step


# --- Chemins -------------------------------------------------------------------------

REAL, SYNTHETIC = "réel", "synthétique"


def settings_path() -> Path:
    return Path(os.environ.get("RECON_SETTINGS", DEFAULT_SETTINGS_PATH))


def schema_path() -> Path:
    return Path(os.environ.get("RECON_SCHEMA", DEFAULT_SCHEMA_PATH))


def rules_path() -> Path:
    return Path(os.environ.get("RECON_RULES", DEFAULT_RULES_PATH))


def current_settings() -> Settings:
    return load_settings(settings_path())


def dataset() -> str:
    return REAL


def interim_dir(settings: Settings | None = None, ds: str | None = None) -> Path:
    base = resolve_path((settings or current_settings()).paths.interim_dir)
    return base / "synthetic" if (ds or dataset()) == SYNTHETIC else base


def reports_dir(settings: Settings | None = None, ds: str | None = None) -> Path:
    base = resolve_path((settings or current_settings()).paths.reports_dir)
    return base / "synthetic" if (ds or dataset()) == SYNTHETIC else base


def runtime_dir(settings: Settings | None = None) -> Path:
    """Suivi mémoire (`memory.csv`) et profils (`profiles/`) : dossier commun au notebook et à l'interface."""
    return resolve_path((settings or current_settings()).paths.reports_dir)


def model_dir(settings: Settings | None = None, ds: str | None = None) -> Path:
    base = resolve_path((settings or current_settings()).paths.models_dir)
    return base / ("synthetic" if (ds or dataset()) == SYNTHETIC else "real") / MODEL_DIRNAME


def read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load_meta(settings: Settings | None = None, ds: str | None = None) -> dict[str, Any] | None:
    path = interim_dir(settings, ds) / "journal_meta.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


PERIOD_LABELS = {"train": "Entraînement", "validation": "Validation", "test": "Test (backtest)",
                 "all": "Tout l'historique"}
MATCHER_LABELS = {"pipeline": "Règles + ML (étapes 4 et 5)", "rules": "Règles (étape 4)", "null": "Rapprocheur vide"}


def pct(value: float | None, digits: int = 1) -> str:
    return "—" if value is None or value != value else f"{value:.{digits}%}"


def num(value: float | None) -> str:
    return "—" if value is None or value != value else f"{int(value):,}".replace(",", "\u202f")


def stale(context_sha: str | None, what: str = "Résultat") -> bool:
    """Avertit si un résultat a été calculé sur un autre journal que le dernier chargement."""
    meta = load_meta()
    if meta and context_sha and context_sha != meta["journal_sha256"]:
        st.warning(f"{what} calculé sur un autre journal que le dernier chargement : le relancer.",
                   icon=":material/warning:")
        return True
    return False


def require_loaded_data() -> dict[str, Any] | None:
    meta = load_meta()
    if meta is None:
        st.warning(f"Aucune donnée chargée pour le jeu **{dataset()}** : lancer d'abord l'étape 1 "
                   "(page Chargement).")
    return meta


# --- Formulaires générés depuis les modèles ------------------------------------------


def _unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    args = typing.get_args(annotation)
    if typing.get_origin(annotation) in (typing.Union, types.UnionType) and type(None) in args:
        rest = [a for a in args if a is not type(None)]
        return rest[0], True
    return annotation, False


def render_model(model: BaseModel, key: str, disabled: bool = False, skip: tuple[str, ...] = ()) -> dict[str, Any]:
    """Un widget par champ (label = description, aide = clé YAML). Retourne les valeurs saisies.

    Si le modèle a un champ `enabled` désactivé, ses autres champs sont grisés.
    """
    values: dict[str, Any] = {}
    fields = type(model).model_fields
    if "enabled" in fields and "enabled" not in skip:
        values["enabled"] = field_widget(model, "enabled", key, disabled)
        disabled = disabled or not values["enabled"]
    for name, info in fields.items():
        if name in values or name in skip:
            continue
        value = getattr(model, name)
        if isinstance(value, BaseModel):
            st.markdown(f"**{info.description or name}**")
            with st.container(border=True):
                values[name] = render_model(value, f"{key}.{name}", disabled)
        else:
            values[name] = field_widget(model, name, key, disabled)
    return values


def field_widget(model: BaseModel, name: str, key: str, disabled: bool = False) -> Any:
    info = type(model).model_fields[name]
    value = getattr(model, name)
    label = info.description or name
    wkey = f"{key}.{name}"
    help_ = f"Clé YAML : `{wkey}`"
    annotation, optional = _unwrap_optional(info.annotation)

    if annotation is bool:
        return st.toggle(label, value=value, key=wkey, help=help_, disabled=disabled)
    if optional:
        active = st.checkbox(f"{label}", value=value is not None, key=f"{wkey}.set", help=help_,
                             disabled=disabled)
        if not active:
            return None
        default = value if value is not None else (info.default if info.default is not None else None)
        return _scalar_widget(annotation, "↳ valeur", default, wkey, help_, disabled)
    if typing.get_origin(annotation) is typing.Literal:
        options = list(typing.get_args(annotation))
        return st.selectbox(label, options, index=options.index(value) if value in options else 0, key=wkey,
                            help=help_, disabled=disabled)
    if typing.get_origin(annotation) is list:
        options = list(dict.fromkeys([*(info.default_factory() if info.default_factory else []), *value]))
        return st.multiselect(label, options=options, default=value, key=wkey, help=help_, disabled=disabled)
    return _scalar_widget(annotation, label, value, wkey, help_, disabled)


def _scalar_widget(annotation: Any, label: str, value: Any, wkey: str, help_: str, disabled: bool) -> Any:
    if annotation is int:
        return int(st.number_input(label, value=int(value if value is not None else 0), step=1,
                                   key=wkey, help=help_, disabled=disabled))
    if annotation is float:
        return float(st.number_input(label, value=float(value if value is not None else 0.0), step=0.001,
                                     format="%.4f", key=wkey, help=help_, disabled=disabled))
    if annotation is date:
        return st.date_input(label, value=value or date.today(), key=wkey, help=help_, disabled=disabled)
    text = st.text_input(label, value="" if value is None else str(value), key=wkey, help=help_,
                         disabled=disabled)
    return text or None


def validation_errors(exc: ValidationError) -> list[str]:
    return [f"`{'.'.join(str(p) for p in e['loc'])}` : {e['msg']}" for e in exc.errors()]


def save_section(settings: Settings, section: str, values: dict[str, Any], key: str) -> None:
    """Boutons Enregistrer / Valeurs par défaut pour une section de `Settings`."""
    left, right = st.columns([1, 1])
    if left.button("Enregistrer", type="primary", key=f"{key}.save", icon=":material/save:"):
        try:
            updated = Settings.model_validate({**settings.model_dump(), section: values})
        except ValidationError as exc:
            st.error("Paramètres invalides :\n\n" + "\n".join(f"- {m}" for m in validation_errors(exc)))
            return
        save_settings(updated, settings_path())
        st.toast(f"Paramètres « {section} » enregistrés dans {settings_path().name}", icon=":material/check:")
    if right.button("Valeurs par défaut", key=f"{key}.reset", icon=":material/restart_alt:"):
        defaults = Settings.model_validate({**settings.model_dump(), section: {}})
        save_settings(defaults, settings_path())
        for k in [k for k in st.session_state if str(k).startswith(f"{key}.")]:
            del st.session_state[k]
        st.rerun()


# --- Exécution en sous-processus -------------------------------------------------------


def run_task(task: str, label: str, **kwargs) -> tuple[int, list[str]]:
    """Exécute `Project.<task>(**kwargs)` (src/api.py) dans un processus séparé, journal en direct.

    Même code que dans le notebook ; le processus séparé garde l'interface réactive et sa mémoire
    indépendante des 2 M de lignes traitées.
    """
    project = {"dataset": "synthetic" if dataset() == SYNTHETIC else "real", "settings_path": str(settings_path()),
               "schema_path": str(schema_path()), "rules_path": str(rules_path())}
    tracking = {"_monitor": st.session_state.get("ui.monitor", True),
                "_profile": st.session_state.get("ui.profile", False)}
    payload = json.dumps({**project, **tracking, **kwargs})
    code_line = "import json, sys; from reconciliation import run_task; run_task(sys.argv[1], json.loads(sys.argv[2]))"
    cmd = [sys.executable, "-c", code_line, task, payload]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    lines: list[str] = []
    with st.status(label, expanded=True) as status:
        shown = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
        st.code(f"Project(dataset={project['dataset']!r}).{task}({shown})", language="python")
        log = st.empty()
        start = time.perf_counter()
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", env=env)
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line.rstrip())
            log.code("\n".join(lines[-30:]), language="text")
        code = proc.wait()
        minutes = (time.perf_counter() - start) / 60
        status.update(label=f"{label} — {'terminé' if code == 0 else f'échec (code {code})'} en {minutes:.1f} min",
                      state="complete" if code == 0 else "error", expanded=code != 0)
    return code, lines


# --- Lecture des sorties -------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _read_parquet_cached(path: str, columns: tuple[str, ...] | None, mtime: float) -> pd.DataFrame:
    return pd.read_parquet(path, columns=list(columns) if columns else None)


def read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Lecture mise en cache, invalidée quand le fichier change."""
    return _read_parquet_cached(str(path), tuple(columns) if columns else None, path.stat().st_mtime)


def read_report(name: str) -> pd.DataFrame | None:
    path = reports_dir() / name
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(show_spinner="Calcul du diagnostic…", max_entries=32)
def _diagnostic(name: str, journal: str, key: Any, args: tuple, kwargs: dict) -> Any:
    return getattr(diagnostics, name)(*args, **kwargs)


def diagnostic(name: str, *args, _key: Any = None, **kwargs) -> Any:
    """`src.diagnostics.<name>(*args, **kwargs)`, recalculé seulement si le journal (ou `_key`) change."""
    meta = load_meta()
    return _diagnostic(name, meta["journal_sha256"] if meta else "", _key, args, kwargs)


def data_split(settings: Settings | None = None):
    """Périodes train / validation / test des données chargées (réglages enregistrés)."""
    settings = settings or current_settings()
    return diagnostic("data_split", interim_dir(settings), settings.split)

# ════════════════════════════════════════════════════════════════════════════════════════════════════
# Pages
# ════════════════════════════════════════════════════════════════════════════════════════════════════



def page_home() -> None:
    st.title("Rapprochement automatique paiements / factures")
    st.caption("POC — gain de taux d'automatisation à précision fixée. Chaque étape ne traite que ce que la précédente "
               "n'a pas résolu. Cette page résume le dernier résultat de chaque étape et ce qui reste à lancer.")

    settings = current_settings()
    meta = load_meta(settings)
    out = reports_dir(settings)
    target = settings.evaluation.target_precision
    TODO, OK, RERUN, BELOW = "à lancer", "à jour", "à relancer", "sous la cible"
    BADGES = {TODO: ("gray", ":material/schedule:"), OK: ("green", ":material/check_circle:"),
              RERUN: ("orange", ":material/sync_problem:"), BELOW: ("red", ":material/error:")}

    def _when(path) -> str:
        return datetime.fromtimestamp(path.stat().st_mtime).strftime("%d/%m %H:%M") if path.exists() else ""

    def _journal_ok(sha: str | None) -> bool:
        return meta is None or sha is None or sha == meta["journal_sha256"]

    def config_state():
        return OK, [f"settings.yaml : `{settings_path()}`", f"rules.yaml : `{rules_path()}`"], []

    def ingestion_state():
        cfg = load_schema(schema_path())
        problems = _check_config(cfg, resolve_path(cfg["base_dir"] or "."))
        lines = [f"schema.yaml : {len(problems)} point(s) à compléter pour les données réelles" if problems
                 else "schema.yaml complet"]
        validation = read_json(out / "sources_validation.json")
        status = TODO if problems and dataset() == REAL else OK
        alerts = []
        if validation is not None:
            lines.append(f"validation complète du {validation['date']} : "
                         + ("prête à charger" if validation["prêt_à_charger"] else
                            f"{len(validation['bloquants'])} point(s) bloquant(s)")
                         + f", {validation['anomalies_qualité']} type(s) d'anomalie de qualité")
            if not validation["prêt_à_charger"] and dataset() == REAL:
                status, alerts = BELOW, ["Sources : la validation complète signale des points bloquants."]
        return status, lines, alerts

    def load_state():
        if meta is None:
            return TODO, ["Aucune donnée chargée."], []
        volumes = read_report("load_volumes.csv")
        counts = dict(zip(volumes["table"], volumes["rows"])) if volumes is not None else {}
        issues = read_report("load_issues.csv")
        n_issues = 0 if issues is None else int((issues["count"] > 0).sum())
        lines = [f"{num(counts.get('payment'))} paiements · {num(counts.get('invoice'))} factures · "
                 f"{num(counts.get('imputation'))} imputations", f"{num(meta['journal_events'])} événements · "
                 f"chargé le {_when(interim_dir(settings) / 'journal_meta.json')}"]
        alerts = [f"{n_issues} type(s) d'anomalie au chargement (page Chargement)."] if n_issues else []
        return OK, lines, alerts

    def split_state():
        if meta is None:
            return TODO, [], []
        try:
            split = data_split(settings)
        except Exception as exc:  # découpage impossible avec ces réglages
            return BELOW, [f"Découpage impossible : {exc}"], [f"Découpage impossible : {exc}"]
        lines = [f"{p.name} : {p.start:%d/%m/%Y} → {p.end:%d/%m/%Y} ({p.days} j)" for p in split.periods]
        return OK, lines, list(split.warnings)

    def allocation_state():
        lines, alerts, status = [], [], TODO
        for period in ("validation", "test"):
            payload = read_json(out / f"allocation_{period}.json")
            if payload is None:
                continue
            s, ctx = payload["summary"], payload["context"]
            status = OK if status in (TODO, OK) else status
            lines.append(f"{period} : rappel {pct(s['rappel_premier_passage'], 2)} (cible {pct(s['rappel_cible'], 0)}) · "
                         f"allocation ferme {pct(s['taux_ferme'])} à {pct(s['précision_ferme'], 2)}")
            if not s["cible_atteinte"]:
                status = BELOW
                alerts.append(f"Allocation ({period}) : rappel {pct(s['rappel_premier_passage'], 2)} sous la cible.")
            if not _journal_ok(ctx.get("journal_sha256")) or ctx.get("settings") != settings.allocation.model_dump():
                status = RERUN if status != BELOW else status
                alerts.append(f"Allocation ({period}) : données ou paramètres modifiés depuis la mesure.")
        return status, lines or ["Rappel non mesuré."], alerts

    def _evaluation_lines(matcher: str):
        lines, alerts, status = [], [], TODO
        for period in ("validation", "test"):
            payload = read_json(out / f"evaluation_{matcher}_{period}.json")
            if payload is None:
                continue
            s, ctx = payload["summary"], payload["context"]
            status = OK if status == TODO else status
            low = clopper_pearson_lower(s["auto"], s["auto_corrects"]) if s["auto"] else None
            lines.append(f"{period} : {pct(s['taux_automatisation'])} automatisés à {pct(s['précision'], 2)} "
                         f"(borne basse {pct(low, 2)})")
            if s["auto"] and not s["cible_atteinte"]:
                status = BELOW
                alerts.append(f"{STEPS['rules' if matcher == 'rules' else 'evaluation'].title} ({period}) : précision "
                              f"{pct(s['précision'], 2)} sous la cible {pct(target, 1)}.")
            if not _journal_ok(ctx.get("journal_sha256")):
                status = RERUN if status != BELOW else status
                alerts.append(f"Évaluation {matcher}/{period} calculée sur un autre journal.")
        return status, lines, alerts

    def rules_state():
        status, lines, alerts = _evaluation_lines("rules")
        rules = load_rules(rules_path())
        lines.insert(0, f"Règles v{rules.version} · {len(rules.active())}/{len(rules.rules)} actives")
        return status, lines, alerts

    def ml_state():
        m = read_json(model_dir(settings) / "model.json")
        if m is None:
            return TODO, ["Aucun modèle entraîné."], []
        val = m.get("metrics", {}).get("validation", {})
        oof = m.get("oof", {}).get("schema", "random_folds (avant walk-forward)")
        kinds = m.get("thresholds", {}).get("kinds", {})
        lines = [f"AUC {val.get('auc', float('nan')):.4f} · precision@1 {pct(val.get('precision_at_1'))} · "
                 f"rappel des candidats {pct(m.get('metrics', {}).get('rappel_candidats_validation'))}",
                 f"scores OOF : {oof} · τ_high " + ", ".join(f"{k} {'jamais' if v > 1 else f'{v:.4f}'}"
                                                          for k, v in kinds.items()),
                 f"entraîné le {_when(model_dir(settings) / 'model.json')}"]
        alerts, status = [], OK
        if not _journal_ok(m.get("journal_sha256")):
            status, alerts = RERUN, ["Modèle entraîné sur un autre journal : le réentraîner."]
        elif m.get("settings") != settings.reconcile_ml.model_dump():
            status, alerts = RERUN, ["Paramètres ML modifiés depuis l'entraînement."]
        if m.get("oof", {}).get("repli"):
            alerts.append(f"Walk-forward : {m['oof']['repli']}.")
        return status, lines, alerts

    def evaluation_state():
        status, lines, alerts = _evaluation_lines("pipeline")
        current = settings.evaluation.current_automation_rate
        payload = read_json(out / "evaluation_pipeline_test.json")
        if payload and current is not None:
            gain = 100 * (payload["summary"]["taux_automatisation"] - current)
            lines.append(f"gain sur le taux actuel ({pct(current)}) : {gain:+.1f} points")
        elif payload:
            lines.append("Taux actuel non renseigné (page Évaluation) : gain non mesuré.")
        return status, lines, alerts

    def analysis_state():
        lines, alerts, status = [], [], TODO
        for matcher in ("rules", "pipeline"):
            payload = read_json(out / f"analysis_{matcher}_test.json")
            if payload is None:
                continue
            status = OK
            s = payload["summary"]
            errors = ", ".join(f"{v} {k}" for k, v in list(s["erreurs_par_type"].items())[:2])
            top = max(((k, v) for k, v in s["résiduel_par_raison"].items() if k != "sans imputation réelle"),
                      key=lambda kv: kv[1], default=None)
            lines.append(f"{matcher} / test : {s['auto fausse']} auto fausses ({errors or '—'})"
                         + (f" · premier blocage du résiduel : {top[0]} ({num(top[1])})" if top else ""))
        return status, lines or ["Aucune analyse (calculée à la fin du backtest)."], alerts

    def runtime_state():
        path = runtime_dir(settings) / "memory.csv"
        if not path.exists():
            return TODO, ["Aucun suivi enregistré."], []
        return OK, [f"Dernier relevé le {_when(path)}"], []

    STATES = {"config": config_state, "ingestion": ingestion_state, "load": load_state, "split": split_state, "allocation": allocation_state, "rules": rules_state,
              "ml": ml_state, "evaluation": evaluation_state, "analysis": analysis_state, "runtime": runtime_state}

    # --- En-tête ---------------------------------------------------------------------------------------------
    cols = st.columns(4)
    cols[0].metric("Jeu de données", dataset())
    cols[1].metric("Événements du journal", num(meta["journal_events"]) if meta else "—")
    cols[2].metric("Normalisation", f"v{meta['normalization_version']}" if meta else "—")
    cols[3].metric("Précision cible", pct(target, 1))

    results = {key: STATES[key]() for key in STEPS if key in STATES}
    alerts = [a for _, _, found in results.values() for a in found]
    pending = [STEPS[k].title for k, (status, _, _) in results.items() if status == TODO and k != "runtime"]
    if alerts:
        with st.container(border=True):
            st.markdown(f"**{len(alerts)} point(s) d'attention**")
            for a in alerts:
                st.markdown(f"- {a}")
    if pending:
        st.info(f"Prochaine étape à lancer : **{pending[0]}**.", icon=":material/arrow_forward:")

    # --- Étapes ----------------------------------------------------------------------------------------------
    st.subheader("Étapes")
    for key, step in STEPS.items():
        status, lines, _ = results.get(key, (OK, [], []))
        color, icon = BADGES[status]
        with st.container(border=True):
            left, mid, right = st.columns([2, 4, 1], vertical_alignment="center")
            left.markdown(f"**{step.title}**  \n{step.summary}")
            mid.markdown("  \n".join(lines) if lines else "—")
            right.badge(status, icon=icon, color=color)


def page_config() -> None:
    step_header("config")
    st.caption("Le mapping des sources réelles (schema.yaml) se fait dans la page « Sources et ingestion ».")
    tab_general, tab_files, tab_env = st.tabs(["Paramètres généraux", "Fichiers de configuration", "Environnement"])

    # --- Paramètres généraux -----------------------------------------------------------------------
    with tab_general:
        settings = current_settings()
        st.subheader("Chemins")
        paths = render_model(settings.paths, "paths")
        save_section(settings, "paths", paths, "paths")
        st.divider()
        st.subheader("Chargement")
        load = render_model(settings.load, "load")
        save_section(settings, "load", load, "load")

    # --- Fichiers de configuration ------------------------------------------------------------------
    def _flatten(value, prefix: str = "") -> dict:
        if isinstance(value, dict):
            return {k: v for key, item in value.items() for k, v in _flatten(item, f"{prefix}{key}.").items()}
        return {prefix.rstrip("."): value}

    with tab_files:
        st.caption("Toute la configuration de la pipeline tient dans trois fichiers, édités par les pages de l'interface, "
                   "le notebook ou à la main. Les pages Découpage, Allocation, Règles, ML et Évaluation éditent chacune "
                   "leur section.")
        current = current_settings()
        rules = load_rules(rules_path())
        diff = [{"paramètre": k, "valeur": scalar(v), "défaut": scalar(d)}
                for (k, v), d in zip(_flatten(current.model_dump()).items(), _flatten(Settings().model_dump()).values())
                if v != d]
        defaults = {r.id: r for r in RulesConfig().rules}
        for r in rules.rules:
            base_rule = defaults.get(r.id)
            for k, v in _flatten({"enabled": r.enabled, "priority": r.priority, **r.params}).items():
                d = None if base_rule is None else _flatten(
                    {"enabled": base_rule.enabled, "priority": base_rule.priority, **base_rule.params}).get(k)
                if v != d:
                    diff.append({"paramètre": f"rules.{r.id}.{k}", "valeur": scalar(v), "défaut": scalar(d)})
        if rules.min_precision != RulesConfig().min_precision:
            diff.append({"paramètre": "rules.min_precision", "valeur": scalar(rules.min_precision),
                         "défaut": scalar(RulesConfig().min_precision)})
        st.markdown(f"**Réglages modifiés par rapport aux valeurs par défaut** ({len(diff)}) · règles v{rules.version}")
        if diff:
            st.dataframe(pd.DataFrame(diff), hide_index=True, width="stretch")
        else:
            st.caption("Aucun : tous les réglages sont à leur valeur par défaut.")

        for label, path, model in [("settings.yaml", settings_path(), Settings), ("rules.yaml", rules_path(), RulesConfig),
                                   ("schema.yaml", schema_path(), None)]:
            with st.expander(f"{label} — `{path}`", icon=":material/description:"):
                text = path.read_text(encoding="utf-8") if path.exists() else ""
                st.code(text or "# fichier absent : valeurs par défaut", language="yaml", height=360)
                c = st.columns([1, 2], vertical_alignment="bottom")
                c[0].download_button("Télécharger", text, file_name=label, mime="text/yaml", icon=":material/download:",
                                     key=f"files.{label}.download")
                upload = c[1].file_uploader(f"Remplacer par un autre {label}", type=["yaml", "yml"],
                                            key=f"files.{label}.upload")
                if upload is not None and st.button(f"Valider et remplacer {label}", key=f"files.{label}.replace",
                                                    icon=":material/upload:"):
                    content = upload.getvalue().decode("utf-8")
                    try:
                        data = yaml.safe_load(content) or {}
                        if model is not None:
                            model.model_validate(data)
                    except ValidationError as exc:
                        st.error("Fichier refusé :\n\n" + "\n".join(f"- {m}" for m in validation_errors(exc)))
                    except yaml.YAMLError as exc:
                        st.error(f"YAML invalide : {exc}")
                    else:
                        path.write_text(content, encoding="utf-8")
                        st.toast(f"{label} remplacé", icon=":material/check:")
                        st.rerun()

    # --- Environnement ----------------------------------------------------------------------------------
    with tab_env:
        import platform
        import sys
        from importlib.metadata import PackageNotFoundError, version

        pod = memory.pod_memory()
        c = st.columns(4)
        c[0].metric("CPU utilisables", memory.available_cpus(), help="Quota du conteneur s'il existe, sinon la machine.")
        c[1].metric("Limite mémoire du pod", f"{pod['limit'] / 2 ** 30:.1f} Go" if pod["limit"] else "—")
        c[2].metric("Mémoire du processus", f"{memory.process_rss() / 2 ** 30:.2f} Go")
        pyspy = find_pyspy()
        c[3].metric("py-spy", "installé" if pyspy else "absent", help=pyspy or "Profil par fonction : repli sur le fil "
                    "interne (rien à installer).")
        versions = []
        for package in ("pandas", "numpy", "pyarrow", "scipy", "scikit-learn", "lightgbm", "pydantic", "streamlit",
                        "altair", "pyyaml"):
            try:
                versions.append({"bibliothèque": package, "version": version(package)})
            except PackageNotFoundError:
                versions.append({"bibliothèque": package, "version": "absente"})
        st.dataframe(pd.DataFrame([{"bibliothèque": "python", "version": platform.python_version()}, *versions]),
                     hide_index=True, width="content")
        pandas_major = int(pd.__version__.split(".")[0])
        if pandas_major < 3:
            st.warning(f"pandas {pd.__version__} : la normalisation attend pandas ≥ 3 (chaînes Arrow, expression "
                       "régulière \\p{M}).", icon=":material/warning:")
        st.caption(f"Interpréteur `{sys.executable}` · {platform.platform()}")


def page_ingestion() -> None:
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


def page_load() -> None:
    step_header("load")
    settings = current_settings()
    ds = dataset()

    # --- Lancement -----------------------------------------------------------------------------
    st.subheader("Lancer le chargement")
    kwargs = {}
    if ds != SYNTHETIC:
        st.caption(f"Sources décrites par `{schema_path()}` (page Configuration).")

    if st.button("Lancer l'étape 1", type="primary", icon=":material/play_arrow:", key="run_load"):
        code, lines = run_task("load", "Étape 1 — chargement", **kwargs)
        if code != 0:
            st.error("Échec du chargement :\n\n" + "\n".join(lines[-8:]))
        else:
            st.rerun()

    # --- Résultats -------------------------------------------------------------------------------
    meta = load_meta(settings)
    if meta is None:
        st.info(f"Aucun chargement pour le jeu **{ds}**.")
        st.stop()

    st.subheader("Dernier chargement")
    volumes = read_report("load_volumes.csv")
    counts = dict(zip(volumes["table"], volumes["rows"])) if volumes is not None else {}
    fmt = lambda n: f"{int(n):,}".replace(",", " ")  # noqa: E731
    m = st.columns(5)
    m[0].metric("Paiements", fmt(counts.get("payment", 0)))
    m[1].metric("Factures", fmt(counts.get("invoice", 0)))
    m[2].metric("Imputations", fmt(counts.get("imputation", 0)))
    m[3].metric("Événements", fmt(meta["journal_events"]))
    issues = read_report("load_issues.csv")
    n_issues = 0 if issues is None else len(issues[issues["count"] > 0])
    m[4].metric("Anomalies", n_issues)
    st.caption(f"Source `{meta['source']}` · normalisation v{meta['normalization_version']} · "
               f"journal `{meta['journal_sha256'][:16]}…`")

    tabs = st.tabs(["Anomalies", "Champs manquants", "Plages de dates", "Journal", "Volumétrie", "Temps d'exécution",
                    "Tables"])
    with tabs[0]:
        if issues is None or issues.empty or n_issues == 0:
            st.success("Aucune anomalie détectée.", icon=":material/check_circle:")
        else:
            st.dataframe(issues[issues["count"] > 0], hide_index=True, width="stretch")
    with tabs[1]:
        missing = read_report("load_missing_fields.csv")
        if missing is not None:
            only = st.toggle("Seulement les champs non mappés ou incomplets", value=True, key="load.only_missing")
            view = missing[(~missing["mapped"]) | (missing["null"] > 0)] if only else missing
            st.dataframe(view, hide_index=True, width="stretch",
                         column_config={"null_pct": st.column_config.ProgressColumn(
                             "null %", min_value=0, max_value=100, format="%.1f %%")})
    with tabs[2]:
        dates = read_report("load_date_ranges.csv")
        if dates is not None:
            st.dataframe(dates, hide_index=True, width="stretch")
    with tabs[3]:
        events = read_report("load_events.csv")
        if events is not None:
            st.dataframe(events, hide_index=True, width="stretch")
    with tabs[4]:
        st.caption("Ce qui grossit avec la longueur de l'historique. Le restant dû n'est reconstruit que par les "
                   "imputations : une facture jamais soldée reste ouverte et pèse sur chaque jour du rejeu (règle R3, "
                   "allocation par montant, candidats ML).")
        if st.toggle("Calculer le stock de factures ouvertes (quelques secondes par million de factures, mis en cache)",
                     key="load.volumetry"):
            try:
                stock = diagnostic("open_invoice_stock", interim_dir(settings), data_split(settings))
            except SplitError as exc:
                st.error(f"Découpage impossible avec les réglages actuels : {exc}")
                st.stop()
            total, debtors = stock["total"], stock["debtors"]
            v = st.columns(4)
            v[0].metric("Factures", num(total["factures"]))
            v[1].metric("Ouvertes au début du test", num(total["ouvertes au début du test"]),
                        help=f"Début du test : {total['début du test']:%d/%m/%Y}")
            v[2].metric("Débiteurs avec factures ouvertes", num(debtors["débiteurs avec factures ouvertes"]))
            v[3].metric("Factures ouvertes du plus gros débiteur", num(debtors["maximum"]),
                        help=f"Médiane {debtors['factures ouvertes par débiteur (médiane)']:.0f}, P99 {debtors['P99']:.0f} ; "
                             f"les 10 plus gros portent {debtors['part portée par les 10 plus gros']:.0%} du stock.")
            curve = stock["curve"].melt(id_vars="mois", value_vars=["ouvertes", "créées", "soldées"], var_name="série",
                                        value_name="factures")
            st.altair_chart(alt.Chart(curve).mark_line(point=True).encode(
                x=alt.X("mois:T", title=None, axis=alt.Axis(format="%m/%Y")), y=alt.Y("factures:Q", title="Factures"),
                color=alt.Color("série:N", title=None, legend=alt.Legend(orient="top")),
                tooltip=[alt.Tooltip("mois:T", format="%m/%Y"), "série:N", alt.Tooltip("factures:Q", format=",")],
            ).properties(height=260, title="Stock en début de mois (créées et soldées : cumuls)"), width="stretch")
            st.markdown("**Âge du stock ouvert au début du test** (retard par rapport à l'échéance)")
            st.dataframe(stock["age"], hide_index=True, width="stretch", column_config={
                "factures": st.column_config.NumberColumn(format="localized"),
                "part": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent"),
                "jamais imputées": st.column_config.NumberColumn(format="localized")})
            st.caption("Les factures très en retard et jamais imputées sont probablement soldées hors du périmètre "
                       "(avoir, passage en perte, autre canal) : à confirmer avec le métier avant de les exclure.")
    with tabs[5]:
        timings = meta.get("timings_s", {})
        st.dataframe(pd.DataFrame({"étape": list(timings), "secondes": list(timings.values())}),
                     hide_index=True, width="content")
    with tabs[6]:
        if volumes is not None:
            st.dataframe(volumes, hide_index=True, width="content")

    # --- Données normalisées ------------------------------------------------------------------------
    st.subheader("Libellés normalisés")
    pay_path = interim_dir(settings) / "payment.parquet"
    if pay_path.exists():
        pf = pq.ParquetFile(pay_path)
        sample = pf.read_row_group(0, columns=["payment_id", "label", "label_norm", "label_numbers"]).to_pandas()
        sample = sample.sample(min(200, len(sample)), random_state=0).sort_values("payment_id")
        sample["label_numbers"] = sample["label_numbers"].map(lambda v: " · ".join(v))
        st.caption(f"Échantillon de {len(sample)} paiements sur {pf.metadata.num_rows:,}.".replace(",", " "))
        st.dataframe(sample, hide_index=True, width="stretch", height=280)

    st.subheader("Testeur de normalisation")
    st.caption("Vérifie si une référence de facture est retrouvée dans un libellé, avec la normalisation courante.")
    c = st.columns(2)
    label = c[0].text_input("Libellé bancaire", "VIR SEPA DUPONT SARL REGLT 12345 FA", key="load.test_label")
    ref = c[1].text_input("Référence de facture", "FA0012345", key="load.test_ref")
    numbers = label_numbers(normalize_text(label).split())
    keys = reference_keys(ref)
    common = sorted(set(numbers) & set(keys))
    c[0].markdown(f"Normalisé : `{normalize_text(label)}`  \nClés : " + " ".join(f"`{k}`" for k in numbers))
    c[1].markdown("Clés : " + " ".join(f"`{k}`" for k in keys))
    if common:
        st.success(f"Référence retrouvée via : {', '.join(common)}", icon=":material/check_circle:")
    else:
        st.warning("Référence non retrouvée dans le libellé.", icon=":material/search_off:")


def page_split() -> None:
    step_header("split")
    settings = current_settings()

    # Palette de référence : 3 premières teintes catégorielles, gris neutres pour purge / hors période.
    DOMAIN = [*PERIODS, PURGE, OUTSIDE]
    COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#8a8983", "#c9c8c1"]
    LABELS = {"train": "Entraînement", "validation": "Validation", "test": "Test (backtest)",
              PURGE: "Purge", OUTSIDE: "Hors période"}

    left, right = st.columns([1, 2], gap="large")
    with left:
        st.subheader("Paramètres")
        values = render_model(settings.split, "split")
        save_section(settings, "split", values, "split")

    with right:
        st.subheader("Aperçu sur les données chargées")
        meta = require_loaded_data()
        if meta is None:
            st.stop()
        try:
            cfg = SplitSettings.model_validate(values)
        except ValidationError as exc:
            st.error("\n".join(f"- {m}" for m in validation_errors(exc)))
            st.stop()

        base = interim_dir(settings)
        pay = read_parquet(base / "payment.parquet", ["value_date", "booking_date"])
        imp = read_parquet(base / "imputation.parquet", ["updated_at"])
        # Jour de connaissance du paiement, comme dans le journal.
        pay_day = pay["booking_date"].fillna(pay["value_date"]).dt.normalize()
        daily = pay_day.value_counts().sort_index().rename_axis("jour").reset_index(name="paiements")

        try:
            split = compute_split(cfg, daily["jour"].min().date(), daily["jour"].max().date())
        except SplitError as exc:
            st.error(f"Découpage impossible : {exc}", icon=":material/error:")
            st.stop()
        for w in split.warnings:
            st.warning(w, icon=":material/warning:")

        daily["période"] = assign_period(daily["jour"], split)
        daily["libellé"] = daily["période"].map(LABELS)
        imp_period = assign_period(imp["updated_at"].dt.normalize(), split)
        rows = []
        for p in split.periods:
            rows.append({
                "période": LABELS[p.name], "début": p.start, "fin": p.end, "jours": p.days,
                "paiements": int(daily.loc[daily["période"] == p.name, "paiements"].sum()),
                "imputations": int((imp_period == p.name).sum()),
            })
        table = pd.DataFrame(rows)
        table["part des paiements"] = table["paiements"] / len(pay_day)
        st.dataframe(table, hide_index=True, width="stretch", column_config={
            "début": st.column_config.DateColumn(format="DD/MM/YYYY"),
            "fin": st.column_config.DateColumn(format="DD/MM/YYYY"),
            "paiements": st.column_config.NumberColumn(format="localized"),
            "imputations": st.column_config.NumberColumn(format="localized"),
            "part des paiements": st.column_config.NumberColumn(format="percent"),
        })
        excluded = int(daily.loc[daily["période"].isin([PURGE, OUTSIDE]), "paiements"].sum())
        st.caption(f"{excluded:,} paiements tombent en purge ou hors période.".replace(",", " "))

        chart = alt.Chart(daily).mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
            x=alt.X("jour:T", title=None, axis=alt.Axis(format="%m/%Y", labelAngle=0)),
            y=alt.Y("paiements:Q", title="Paiements par jour"),
            color=alt.Color("libellé:N", title=None,
                            scale=alt.Scale(domain=[LABELS[d] for d in DOMAIN], range=COLORS),
                            legend=alt.Legend(orient="top")),
            tooltip=[alt.Tooltip("jour:T", title="Jour", format="%d/%m/%Y"),
                     alt.Tooltip("libellé:N", title="Période"),
                     alt.Tooltip("paiements:Q", title="Paiements", format=",")],
        ).properties(height=320)
        st.altair_chart(chart, width="stretch")
        st.caption("L'aperçu suit les valeurs saisies, même non enregistrées. Les paiements sont datés par "
                   "leur date de connaissance (booking_date si disponible, sinon value_date).")

    # --- Purge, rétention, lot quotidien ----------------------------------------------------------------
    st.divider()
    st.subheader("Purge, rétention et lot quotidien")
    st.caption("La purge doit couvrir le délai entre l'arrivée d'un paiement et son imputation : sinon un paiement de fin "
               "d'entraînement a son label dans la validation (fuite F3). La rétention borne le reliquat : un paiement "
               "imputé plus tard sort du lot sans décision. Le lot quotidien (paiements du jour + reliquat) fixe le coût "
               "de chaque jour du rejeu.")
    if st.toggle("Calculer (quelques secondes, mis en cache)", key="split.diagnostics"):
        delays = diagnostic("imputation_delays", base, cfg.purge_days, cfg.retention_days)
        d = delays["describe"]
        m = st.columns(5)
        m[0].metric("Délai médian", f"{d['50%']:.1f} j")
        m[1].metric("P95", f"{d['95%']:.1f} j")
        m[2].metric(f"Imputés après la purge ({cfg.purge_days} j)", pct(delays["au-delà de la purge"]),
                    help="Part des paiements imputés dont le délai dépasse la purge : leur label peut franchir la frontière "
                         "entre deux périodes.")
        m[3].metric(f"Imputés après la rétention ({cfg.retention_days} j)", pct(delays["au-delà de la rétention"]),
                    help="Sortis du reliquat avant leur imputation : jamais proposés.")
        m[4].metric("Jamais imputés", pct(delays["jamais imputés"]))
        if delays["au-delà de la purge"] > 0.01:
            st.warning(f"{pct(delays['au-delà de la purge'])} des paiements sont imputés après la purge : l'augmenter "
                       f"(P95 = {d['95%']:.0f} j, P99 = {d['99%']:.0f} j).", icon=":material/warning:")
        batches = diagnostic("pending_batches", base, split, cfg.retention_days)
        if len(batches):
            long_b = batches.melt(id_vars=["période", "jour"], value_vars=["paiements du jour", "lot (majorant)"],
                                  var_name="mesure", value_name="paiements")
            st.altair_chart(alt.Chart(long_b).mark_line(point=True).encode(
                x=alt.X("jour:T", title=None, axis=alt.Axis(format="%d/%m/%Y")), y=alt.Y("paiements:Q"),
                color=alt.Color("mesure:N", title=None, legend=alt.Legend(orient="top")),
                tooltip=[alt.Tooltip("jour:T", format="%d/%m/%Y"), "période:N", "mesure:N",
                         alt.Tooltip("paiements:Q", format=",")],
            ).properties(height=240, title="Lot quotidien sur la validation et le test (un jour par semaine)"),
                width="stretch")
            st.caption("Majorant : le reliquat compte les paiements non encore imputés, sans les auto-validations du "
                       "moteur, qui le vident aussi.")

    # --- Rejeu quotidien ------------------------------------------------------------------------------
    st.divider()
    st.subheader("Rejeu quotidien")
    st.caption("Chaque jour D, l'état est figé à la veille (événements < D) et le lot contient les paiements du jour "
               "plus le reliquat non résolu, dans la limite de la rétention. Un paiement sort du reliquat quand une "
               "imputation réelle est prononcée, quand le rapprocheur l'auto-valide, ou à expiration.")

    period = st.segmented_control("Période", options=[*PERIODS, "all"], default="test", key="replay.period",
                                  format_func=lambda p: {"all": "Tout l'historique", **LABELS}[p]) or "test"
    if st.button("Rejouer avec le rapprocheur vide", icon=":material/replay:", key="replay.run"):
        code, lines = run_task("replay", f"Rejeu — {period}", period=period, matcher="null")
        if code != 0:
            st.error("\n".join(lines[-5:]))

    summary_path = reports_dir(settings) / f"replay_null_{period}.json"
    daily_path = reports_dir(settings) / f"replay_null_{period}_daily.csv"
    if not summary_path.exists():
        st.info("Aucun rejeu pour cette période.")
        st.stop()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["journal_sha256"] != meta["journal_sha256"]:
        st.warning("Ce rejeu a été calculé sur un autre journal : le relancer.", icon=":material/warning:")
    fmt = lambda n: f"{int(n):,}".replace(",", "\u202f")  # noqa: E731
    m = st.columns(5)
    m[0].metric("Jours", summary["days"])
    m[1].metric("Paiements arrivés", fmt(summary["new_payments"]))
    m[2].metric("Lignes de lot", fmt(summary["batch_rows"]))
    m[3].metric("Sortis par expiration", fmt(summary["expired"]))
    m[4].metric("Durée du rejeu", f"{summary['timings_s']['rejeu']} s")
    timings = " · ".join(f"{k} {v} s" for k, v in summary["timings_s"].items())
    st.caption(f"Du {summary['start']} au {summary['end']} · rapprocheur « {summary['matcher']} » · {timings}")

    daily = pd.read_csv(daily_path, parse_dates=["day"])
    long = daily.melt(id_vars="day", value_vars=["new", "carried"], var_name="origine", value_name="paiements")
    long["origine"] = long["origine"].map({"new": "Arrivés le jour même", "carried": "Reliquat"})
    bars = alt.Chart(long).mark_bar().encode(
        x=alt.X("day:T", title=None, axis=alt.Axis(format="%d/%m", labelAngle=0)),
        y=alt.Y("paiements:Q", title="Paiements dans le lot", stack=True),
        color=alt.Color("origine:N", title=None, scale=alt.Scale(domain=["Arrivés le jour même", "Reliquat"],
                                                                   range=COLORS[:2]),
                        legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("day:T", title="Jour", format="%d/%m/%Y"), alt.Tooltip("origine:N", title="Origine"),
                 alt.Tooltip("paiements:Q", title="Paiements", format=",")],
    ).properties(height=280)
    st.altair_chart(bars, width="stretch")
    with st.expander("Statistiques quotidiennes", icon=":material/table:"):
        st.dataframe(daily, hide_index=True, width="stretch")


def page_allocation() -> None:
    step_header("allocation")
    settings = current_settings()
    alloc = settings.allocation

    # --- Paramètres -------------------------------------------------------------------------------------
    with st.expander("Signaux et paramètres", icon=":material/tune:", expanded=False):
        st.caption("Du plus fort au plus faible. Un débiteur manqué ici est perdu pour la suite : "
                   "désactiver un signal se mesure sur le rappel d'allocation.")
        signals = {}
        titles = {"client_file": "1 · Client file", "reference": "2 · Référence", "iban": "3 · IBAN", "name": "4 · Nom",
                  "amount": "5 · Montant"}
        cols = st.columns(len(titles))
        for col, (name, title) in zip(cols, titles.items()):
            with col.container(border=True):
                st.markdown(f"**{title}**")
                signals[name] = render_model(getattr(alloc.signals, name), f"allocation.signals.{name}")
        values = {"signals": signals, **render_model(alloc, "allocation", skip=("signals",))}
        if not any(s["enabled"] for s in signals.values()):
            st.error("Aucun signal actif : aucun paiement ne pourra être alloué.", icon=":material/error:")
        save_section(settings, "allocation", values, "allocation")
        st.caption("Le signal montant est une extension du brief (clé « montant exact » de la spec) : il départage "
                   "les paiements sans référence ni IBAN exploitable.")

    # --- Client files -------------------------------------------------------------------------------------
    with st.expander("Diagnostic des client files", icon=":material/description:"):
        st.caption("Un client file n'est rattaché à un paiement que si son montant total égale exactement le montant du "
                   "paiement et qu'au moins un indice le corrobore (date à ± tolérance, IBAN, référence de virement dans "
                   "le libellé), sans ambiguïté. Le diagnostic retrouve le vrai paiement de chaque fichier par les "
                   "factures citées et les imputations réelles, puis mesure chaque condition.")
        if st.toggle("Calculer (quelques secondes, mis en cache)", key="allocation.client_files"):
            cf = diagnostic("client_file_conditions", interim_dir(settings), alloc.signals.client_file.date_tolerance_days)
            if cf is None:
                st.info("Aucun client file chargé : remplir client_file et client_file_line dans schema.yaml "
                        "(page Configuration).")
            else:
                m = st.columns(3)
                m[0].metric("Client files", num(cf["fichiers"]))
                m[1].metric("Lignes", num(cf["lignes"]))
                m[2].metric("Vrai paiement retrouvé", num(cf["vrai paiement retrouvé"]),
                            help="Fichiers dont une ligne cite une facture unique imputée par un paiement réel.")
                left, right = st.columns(2, gap="large")
                ratio = st.column_config.ProgressColumn("part", min_value=0, max_value=1, format="percent")
                with left:
                    st.markdown("**Conditions du rattachement** (fichiers dont le vrai paiement est connu)")
                    st.dataframe(cf["conditions"].rename("part").rename_axis("condition").reset_index(), hide_index=True,
                                 width="stretch", column_config={"part": ratio})
                with right:
                    st.markdown("**Remplissage des champs utilisés**")
                    st.dataframe(cf["remplissage"].rename("part").rename_axis("champ").reset_index(), hide_index=True,
                                 width="stretch", column_config={"part": ratio})
                    st.markdown("**Références des lignes** (égalité exacte après normalisation)")
                    st.dataframe(cf["références des lignes"].rename("part").rename_axis("la référence désigne")
                                 .reset_index(), hide_index=True, width="stretch", column_config={"part": ratio})
                st.markdown("**Délai de réception du fichier − date de valeur (jours)** et **écart montant total − "
                            "paiement quand il n'est pas nul (€)**")
                st.dataframe(pd.DataFrame({"délai (j)": cf["délai de réception (j)"],
                                           "écart (€)": cf["écart total − paiement (€)"]}).T.round(2), width="stretch")

    # --- Mesure ---------------------------------------------------------------------------------------------
    st.subheader("Mesure du rappel")
    c = st.columns([3, 1], vertical_alignment="bottom")
    period = c[0].segmented_control("Période", options=[*PERIODS, "all"], default="validation",
                                    key="allocation.period", format_func=PERIOD_LABELS.get) or "validation"
    if c[1].button("Mesurer l'allocation", type="primary", icon=":material/play_arrow:", key="allocation.run"):
        code, lines = run_task("measure_allocation", f"Allocation — {PERIOD_LABELS[period]}", period=period)
        if code != 0:
            st.error("\n".join(lines[-5:]))

    out = reports_dir(settings)
    tag = f"allocation_{period}"
    if not (out / f"{tag}.json").exists():
        st.info("Aucune mesure pour cette période. Comptez ≈ 4 min pour deux mois à 2 M de paiements.")
        st.stop()
    payload = json.loads((out / f"{tag}.json").read_text(encoding="utf-8"))
    s, ctx = payload["summary"], payload["context"]
    meta = load_meta(settings)
    if meta and ctx["journal_sha256"] != meta["journal_sha256"]:
        st.warning("Mesure calculée sur un autre journal : la relancer.", icon=":material/warning:")
    if ctx["settings"] != alloc.model_dump():
        st.warning("Les paramètres ont changé depuis cette mesure : la relancer pour les évaluer.",
                   icon=":material/sync_problem:")

    pct = lambda v: "—" if v is None else f"{v:.2%}"  # noqa: E731
    recall = s["rappel_premier_passage"]
    if s["cible_atteinte"]:
        st.success(f"Rappel {pct(recall)} ≥ cible {pct(s['rappel_cible'])} : l'étape 4 peut s'appuyer sur l'allocation.",
                   icon=":material/check_circle:")
    else:
        st.error(f"Rappel {pct(recall)} < cible {pct(s['rappel_cible'])} : ne pas avancer à l'étape 4 (brief §9).",
                 icon=":material/block:")
    m = st.columns(5)
    m[0].metric("Rappel (1er passage)", pct(recall), help="Le vrai débiteur figure dans la liste le jour d'arrivée.")
    m[1].metric("Vrai débiteur en tête", pct(s["top1"]))
    m[2].metric("Allocation ferme", pct(s["taux_ferme"]))
    m[3].metric("Précision ferme", pct(s["précision_ferme"]))
    m[4].metric("Sans candidat", pct(s["sans_candidat"]), help="Partent en file analyste.")
    st.caption(f"{PERIOD_LABELS[period]} du {ctx['start']} au {ctx['end']} · "
               f"{s['avec_imputation_réelle']:,} paiements avec imputation réelle · ".replace(",", " ")
               + f"{s['candidats_moyens']:.1f} candidats en moyenne · client file rattaché : {pct(s['avec_client_file'])} · "
               f"rappel au dernier passage {pct(s['rappel_dernier_passage'])} · "
               f"index {ctx['timings_s'].get('construction des index', '—')} s, rejeu {ctx['timings_s'].get('rejeu', '—')} s")

    percent = st.column_config.NumberColumn(format="percent")
    cfg = {k: percent for k in ("rappel", "rappel_dernier_passage", "top1", "taux_ferme", "précision_ferme", "part")}
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Par route IBAN**")
        st.dataframe(pd.read_csv(out / f"{tag}_by_route.csv"), hide_index=True, width="stretch", column_config=cfg)
        st.markdown("**Par statut d'allocation**")
        st.dataframe(pd.read_csv(out / f"{tag}_by_status.csv"), hide_index=True, width="stretch", column_config=cfg)
    with right:
        st.markdown("**Signaux ayant trouvé le vrai débiteur**")
        st.dataframe(pd.read_csv(out / f"{tag}_found_by.csv"), hide_index=True, width="stretch", height=360,
                     column_config=cfg)
    with st.expander("Paiements manqués (échantillon)", icon=":material/search_off:"):
        st.dataframe(pd.read_csv(out / f"{tag}_misses.csv"), hide_index=True, width="stretch")


def page_rules() -> None:
    step_header("rules")
    saved = load_rules(rules_path())

    c = st.columns(3)
    c[0].metric("Version du jeu de règles", saved.version)
    c[1].metric("Règles actives", f"{len(saved.active())} / {len(saved.rules)}")
    min_precision = c[2].number_input("Précision individuelle minimale", value=float(saved.min_precision),
                                      step=0.001, format="%.3f", key="rules.min_precision",
                                      help="Une règle mesurée sous ce seuil est désactivée.")
    st.caption(f"Fichier `{rules_path()}`. Règles appliquées par priorité croissante ; validation seulement si la "
               "solution est unique, sinon le paiement passe à l'étape 5.")

    rules = []
    for rule in sorted(saved.rules, key=lambda r: r.priority):
        k = f"rules.{rule.id}"
        with st.container(border=True):
            head = st.columns([3, 1, 1], vertical_alignment="center")
            enabled = head[0].toggle(f"**{rule.name}** · `{rule.id}`", value=rule.enabled, key=f"{k}.enabled")
            priority = head[1].number_input("Priorité", min_value=1, value=rule.priority, step=1, key=f"{k}.priority")
            head[2].badge("active" if enabled else "désactivée", color="green" if enabled else "gray")
            st.caption(rule.description)
            params = {}
            if rule.params:
                pcols = st.columns(max(len(rule.params), 2))
                for col, (name, value) in zip(pcols, rule.params.items()):
                    if isinstance(value, int):
                        params[name] = int(col.number_input(name, value=value, step=1, key=f"{k}.{name}",
                                                            disabled=not enabled))
                    else:
                        params[name] = float(col.number_input(name, value=float(value), step=0.001, format="%.4f",
                                                              key=f"{k}.{name}", disabled=not enabled))
            rules.append({**rule.model_dump(), "enabled": enabled, "priority": int(priority), "params": params})

    candidate = {"version": saved.version, "min_precision": min_precision, "rules": rules}

    def _comparable(cfg: dict) -> dict:
        return {**cfg, "rules": sorted(cfg["rules"], key=lambda r: r["id"])}

    changed = _comparable(candidate) != _comparable(saved.model_dump())

    if st.button("Enregistrer les règles", type="primary", icon=":material/save:", key="rules.save", disabled=not changed):
        try:
            # Toute modification crée une nouvelle version, tracée dans les décisions.
            updated = RulesConfig.model_validate({**candidate, "version": saved.version + 1})
        except ValidationError as exc:
            st.error("Règles invalides :\n\n" + "\n".join(f"- {m}" for m in validation_errors(exc)))
        else:
            save_rules(updated, rules_path())
            st.toast(f"Règles enregistrées — version {updated.version}", icon=":material/check:")
            st.rerun()
    if changed:
        st.caption(f"Modifications non enregistrées : l'enregistrement créera la version {saved.version + 1}.")

    # --- Mesure ---------------------------------------------------------------------------------------------
    st.subheader("Mesure de la baseline")
    c = st.columns([3, 1], vertical_alignment="bottom")
    period = c[0].segmented_control("Période", options=[*PERIODS, "all"], default="validation", key="rules.period",
                                    format_func=PERIOD_LABELS.get) or "validation"
    if c[1].button("Rejouer et évaluer", type="primary", icon=":material/play_arrow:", key="rules.run"):
        code, lines = run_task("backtest", f"Règles — {PERIOD_LABELS[period]}", period=period, matcher="rules")
        if code != 0:
            st.error("\n".join(lines[-5:]))

    out = reports_dir(current_settings())
    name = f"evaluation_rules_{period}"
    payload = read_json(out / f"{name}.json")
    if payload is None:
        st.info("Aucune mesure pour cette période. Comptez ≈ 5 min pour deux mois à 2 M de paiements.")
        st.stop()
    summary, ctx = payload["summary"], payload["context"]
    stale(ctx.get("journal_sha256"), "Mesure")
    low = clopper_pearson_lower(summary["auto"], summary["auto_corrects"]) if summary["auto"] else None
    m = st.columns(4)
    m[0].metric("Taux d'automatisation", pct(summary["taux_automatisation"], 2),
                help="Part des paiements du périmètre que les règles auto-valident, telles qu'elles tourneraient en "
                     "production.")
    m[1].metric("Précision", pct(summary["précision"], 2), help="Part des auto-validations exactement égales à "
                "l'imputation réelle (groupe de factures complet).")
    m[2].metric("Précision min. (95 %)", pct(low, 2),
                help="Borne basse de Clopper-Pearson : la vraie précision est au-dessus avec 95 % de confiance.")
    current = current_settings().evaluation.current_automation_rate
    m[3].metric("Taux actuel (référence)", pct(current),
                delta=None if current is None else f"{100 * (summary['taux_automatisation'] - current):+.1f} pts")
    st.caption(f"{PERIOD_LABELS[period]} du {ctx['start']} au {ctx['end']} · {num(summary['paiements_périmètre'])} "
               f"paiements · {num(summary['auto'])} auto-validés dont {num(summary['auto_corrects'])} corrects · "
               f"rejeu {ctx['timings_s'].get('rejeu', '—')} s. Plafond a posteriori à précision ≥ "
               f"{summary['précision_cible']:.1%} : {pct(summary['taux_automatisation_à_précision_cible'], 2)} "
               "(propositions triées par score après coup, avec la vérité : un majorant, pas un résultat de production).")
    if summary["auto"] and not summary["cible_atteinte"]:
        st.error(f"Précision {pct(summary['précision'], 2)} sous la cible {pct(summary['précision_cible'], 1)}.",
                 icon=":material/error:")

    cascade = with_precision_bounds(pd.read_csv(out / f"{name}_by_rule.csv")).rename(columns={
        "rule": "rule_id", "auto": "décidés (cascade)", "précision": "précision (cascade)",
        "précision min. 95 %": "précision min. 95 % (cascade)", "taux_automatisation": "couverture (cascade)"})[
        ["rule_id", "décidés (cascade)", "précision (cascade)", "précision min. 95 % (cascade)", "couverture (cascade)"]]
    alone_path = out / f"{name}_by_rule_alone.csv"
    table = pd.read_csv(alone_path).rename(columns={"précision": "précision (seule)", "couverture": "couverture (seule)",
                                                    "propositions": "propositions (seule)"}) \
        if alone_path.exists() else pd.DataFrame({"rule_id": [r.id for r in saved.rules]})
    table = table.merge(cascade, on="rule_id", how="left")
    below = table[table["précision (seule)"].notna() & (table["précision (seule)"] < saved.min_precision)] \
        if "précision (seule)" in table else table.iloc[0:0]
    percent = st.column_config.NumberColumn(format="percent")
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        c: percent for c in ("précision (seule)", "couverture (seule)", "précision (cascade)",
                             "précision min. 95 % (cascade)", "couverture (cascade)")})
    st.caption("« Seule » : chaque règle évaluée indépendamment sur tous les paiements (R4, coûteuse, sur le résiduel "
               "des règles précédentes). « Cascade » : ce que chaque règle décide effectivement, après arbitrage. "
               "« Précision min. » : borne basse à 95 % ; avec peu de décisions, une précision observée de 100 % peut "
               "cacher une précision réelle bien plus basse.")
    if len(below):
        st.warning(f"{len(below)} règle(s) sous le seuil de précision {saved.min_precision:.1%} : "
                   + ", ".join(below["rule_id"]), icon=":material/warning:")
        if st.button("Désactiver ces règles", icon=":material/block:", key="rules.disable_below"):
            updated = saved.model_copy(deep=True)
            for r in updated.rules:
                if r.id in set(below["rule_id"]):
                    r.enabled = False
            updated.version += 1
            save_rules(updated, rules_path())
            st.rerun()
    else:
        st.success(f"Toutes les règles mesurées tiennent la précision minimale {saved.min_precision:.1%}.",
                   icon=":material/check_circle:")

    tabs = st.tabs(["Par mois", "Par type de groupe", "Durées"])
    for tab, kind in zip(tabs[:2], ("by_month", "by_group")):
        with tab:
            path = out / f"{name}_{kind}.csv"
            if path.exists():
                st.dataframe(with_precision_bounds(pd.read_csv(path)), hide_index=True, width="stretch", column_config={
                    "précision": percent, "précision min. 95 %": percent,
                    "taux_automatisation": st.column_config.NumberColumn("taux d'automatisation", format="percent")})
    with tabs[2]:
        timings = ctx.get("timings_s", {})
        st.dataframe(pd.DataFrame({"phase": list(timings), "secondes": list(timings.values())}), hide_index=True,
                     width="content")
        st.caption("Détail par règle et par sous-étape : page Temps de run (suivi activé dans la barre latérale).")


def page_ml() -> None:
    step_header("ml")
    settings = current_settings()
    ml = settings.reconcile_ml
    st.caption("Ne traite que le résiduel de l'étape 4. Chaque brique est mesurée et retirée si elle n'apporte rien.")

    values: dict = {}
    tabs = st.tabs(["Candidats", "Features", "Scoring et entraînement", "Ensembles", "Décision", "LLM sur libellés"])
    with tabs[0]:
        st.caption("Clés de génération des candidats (union). Filtres durs toujours actifs : devise, facture ouverte, "
                   "agreement actif.")
        values["candidates"] = render_model(ml.candidates, "reconcile_ml.candidates")
    with tabs[1]:
        st.caption("Familles de features du modèle de paires (LightGBM binaire).")
        families = {}
        cols = st.columns(2)
        for i, name in enumerate(type(ml.features).model_fields):
            with cols[i % 2]:
                families[name] = field_widget(ml.features, name, "reconcile_ml.features")
        values["features"] = families
    with tabs[2]:
        st.caption("Pass 1 : LightGBM sur les features de la paire. Pass 2 : mêmes features plus la compétition entre "
                   "candidats du paiement (rang, marge au second), calculée sur des scores de pass 1 hors échantillon. "
                   "Schéma de ces scores : walk_forward (chaque mois scoré par un modèle du seul passé, comme en "
                   "production) ou random_folds (version précédente, fuite F4).")
        values.update(render_model(ml, "reconcile_ml", skip=("candidates", "features", "sets", "decision", "llm_labels")))
    with tabs[3]:
        values["sets"] = render_model(ml.sets, "reconcile_ml.sets")
    with tabs[4]:
        st.caption(f"Auto-validation si p ≥ τ_high et marge au second ≥ δ. τ_high est calibré par type de proposition "
                   f"pour une précision de {settings.evaluation.target_precision:.1%} sur la boucle de validation "
                   "(page Évaluation pour la cible). Après un changement ici : « Recalibrer les seuils ».")
        values["decision"] = render_model(ml.decision, "reconcile_ml.decision")
    with tabs[5]:
        values["llm_labels"] = render_model(ml.llm_labels, "reconcile_ml.llm_labels")

    save_section(settings, "reconcile_ml", values, "reconcile_ml")

    # --- Entraînement et backtest -------------------------------------------------------------------------------
    st.subheader("Modèle")
    folder = model_dir(settings)
    trained = (folder / "model.json").exists()
    c = st.columns([1, 1, 1, 1], vertical_alignment="center")
    reuse = c[0].checkbox("Reprendre le jeu déjà construit", key="ml.reuse_dataset", disabled=not trained,
                          help="Saute le rejeu train + validation (la plus grosse part du temps) si les données et les "
                               "paramètres de construction n'ont pas changé ; refusé sinon. Utile après un changement "
                               "d'entraînement seul (schéma OOF, hyperparamètres).")
    if c[1].button("Entraîner (train + validation)", type="primary", icon=":material/model_training:", key="ml.fit"):
        code, lines = run_task("train", "Entraînement", **({"reuse_dataset": True} if reuse else {}))
        if code != 0:
            st.error("\n".join(lines[-5:]))
    if c[2].button("Recalibrer les seuils", icon=":material/tune:", key="ml.calibrate", disabled=not trained,
                   help="Après un changement de la marge δ ou de la précision cible : seuils recalculés sans réentraîner."):
        code, lines = run_task("calibrate", "Recalibration des seuils")
        if code != 0:
            st.error("\n".join(lines[-5:]))
    if c[3].button("Backtest (période de test)", icon=":material/play_arrow:", key="ml.backtest", disabled=not trained):
        code, lines = run_task("backtest", "Backtest règles + ML", period="test", matcher="pipeline")
        if code != 0:
            st.error("\n".join(lines[-5:]))
        else:
            st.success("Backtest terminé : résultats dans la page Évaluation (rapprocheur « Règles + ML »).",
                       icon=":material/check_circle:")

    meta = read_json(folder / "model.json")
    if meta is None:
        st.info("Aucun modèle entraîné pour ce jeu de données. Comptez ≈ 30 min à 2 M de paiements.")
        st.stop()
    stale(meta.get("journal_sha256"), "Modèle")
    if meta.get("settings") != ml.model_dump():
        changed = sorted(k for k, v in ml.model_dump().items() if meta.get("settings", {}).get(k) != v)
        st.warning(f"Paramètres ML modifiés depuis l'entraînement ({', '.join(changed)}) : réentraîner, ou recalibrer "
                   "si seule la décision a changé.", icon=":material/sync_problem:")
    metrics, th = meta.get("metrics", {}), meta.get("thresholds", {})
    val = metrics.get("validation", {})
    m = st.columns(5)
    m[0].metric("AUC (validation)", "—" if val.get("auc") is None else f"{val['auc']:.4f}")
    m[1].metric("Precision@1", pct(val.get("precision_at_1"), 2), help="Le vrai candidat est-il le mieux scoré, "
                "parmi les paiements dont le vrai candidat est dans la liste ?")
    m[2].metric("MRR", "—" if val.get("mrr") is None else f"{val['mrr']:.3f}")
    m[3].metric("Rappel des candidats", pct(metrics.get("rappel_candidats_validation")),
                help="Part des paiements du résiduel dont la vraie imputation est dans les candidats : plafond du ML.")
    m[4].metric("Auto-validations hors ligne", num(metrics.get("auto_validation")),
                help=f"Sur le jeu de validation, une fois : précision {pct(metrics.get('précision_auto_validation'), 2)}. "
                     "La boucle quotidienne (seuils en ligne) est la référence.")
    oof = meta.get("oof", {})
    st.caption(f"Journal `{meta.get('journal_sha256', '')[:16]}…` · featurisation v{meta.get('featurization_version')} · "
               f"règles v{meta.get('rules_version')} · entraînement {meta.get('periods', {}).get('train')} · "
               f"validation {meta.get('periods', {}).get('validation')} · scores OOF : "
               f"{oof.get('schema', 'random_folds (modèle antérieur au walk-forward)')}")

    tabs = st.tabs(["Seuils", "Walk-forward (pass 2)", "Importance des features", "Calibration", "Durées"])
    with tabs[0]:
        kinds, offline = th.get("kinds", {}), th.get("offline", {}).get("kinds", {})
        show = lambda v: "jamais" if v is None or v > 1 else f"{v:.4f}"  # noqa: E731
        st.dataframe(pd.DataFrame({"type": list(kinds), "τ_high en ligne (appliqué)": [show(v) for v in kinds.values()],
                                   "τ_high hors ligne": [show(offline.get(k)) for k in kinds]}),
                     hide_index=True, width="stretch")
        st.caption(f"τ_low = {th.get('tau_low')} · δ = {th.get('min_margin')} · méthode {th.get('method', '—')}. "
                   "En ligne : la validation est rejouée jour par jour et un paiement est auto-validé le premier jour où "
                   "son score franchit τ, comme en production. Hors ligne : une seule proposition par paiement. "
                   "« jamais » : précision cible non atteinte ou moins de 20 propositions.")
        proposals = interim_dir(settings) / "ml" / "validation_daily_proposals.parquet"
        if proposals.exists():
            curve = diagnostic("threshold_curve", proposals, th.get("min_margin", ml.decision.min_margin),
                               applied=kinds, _key=proposals.stat().st_mtime)
            target = settings.evaluation.target_precision
            shown = curve[curve["précision"] >= target - 0.02]
            if len(shown):
                points = curve[curve["appliqué"]]
                base = alt.Chart(shown).encode(
                    x=alt.X("auto-validations:Q", title="Auto-validations sur la validation"),
                    color=alt.Color("type:N", title=None, legend=alt.Legend(orient="top")))
                tooltip = ["type", alt.Tooltip("τ_high:Q", format=".4f"), alt.Tooltip("auto-validations:Q", format=","),
                           alt.Tooltip("précision:Q", format=".2%"), alt.Tooltip("précision min. 95 %:Q", format=".2%")]
                layers = [base.mark_line().encode(y=alt.Y("précision:Q", title="Précision", axis=alt.Axis(format="%"),
                                                          scale=alt.Scale(zero=False)), tooltip=tooltip),
                          base.mark_line(strokeDash=[2, 3]).encode(y="précision min. 95 %:Q", tooltip=tooltip),
                          alt.Chart(pd.DataFrame({"y": [target]})).mark_rule(strokeDash=[4, 4], color="#8a8983")
                          .encode(y="y:Q")]
                if len(points):
                    layers.append(alt.Chart(points).mark_point(size=90, filled=True).encode(
                        x="auto-validations:Q", y="précision:Q", color="type:N", tooltip=tooltip))
                st.altair_chart(alt.layer(*layers).properties(
                    height=300, title="Précision selon le volume auto-validé (trait plein), borne basse 95 % (pointillé), "
                                      "point : seuil appliqué"), width="stretch")
            st.caption("Plus τ_high baisse, plus le volume monte et la précision baisse. Le seuil appliqué est le plus bas "
                       "qui tient la cible sur la validation : avec la variance d'échantillonnage, la précision du test "
                       "peut passer sous la cible (malédiction du gagnant). La borne basse dit quelle marge garder.")
    with tabs[1]:
        if not oof:
            st.info("Modèle entraîné avant le walk-forward : scores OOF par folds aléatoires (fuite F4, "
                    "docs/fuites_ml.md). Réentraîner avec reconcile_ml.training.oof_scheme = walk_forward.",
                    icon=":material/history:")
        elif oof.get("schema") == "random_folds":
            st.info("Schéma random_folds : 3 folds par paiement ; un modèle de fold voit le futur du train (fuite F4).",
                    icon=":material/warning:")
        else:
            if oof.get("repli"):
                st.warning(f"Repli : {oof['repli']}.", icon=":material/warning:")
            months = pd.DataFrame(oof.get("mois", []))
            w = st.columns(3)
            w[0].metric("Modèles intermédiaires", int(months["modele"].sum()) if len(months) else 0)
            w[1].metric("Paires de la pass 2", num(oof.get("paires_pass2")))
            w[2].metric("Paires sans score (exclues)", num(oof.get("paires_sans_score")),
                        help="Premiers mois sans historique suffisant : exclus de l'apprentissage de la pass 2.")
            if len(months):
                st.dataframe(months.rename(columns={"date": "coupure Dₖ", "paires_apprises": "paires apprises",
                                                    "paires_scorees": "paires scorées", "modele": "modèle"}),
                             hide_index=True, width="stretch", column_config={
                                 "paires apprises": st.column_config.NumberColumn(format="localized"),
                                 "positifs": st.column_config.NumberColumn(format="localized"),
                                 "paires scorées": st.column_config.NumberColumn(format="localized")})
            st.caption("À chaque coupure Dₖ, un modèle apprend sur les paiements arrivés avant Dₖ − purge, avec les seules "
                       "imputations connues à Dₖ, puis score le mois suivant. Pas de N mois "
                       "(walk_forward_step_months) : N fois moins de modèles.")
    with tabs[2]:
        imp_path = folder / "feature_importance.csv"
        if imp_path.exists():
            importance = pd.read_csv(imp_path)
            left, right = st.columns([2, 3], gap="large")
            with left:
                st.markdown("**Par famille** (gain, passe 1)")
                st.dataframe(diagnostic("importance_by_family", importance, _key=imp_path.stat().st_mtime),
                             hide_index=True, width="stretch", column_config={
                                 "gain": st.column_config.NumberColumn(format="%.3e"),
                                 "part du gain": st.column_config.ProgressColumn(min_value=0, max_value=1,
                                                                                 format="percent")})
                st.caption("Une famille sans gain est candidate au retrait (page Features), à mesurer sur l'évaluation.")
            with right:
                st.markdown("**Par feature**")
                st.dataframe(importance, hide_index=True, width="stretch", height=420,
                             column_config={"gain": st.column_config.NumberColumn(format="%.3e")})
    with tabs[3]:
        calib = folder / "calibration_validation.csv"
        if calib.exists():
            st.dataframe(pd.read_csv(calib), hide_index=True, width="stretch")
            st.caption("Régression isotonique apprise sur la validation : sert à l'affichage des probabilités. Les seuils "
                       "sont calibrés directement sur la précision de la boucle, pas sur ces probabilités.")
    with tabs[4]:
        timings = meta.get("timings_s", {})
        st.dataframe(pd.DataFrame({"phase": list(timings), "minutes": [v / 60 for v in timings.values()]}),
                     hide_index=True, width="content", column_config={"minutes": st.column_config.NumberColumn(
                         format="%.1f")})
        st.caption("Détail par sous-étape et par fonction : page Temps de run.")


def page_evaluation() -> None:
    step_header("evaluation")
    settings = current_settings()
    ev = settings.evaluation
    out = reports_dir(settings)

    with st.expander("Cible et chiffres de référence", icon=":material/tune:"):
        values = render_model(ev, "evaluation")
        save_section(settings, "evaluation", values, "evaluation")

    # --- Toutes les évaluations disponibles -------------------------------------------------------------------
    rows = []
    for path in sorted(out.glob("evaluation_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        s, ctx = payload["summary"], payload["context"]
        rows.append({"rapprocheur": MATCHER_LABELS.get(ctx.get("matcher"), ctx.get("matcher")),
                     "période": PERIOD_LABELS.get(ctx.get("period"), ctx.get("period")),
                     "du": ctx.get("start"), "au": ctx.get("end"), "paiements": s["paiements_périmètre"],
                     "taux d'automatisation": s["taux_automatisation"], "précision": s["précision"],
                     "précision min. 95 %": clopper_pearson_lower(s["auto"], s["auto_corrects"]) if s["auto"] else None,
                     "calculée le": datetime.fromtimestamp(path.stat().st_mtime).strftime("%d/%m %H:%M")})
    if rows:
        with st.expander(f"Toutes les évaluations disponibles ({len(rows)})", icon=":material/table_chart:",
                         expanded=False):
            percent = st.column_config.NumberColumn(format="percent")
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
                "paiements": st.column_config.NumberColumn(format="localized"), "taux d'automatisation": percent,
                "précision": percent, "précision min. 95 %": percent})

    # --- Lancement --------------------------------------------------------------------------------
    c = st.columns([2, 2, 1], vertical_alignment="bottom")
    period = c[0].segmented_control("Période", options=[*PERIODS, "all"], default="test", key="evaluation.period",
                                    format_func=PERIOD_LABELS.get) or "test"
    matcher = c[1].selectbox("Rapprocheur", list(MATCHER_LABELS), format_func=MATCHER_LABELS.get,
                             key="evaluation.matcher", help="« Règles + ML » exige un modèle entraîné (page 5).")
    if c[2].button("Rejouer et évaluer", type="primary", icon=":material/play_arrow:", key="evaluation.run"):
        code, lines = run_task("backtest", f"Rejeu et évaluation — {PERIOD_LABELS[period]}", period=period,
                               matcher=matcher)
        if code != 0:
            st.error("\n".join(lines[-5:]))

    name = f"evaluation_{matcher}_{period}"
    payload = read_json(out / f"{name}.json")
    if payload is None:
        st.info("Aucune évaluation pour cette période et ce rapprocheur.")
        st.stop()
    s, ctx = payload["summary"], payload["context"]
    stale(ctx.get("journal_sha256"), "Évaluation")

    def table(kind: str) -> pd.DataFrame:
        return pd.read_csv(out / f"{name}_{kind}.csv")

    percent = st.column_config.NumberColumn(format="percent")
    rate = st.column_config.NumberColumn("taux d'automatisation", format="percent")
    low = clopper_pearson_lower(s["auto"], s["auto_corrects"]) if s["auto"] else None

    # --- Indicateurs --------------------------------------------------------------------------------
    st.subheader(f"Taux d'automatisation, précision cible {s['précision_cible']:.1%}")
    m = st.columns(5)
    m[0].metric("Taux d'automatisation effectif", pct(s["taux_automatisation"]),
                help="Part des paiements du périmètre auto-validés par le moteur, seuils fixés sur la validation : ce "
                     "que donnerait la production.")
    m[1].metric("Précision des auto-validations", pct(s["précision"], 2))
    m[2].metric("Précision min. (95 %)", pct(low, 2),
                help="Borne basse de Clopper-Pearson : la vraie précision est au-dessus avec 95 % de confiance.")
    m[3].metric("Paiements du périmètre", num(s["paiements_périmètre"]))
    m[4].metric("Plafond a posteriori", pct(s["taux_automatisation_à_précision_cible"]),
                help="Taux atteignable si l'on triait après coup toutes les propositions par score en connaissant la "
                     "vérité du test pour placer le seuil : un majorant pour situer la marge, pas un résultat.")
    st.caption(f"{PERIOD_LABELS[period]} du {ctx['start']} au {ctx['end']} · {MATCHER_LABELS.get(matcher, matcher)} · "
               f"{num(s['auto'])} auto-validés dont {num(s['auto_corrects'])} corrects · {num(s['revue'])} en revue · "
               f"{num(s['sans_imputation_réelle'])} paiements sans imputation réelle · paires : précision "
               f"{pct(s.get('précision_paires'), 2)}, rappel {pct(s.get('rappel_paires'))}.")
    if s["auto"] and not s["cible_atteinte"]:
        st.error(f"Précision {pct(s['précision'], 2)} sous la cible {pct(s['précision_cible'])}.", icon=":material/error:")
    elif low is not None and low < s["précision_cible"]:
        st.info(f"Précision observée au-dessus de la cible, mais borne basse à 95 % ({pct(low, 2)}) en dessous : le "
                "volume ne suffit pas encore à garantir la cible.", icon=":material/info:")

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Tableau en cascade**")
        st.dataframe(table("cascade"), hide_index=True, width="stretch", column_config={
            "taux_automatisation": rate, "précision": percent,
            "gain_points": st.column_config.NumberColumn("gain (points)", format="%.1f")})
        if ev.current_automation_rate is None:
            st.caption("Renseigner le taux actuel (Cible et chiffres de référence) pour mesurer le gain.")
    with right:
        st.markdown("**Par type de groupe**")
        st.dataframe(with_precision_bounds(table("by_group")), hide_index=True, width="stretch", column_config={
            "group_type": "type", "paiements": st.column_config.NumberColumn(format="localized"),
            "précision": percent, "précision min. 95 %": percent, "taux_automatisation": rate})

    curve = table("curve")
    st.markdown("**Courbe automatisation / précision**")
    if curve.empty:
        st.info("Aucune proposition : la courbe apparaîtra avec un rapprocheur qui décide.", icon=":material/info:")
    else:
        line = alt.Chart(curve).mark_line(strokeWidth=2, color="#2a78d6").encode(
            x=alt.X("taux_automatisation:Q", title="Taux d'automatisation", axis=alt.Axis(format="%")),
            y=alt.Y("précision:Q", title="Précision", axis=alt.Axis(format="%"), scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("seuil:Q", format=".3f"), alt.Tooltip("taux_automatisation:Q", format=".1%"),
                     alt.Tooltip("précision:Q", format=".2%")])
        target = alt.Chart(pd.DataFrame({"y": [s["précision_cible"]]})).mark_rule(
            strokeDash=[4, 4], color="#8a8983").encode(y="y:Q")
        point = alt.Chart(pd.DataFrame({"taux_automatisation": [s["taux_automatisation"]], "précision": [s["précision"]],
                                        "point": ["fonctionnement réel"]})).mark_point(
            size=110, filled=True, color="#eb6834").encode(x="taux_automatisation:Q", y="précision:Q",
                                                           tooltip=["point", alt.Tooltip("taux_automatisation:Q", format=".1%"),
                                                                    alt.Tooltip("précision:Q", format=".2%")])
        st.altair_chart((line + target + point).properties(height=300), width="stretch")
        st.caption("Courbe : propositions triées par score après coup (a posteriori). Point orange : ce que le moteur a "
                   "réellement auto-validé avec ses seuils de validation.")

    if s.get("ml_paiements_en_ml") is not None:
        st.markdown("**Scoring ML sur la période** (premier passage de chaque paiement en ML)")
        mm = st.columns(4)
        mm[0].metric("Paiements passés en ML", num(s["ml_paiements_en_ml"]))
        mm[1].metric("Rappel des candidats", pct(s.get("ml_rappel_candidats")))
        p_at_1 = s.get("ml_precision_at_1_avec_positif")
        mm[2].metric("Precision@1", pct(p_at_1 if p_at_1 is not None else s.get("ml_precision_at_1"), 2),
                     help="Paiements dont une vraie facture est candidate : même périmètre que la validation (page 5). "
                          f"Sur tous les paiements passés en ML : {pct(s.get('ml_precision_at_1'), 2)}.")
        mrr = s.get("ml_mrr_avec_positif", s.get("ml_mrr"))
        mm[3].metric("MRR", "—" if mrr is None else f"{mrr:.3f}")

    kinds = [("by_month", "Par mois"), ("by_step", "Par étape"), ("by_rule", "Par règle"),
             ("by_client_file", "Avec / sans client file"), ("ml_calibration", "Calibration ML")]
    kinds = [(k, label) for k, label in kinds if (out / f"{name}_{k}.csv").exists()]
    tabs = st.tabs([label for _, label in kinds])
    for tab, (kind, _) in zip(tabs, kinds):
        with tab:
            df = table(kind)
            if df.empty:
                st.caption("Aucune auto-validation.")
                continue
            if {"auto", "corrects"} <= set(df.columns):
                df = with_precision_bounds(df)
            st.dataframe(df, hide_index=True, width="stretch", column_config={
                "précision": percent, "précision min. 95 %": percent, "taux_automatisation": rate})

    with st.expander("Rapport markdown", icon=":material/description:"):
        st.markdown((out / f"{name}.md").read_text(encoding="utf-8"))

    # --- Export et contrôles -------------------------------------------------------------------------------
    st.subheader("Export des décisions")
    st.caption("Une ligne par couple (paiement, facture) proposé ou réellement imputé, à joindre à la table d'imputation "
               "d'origine sur (payment_id, invoice_id) pour la revue métier.")
    c = st.columns([1, 1, 2], vertical_alignment="bottom")
    fmt = c[0].selectbox("Format", ["parquet", "csv"], key="evaluation.export_fmt")
    export = out / f"decisions_{matcher}_{period}.{fmt}"
    if c[1].button("Exporter", icon=":material/output:", key="evaluation.export"):
        code, lines = run_task("export_decisions", "Export des décisions", period=period, matcher=matcher, fmt=fmt)
        if code != 0:
            st.error("\n".join(lines[-5:]))
    if export.exists():
        size = export.stat().st_size / 2 ** 20
        c[2].caption(f"`{export}` · {size:.1f} Mo · {datetime.fromtimestamp(export.stat().st_mtime):%d/%m %H:%M}")
        if size < 200:
            with open(export, "rb") as handle:
                c[2].download_button("Télécharger", handle, file_name=export.name, icon=":material/download:",
                                     key="evaluation.download")

    parquet_export = out / f"decisions_{matcher}_{period}.parquet"
    with st.expander("Fuite F2 : IBAN connus avant le paiement qui les révèle (pire cas)", icon=":material/policy:"):
        st.caption("Le référentiel d'IBAN des débiteurs n'est pas daté : un IBAN appris grâce à un paiement peut aider à "
                   "rapprocher ce même paiement. Majorant : part des paiements dont l'IBAN est le premier imputé à ce "
                   "débiteur. Avec l'export parquet des décisions, la précision des auto-validations concernées.")
        if st.toggle("Calculer (quelques secondes, mis en cache)", key="evaluation.iban"):
            try:
                split = data_split(settings)
            except SplitError as exc:
                st.error(f"Découpage impossible : {exc}")
                st.stop()
            decisions = parquet_export if parquet_export.exists() else None
            f2 = diagnostic("iban_worst_case", interim_dir(settings), split, decisions,
                            _key=decisions.stat().st_mtime if decisions else None)
            st.dataframe(f2["par période"].reset_index(), hide_index=True, width="stretch",
                         column_config={c: percent for c in f2["par période"].columns})
            if "auto-validations" in f2:
                st.markdown(f"**Auto-validations** : {pct(f2['part révélée'], 2)} reposent peut-être sur un IBAN révélé")
                st.dataframe(f2["auto-validations"], hide_index=True, width="stretch",
                             column_config={"précision": percent})
            else:
                st.caption("Exporter les décisions au format parquet pour voir l'effet sur les auto-validations.")


def page_analysis() -> None:
    step_header("analysis")
    settings = current_settings()
    out = reports_dir(settings)
    target = settings.evaluation.target_precision
    percent = st.column_config.NumberColumn(format="percent")
    share = st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent")
    COLORS = {AUTO_OK: "#1baf7a", AUTO_KO: "#d62728", IN_REVIEW: "#eb6834", NO_DECISION: "#8a8983"}
    EXAMPLE_COLUMNS = ["payment_id", "arrival_day", "montant", "libellé", "issue", "règle", "score", "factures_proposées",
                       "débiteur_proposé", "factures_réelles", "débiteur_réel", "type_groupe", "allocation",
                       "vrai_débiteur_alloué", "candidats", "rang_vraie", "score_vraie", "score_premier", "raison",
                       "type_erreur"]

    def analysis_path(matcher: str, period: str):
        return out / f"analysis_{matcher}_{period}.parquet"

    def load(matcher: str, period: str) -> pd.DataFrame | None:
        path = analysis_path(matcher, period)
        return read_parquet(path) if path.exists() else None

    def examples(frame: pd.DataFrame, key: str, rows: int = 500) -> None:
        cols = [c for c in EXAMPLE_COLUMNS if c in frame.columns and frame[c].notna().any()]
        st.dataframe(frame[cols].head(rows), hide_index=True, width="stretch", height=320, column_config={
            "montant": st.column_config.NumberColumn("montant (€)", format="%.2f"),
            "score": st.column_config.NumberColumn(format="%.4f"),
            "score_vraie": st.column_config.NumberColumn(format="%.4f"),
            "score_premier": st.column_config.NumberColumn(format="%.4f")})
        st.download_button(f"Télécharger les {num(len(frame))} paiements (CSV)", frame[cols].to_csv(index=False),
                           file_name=f"{key}.csv", mime="text/csv", icon=":material/download:", key=f"analysis.dl.{key}")

    # --- Choix et calcul -----------------------------------------------------------------------------------------
    st.caption("Une ligne par paiement du périmètre, même vérité et même décision retenue que la page Évaluation : "
               "pourquoi une auto-validation est fausse, pourquoi un paiement reste manuel, où le moteur est bon ou "
               "faible. Calculée à la fin de chaque backtest, ou ici à la demande.")
    c = st.columns([2, 2, 1], vertical_alignment="bottom")
    period = c[0].segmented_control("Période", options=[*PERIODS, "all"], default="test", key="analysis.period",
                                    format_func=PERIOD_LABELS.get) or "test"
    matcher = c[1].selectbox("Rapprocheur", ["pipeline", "rules"], format_func=MATCHER_LABELS.get, key="analysis.matcher")
    replay = read_json(out / f"replay_{matcher}_{period}.json")
    if c[2].button("Calculer l'analyse", type="primary", icon=":material/troubleshoot:", key="analysis.run",
                   disabled=replay is None, help="Exige un rejeu de cette période (page Évaluation)."):
        code, lines = run_task("analyze", f"Analyse — {PERIOD_LABELS[period]}", period=period, matcher=matcher)
        if code != 0:
            st.error("\n".join(lines[-6:]))

    payload = read_json(out / f"analysis_{matcher}_{period}.json")
    p = load(matcher, period)
    if p is None or payload is None:
        st.info("Aucune analyse pour ce rapprocheur et cette période : lancer le backtest (page Évaluation) ou "
                "« Calculer l'analyse » si le rejeu existe.", icon=":material/info:")
        st.stop()
    ctx = payload["context"]
    stale(ctx.get("journal_sha256"), "Analyse")
    replay_file = out / f"replay_{matcher}_{period}.json"
    if replay_file.exists() and replay_file.stat().st_mtime > analysis_path(matcher, period).stat().st_mtime:
        st.warning("Le rejeu est plus récent que l'analyse : la recalculer.", icon=":material/sync_problem:")

    n = len(p)
    counts = p["issue"].value_counts()
    auto = int(counts.get(AUTO_OK, 0) + counts.get(AUTO_KO, 0))
    m = st.columns(5)
    m[0].metric("Paiements du périmètre", num(n))
    m[1].metric("Auto justes", num(counts.get(AUTO_OK, 0)), help=pct(counts.get(AUTO_OK, 0) / n if n else None))
    m[2].metric("Auto fausses", num(counts.get(AUTO_KO, 0)),
                help=f"Précision {pct(counts.get(AUTO_OK, 0) / auto if auto else None, 2)}")
    m[3].metric("En revue", num(counts.get(IN_REVIEW, 0)), help=pct(counts.get(IN_REVIEW, 0) / n if n else None))
    m[4].metric("Sans décision", num(counts.get(NO_DECISION, 0)), help=pct(counts.get(NO_DECISION, 0) / n if n else None))
    st.caption(f"{PERIOD_LABELS[period]} du {ctx['start']} au {ctx['end']} · {MATCHER_LABELS.get(matcher, matcher)} · "
               f"calculée le {datetime.fromtimestamp(analysis_path(matcher, period).stat().st_mtime):%d/%m %H:%M}"
               + ("" if ctx.get("ml_candidates") else " · sans candidats ML (rapprocheur de règles)"))

    tab_errors, tab_residual, tab_segments, tab_compare = st.tabs(
        ["Analyse des erreurs", "Résiduel manuel expliqué", "Segments", "Comparaison des rapprocheurs"])

    # --- Erreurs -------------------------------------------------------------------------------------------------
    with tab_errors:
        wrong = p[p["issue"] == AUTO_KO]
        st.caption("Chaque auto-validation fausse comparée à l'imputation réelle du paiement. « Paiement jamais imputé » "
                   ": le moteur a validé un paiement qui n'a aucune imputation réelle. « Bon débiteur, autre facture » : "
                   "même débiteur, aucune facture commune. « Ensemble incomplet / factures en trop » : la proposition est "
                   "contenue dans la vérité, ou la contient.")
        if wrong.empty:
            st.success("Aucune auto-validation fausse.", icon=":material/check_circle:")
        else:
            by_type = wrong["type_erreur"].value_counts().reindex([e for e in ERROR_TYPES if e in set(wrong["type_erreur"])])
            left, right = st.columns([2, 3], gap="large")
            with left:
                chart = by_type.rename("paiements").rename_axis("type").reset_index()
                st.altair_chart(alt.Chart(chart).mark_bar(color=COLORS[AUTO_KO]).encode(
                    x=alt.X("paiements:Q"), y=alt.Y("type:N", sort=None, title=None),
                    tooltip=["type", "paiements"]).properties(height=34 * len(chart) + 30), width="stretch")
            with right:
                by = st.radio("Par", ["règle", "étape", "type_groupe", "tranche_montant"], horizontal=True,
                              key="analysis.errors.by", format_func=lambda v: v.replace("_", " "))
                st.dataframe(error_table(p, by), hide_index=True, width="stretch")
            st.markdown("**Précision par règle**, avec sa borne basse à 95 %")
            rules = rate_by(p[p["issue"].isin([AUTO_OK, AUTO_KO])], "règle")
            st.dataframe(rules[["règle", "auto", "justes", "précision", "précision min. 95 %"]], hide_index=True,
                         width="stretch", column_config={"précision": percent, "précision min. 95 %": percent})
            st.markdown("**Auto-validations fausses**")
            f = st.columns(2)
            types = f[0].multiselect("Type d'erreur", list(by_type.index), key="analysis.errors.types")
            rules_f = f[1].multiselect("Règle", sorted(wrong["règle"].unique()), key="analysis.errors.rules")
            keep = pd.Series(True, index=wrong.index)
            if types:
                keep &= wrong["type_erreur"].isin(types)
            if rules_f:
                keep &= wrong["règle"].isin(rules_f)
            shown = wrong[keep]
            examples(shown.sort_values("montant", ascending=False), "erreurs")

    # --- Résiduel ------------------------------------------------------------------------------------------------
    with tab_residual:
        st.caption("Pour chaque paiement non auto-validé, le premier blocage rencontré, du débiteur au seuil : un "
                   "paiement n'est compté qu'une fois. « Gain maximal » : points d'automatisation si le levier était "
                   "parfait (borne haute, à précision inchangée). Candidats ML : ceux du premier passage du paiement en "
                   "ML.")
        reasons = reason_table(p)
        if reasons.empty:
            st.success("Aucun paiement résiduel.", icon=":material/check_circle:")
        else:
            left, right = st.columns([3, 2], gap="large")
            with left:
                st.dataframe(reasons, hide_index=True, width="stretch", column_config={
                    "paiements": st.column_config.NumberColumn(format="localized"), "part du résiduel": share,
                    "gain maximal (points)": st.column_config.NumberColumn(format="%.2f")})
            with right:
                levers = reasons[reasons["raison"] != NO_TRUTH]
                st.altair_chart(alt.Chart(levers).mark_bar(color="#2a78d6").encode(
                    x=alt.X("gain maximal (points):Q"), y=alt.Y("raison:N", sort=None, title=None,
                                                                axis=alt.Axis(labelLimit=260)),
                    tooltip=["raison", "paiements", alt.Tooltip("gain maximal (points):Q", format=".2f")],
                ).properties(height=30 * len(levers) + 30), width="stretch")
            if "vrai_débiteur_alloué" in p and p["vrai_débiteur_alloué"].isna().all() and ctx.get("payment_info"):
                st.caption("Le rejeu n'a pas enregistré la liste des débiteurs alloués (version antérieure) : relancer le "
                           "backtest pour distinguer « débiteur non alloué » parmi les vraies factures absentes des "
                           "candidats.")
            residual = p[p["raison"] != ""]
            st.markdown("**Raison × type de groupe réel**")
            st.dataframe(pd.crosstab(residual["raison"], residual["type_groupe"], margins=True, margins_name="total")
                         .reindex([r for r in [*REASONS, "total"] if r in set(residual["raison"]) | {"total"}]),
                         width="stretch")
            right_prop = residual[residual["raison"].isin(REASONS[:2]) & residual["score"].notna()]
            if len(right_prop):
                st.markdown("**Propositions justes restées en revue** : score de la proposition")
                st.altair_chart(alt.Chart(right_prop[["score", "règle"]]).mark_bar().encode(
                    x=alt.X("score:Q", bin=alt.Bin(maxbins=40), title="score"), y=alt.Y("count()", title="paiements"),
                    color=alt.Color("règle:N", title=None, legend=alt.Legend(orient="top")),
                ).properties(height=220), width="stretch")
                st.caption("Proches de τ_high : candidates à une baisse du seuil si la précision le permet (page ML, courbe "
                           "des seuils). Score élevé mais en revue : marge au second insuffisante ou type sans seuil.")
            ranked = residual[residual["rang_vraie"].notna()] if "rang_vraie" in residual else residual.iloc[0:0]
            if len(ranked):
                st.markdown("**Rang de la vraie facture parmi les candidats** (paiements résiduels passés en ML)")
                dist = ranked["rang_vraie"].clip(upper=10).astype(int).value_counts().sort_index()
                dist.index = [f"{i}" if i < 10 else "≥ 10" for i in dist.index]
                st.bar_chart(dist.rename("paiements"), height=200)
            reason = st.selectbox("Exemples pour la raison", list(reasons["raison"]), key="analysis.residual.reason")
            examples(residual[residual["raison"] == reason].sort_values("montant", ascending=False), "residuel")

    # --- Segments ------------------------------------------------------------------------------------------------
    with tab_segments:
        available = {label: col for label, col in SEGMENTS.items() if col in p and p[col].nunique() > 1}
        c = st.columns(2)
        label = c[0].selectbox("Segment", list(available), key="analysis.segment")
        cross = c[1].selectbox("Croiser avec", [None, *[s for s in available if s != label]], key="analysis.cross",
                               format_func=lambda v: "—" if v is None else v)
        col = available[label]
        table = rate_by(p, col)
        left, right = st.columns([3, 2], gap="large")
        with left:
            st.dataframe(table, hide_index=True, width="stretch", column_config={
                "paiements": st.column_config.NumberColumn(format="localized"), "part des paiements": share,
                "taux d'automatisation": share, "précision": percent, "précision min. 95 %": percent})
        with right:
            base = alt.Chart(table).encode(y=alt.Y(f"{col}:N", sort=None, title=None, axis=alt.Axis(labelLimit=200)))
            st.altair_chart(base.mark_bar(color="#2a78d6").encode(
                x=alt.X("taux d'automatisation:Q", axis=alt.Axis(format="%")),
                tooltip=[col, "paiements", alt.Tooltip("taux d'automatisation:Q", format=".1%")],
            ).properties(height=26 * len(table) + 40, title="Taux d'automatisation"), width="stretch")
            prec = table.dropna(subset=["précision"])
            if len(prec):
                pbase = alt.Chart(prec).encode(y=alt.Y(f"{col}:N", sort=None, title=None, axis=alt.Axis(labelLimit=200)))
                rule = pbase.mark_rule(color="#8a8983").encode(x=alt.X("précision min. 95 %:Q", axis=alt.Axis(format="%"),
                                                                        scale=alt.Scale(zero=False), title="précision"),
                                                                x2="précision:Q")
                dots = pbase.mark_point(filled=True, size=70, color="#1baf7a").encode(
                    x="précision:Q", tooltip=[col, "auto", alt.Tooltip("précision:Q", format=".2%"),
                                              alt.Tooltip("précision min. 95 %:Q", format=".2%")])
                target_line = alt.Chart(pd.DataFrame({"x": [target]})).mark_rule(strokeDash=[4, 4], color="#d62728") \
                    .encode(x="x:Q")
                st.altair_chart((rule + dots + target_line).properties(height=26 * len(prec) + 40,
                                                                       title="Précision (trait : borne basse 95 %)"),
                                width="stretch")
        st.caption("Un segment dont la borne basse passe sous la cible (trait rouge) est candidat à un seuil propre ou à "
                   "une règle désactivée sur ce segment.")
        if cross:
            col2 = available[cross]
            grid = p.assign(_auto=p["issue"].isin([AUTO_OK, AUTO_KO]))
            pivot = grid.pivot_table(index=col, columns=col2, values="_auto", aggfunc="mean")
            sizes = grid.pivot_table(index=col, columns=col2, values="_auto", aggfunc="size")
            heat = pivot.stack().rename("taux").reset_index().merge(sizes.stack().rename("paiements").reset_index(),
                                                                    on=[col, col2])
            st.altair_chart(alt.Chart(heat).mark_rect().encode(
                x=alt.X(f"{col2}:N", title=cross), y=alt.Y(f"{col}:N", title=label),
                color=alt.Color("taux:Q", title="automatisation", scale=alt.Scale(scheme="blues"), legend=alt.Legend(format="%")),
                tooltip=[col, col2, alt.Tooltip("taux:Q", format=".1%"), alt.Tooltip("paiements:Q", format=",")],
            ).properties(height=28 * pivot.shape[0] + 40, title=f"Taux d'automatisation : {label} × {cross}"),
                width="stretch")

    # --- Comparaison ---------------------------------------------------------------------------------------------
    with tab_compare:
        before_m, after_m = ("rules", "pipeline")
        before, after = load(before_m, period), load(after_m, period)
        st.caption("Mêmes paiements, deux rapprocheurs : règles seules (étape 4), puis règles + ML (étapes 4 et 5). "
                   "Ce que le ML ajoute, ce qu'il change, segment par segment.")
        if before is None or after is None:
            missing = [MATCHER_LABELS[mm] for mm, frame in ((before_m, before), (after_m, after)) if frame is None]
            st.info(f"Analyse manquante pour {', '.join(missing)} sur cette période : la calculer en choisissant ce "
                    "rapprocheur ci-dessus (le rejeu doit exister), ou lancer son backtest (page Évaluation).",
                    icon=":material/info:")
        else:
            table = compare_outcomes(before, after)
            common = before[["payment_id", "issue"]].merge(after[["payment_id", "issue"]], on="payment_id",
                                                           suffixes=("_avant", "_après"))
            gained = int(((common["issue_avant"] != AUTO_OK) & (common["issue_après"] == AUTO_OK)).sum())
            new_errors = int(((common["issue_avant"] != AUTO_KO) & (common["issue_après"] == AUTO_KO)).sum())
            lost = int(((common["issue_avant"] == AUTO_OK) & (common["issue_après"] != AUTO_OK)).sum())
            mm = st.columns(4)
            mm[0].metric("Paiements comparés", num(len(common)))
            mm[1].metric("Auto justes gagnées", num(gained), help=f"{100 * gained / max(len(common), 1):+.2f} points")
            mm[2].metric("Auto fausses ajoutées", num(new_errors))
            mm[3].metric("Auto justes perdues", num(lost), help="Paiements que les règles seules validaient justement et "
                                                                "que la pipeline complète ne valide plus (arbitrage).")
            st.markdown("**Issue avant (lignes, règles seules) → après (colonnes, règles + ML)**")
            st.dataframe(table, width="stretch")
            seg_label = st.selectbox("Gain par segment", list(available), key="analysis.compare.segment")
            seg = available[seg_label]
            a = rate_by(before, seg)[[seg, "paiements", "taux d'automatisation", "précision"]]
            b = rate_by(after, seg)[[seg, "taux d'automatisation", "précision"]]
            merged = a.merge(b, on=seg, suffixes=(" règles", " règles + ML"))
            merged["gain (points)"] = 100 * (merged["taux d'automatisation règles + ML"] - merged["taux d'automatisation règles"])
            st.dataframe(merged, hide_index=True, width="stretch", column_config={
                "taux d'automatisation règles": share, "taux d'automatisation règles + ML": share,
                "précision règles": percent, "précision règles + ML": percent,
                "gain (points)": st.column_config.NumberColumn(format="%+.1f")})


def page_runtime() -> None:
    step_header("runtime")
    settings = current_settings()
    folder = runtime_dir(settings)
    minutes = st.column_config.NumberColumn(format="%.1f")
    share = st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent")

    # --- Durées enregistrées par les rapports -------------------------------------------------------------
    st.subheader("Durées enregistrées par les rapports")
    st.caption("Chaque étape écrit ses durées dans son rapport : disponible même sans suivi.")
    rows = []
    out = reports_dir(settings)
    for path in sorted(out.glob("replay_*.json")):
        ctx = json.loads(path.read_text(encoding="utf-8"))
        label = f"rejeu {PERIOD_LABELS.get(ctx['period'], ctx['period'])} · {MATCHER_LABELS.get(ctx['matcher'], ctx['matcher'])}"
        rows += [{"exécution": label, "phase": k, "minutes": v / 60} for k, v in ctx.get("timings_s", {}).items()]
    for path in sorted(out.glob("allocation_*.json")):
        ctx = json.loads(path.read_text(encoding="utf-8")).get("context", {})
        label = f"allocation {PERIOD_LABELS.get(ctx.get('period'), ctx.get('period'))}"
        rows += [{"exécution": label, "phase": k, "minutes": v / 60} for k, v in ctx.get("timings_s", {}).items()]
    model = model_dir(settings) / "model.json"
    if model.exists():
        timings = json.loads(model.read_text(encoding="utf-8")).get("timings_s", {})
        rows += [{"exécution": "entraînement ML", "phase": k, "minutes": v / 60} for k, v in timings.items()]
    loaded = load_meta(settings)
    if loaded:
        rows += [{"exécution": "chargement", "phase": k, "minutes": v / 60} for k, v in loaded.get("timings_s", {}).items()]
    if rows:
        durations = pd.DataFrame(rows)
        totals = durations.groupby("exécution", sort=False)["minutes"].sum().rename("total (min)").reset_index()
        left, right = st.columns([2, 3], gap="large")
        left.dataframe(totals, hide_index=True, width="stretch", column_config={"total (min)": minutes})
        right.dataframe(durations, hide_index=True, width="stretch", column_config={"minutes": minutes}, height=280)
    else:
        st.caption("Aucun rapport : lancer une étape.")

    # --- Suivi du temps et de la mémoire -------------------------------------------------------------------
    st.subheader("Temps par étape et sous-étape")
    history = folder / "memory.csv"
    if not history.exists():
        st.info(f"Aucun suivi dans `{history}`. Activer « Suivi du temps et de la mémoire » dans la barre latérale (étapes "
                "lancées d'ici) ou démarrer le moniteur dans le notebook.", icon=":material/timer_off:")
    else:
        sessions = memory.sessions(history)
        st.caption(f"`{history}` : relevé chaque seconde de l'étape, du jour et de la sous-étape en cours (règle, signal "
                   "d'allocation, phase du ML). Le temps entre deux relevés est attribué à la phase en cours ; l'inactivité "
                   "entre deux étapes est exclue. Coût nul : aucun chronomètre dans le code.")
        options = [None, *sessions["session"].tolist()[::-1]]
        labels = {None: "toutes les sessions"} | {
            r.session: f"{r.début} · {r.minutes:.0f} min · {r.étapes or '—'}" for r in sessions.itertuples()}
        session = st.selectbox("Session", options, index=1 if len(options) > 1 else 0, format_func=labels.get,
                               key="runtime.session", help="Une session par démarrage du suivi (une étape lancée d'ici, "
                                                           "ou un démarrage du moniteur dans le notebook).")
        position = None if session is None else sessions["session"].tolist().index(session)
        profile = memory.time_profile(history, position)
        if profile.empty:
            st.caption("Aucun relevé hors inactivité dans cette session.")
        else:
            by_step = profile.groupby("étape", sort=False)["minutes"].sum().rename("minutes").reset_index()
            by_step["part du total"] = by_step["minutes"] / by_step["minutes"].sum()
            left, right = st.columns([2, 3], gap="large")
            with left:
                st.markdown("**Par étape**")
                st.dataframe(by_step, hide_index=True, width="stretch",
                             column_config={"minutes": minutes, "part du total": share})
            with right:
                st.markdown("**Par sous-étape**")
                st.dataframe(profile, hide_index=True, width="stretch", height=320, column_config={
                    "minutes": minutes, "part de l'étape": share, "part du total": share})
            st.altair_chart(memory.time_chart(history, position), width="stretch")
            st.markdown("**Mémoire**")
            if position is not None:
                st.altair_chart(memory.chart(history, position), width="stretch")
            st.dataframe(memory.by_phase(history, position), hide_index=True, width="stretch", height=260)
        with st.expander("Sessions", icon=":material/history:"):
            st.dataframe(sessions, hide_index=True, width="stretch", column_config={"minutes": minutes})

    # --- Profil par fonction -------------------------------------------------------------------------------
    st.subheader("Profil par fonction")
    runs = profiling.saved_runs(folder / "profiles")
    pyspy = profiling.find_pyspy()
    st.caption("Échantillonnage de la pile d'appels pendant l'étape : temps « propre » de chaque fonction du projet "
               "(bibliothèques comprises : c'est là qu'il faut optimiser) et temps « inclus » (avec ce qu'elle appelle). "
               f"Moteur : {'py-spy (' + pyspy + ')' if pyspy else 'fil interne (py-spy absent)'}. Activer « Profil par "
               "fonction » dans la barre latérale, puis relancer l'étape.")
    if runs.empty:
        st.info("Aucun profil enregistré.", icon=":material/insights:")
    else:
        choice = st.selectbox("Profil", runs.index, key="runtime.profile",
                              format_func=lambda i: f"{runs.loc[i, 'nom']} · {runs.loc[i, 'fin']} · "
                                                    f"{runs.loc[i, 'secondes'] / 60:.1f} min · {runs.loc[i, 'moteur']}")
        top = st.slider("Fonctions affichées", 10, 100, 30, step=10, key="runtime.top")
        run = runs.loc[choice].to_dict()
        summary = profiling.saved_summary(run, top=top)
        st.dataframe(summary, hide_index=True, width="stretch", height=420, column_config={
            "propre (s)": st.column_config.NumberColumn(format="%.1f"),
            "inclus (s)": st.column_config.NumberColumn(format="%.1f"), "part propre": share})
        with open(run["fichier"], "rb") as handle:
            st.download_button("Télécharger les piles (speedscope.app, flamegraph.pl)", handle,
                               file_name=f"{run['nom']}.txt".replace(" · ", "_").replace(" ", "_"),
                               icon=":material/download:", key="runtime.download")


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# Navigation
# ════════════════════════════════════════════════════════════════════════════════════════════════════


def main() -> None:
    st.set_page_config(page_title="Rapprochement automatique", page_icon=":material/join:", layout="wide")

    def page(key: str, fn, default: bool = False) -> st.Page:
        step = STEPS[key]
        return st.Page(fn, title=step.title, icon=step.icon, url_path=key, default=default)

    nav = st.navigation({
        "": [st.Page(page_home, title="Vue d'ensemble", icon=":material/home:", default=True)],
        "Préparation": [page("config", page_config), page("ingestion", page_ingestion), page("load", page_load),
                        page("split", page_split)],
        "Rapprochement": [page("allocation", page_allocation), page("rules", page_rules), page("ml", page_ml)],
        "Mesure": [page("evaluation", page_evaluation), page("analysis", page_analysis), page("runtime", page_runtime)],
    })

    with st.sidebar:
        meta = load_meta()
        if meta:
            st.caption(f"Dernier chargement : `{meta['source']}` · {meta['journal_events']:,} événements · "
                       f"normalisation v{meta['normalization_version']}".replace(",", " "))
        else:
            st.caption("Aucune donnée chargée pour ce jeu.")
        st.divider()
        st.markdown("**Étapes lancées d'ici**")
        st.toggle("Suivi du temps et de la mémoire", value=True, key="ui.monitor",
                  help="Relevé chaque seconde dans reports/memory.csv, comme le notebook : coût nul. "
                       "Résultat dans la page Temps de run.")
        st.toggle("Profil par fonction", value=False, key="ui.profile",
                  help="Échantillonnage de la pile d'appels (py-spy si installé, sinon fil interne, impact < 3 %) : "
                       "temps par fonction du projet, dans la page Temps de run.")

    nav.run()


if __name__ == "__main__":
    main()
