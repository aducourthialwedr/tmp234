from datetime import datetime

import streamlit as st

from src.diagnostics import clopper_pearson_lower
from src.settings import load_rules
from src.ui.common import (
    REAL, STEPS, current_settings, data_split, dataset, interim_dir, load_meta, model_dir, num, pct, read_json,
    read_report, reports_dir, rules_path, runtime_dir, schema_path, settings_path,
)

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
    from src.config import resolve_path
    from src.load.loader import _check_config
    from src.load.schema_io import load_schema
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
