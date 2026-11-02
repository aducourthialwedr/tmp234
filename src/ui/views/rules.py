import pandas as pd
import streamlit as st
from pydantic import ValidationError

from src.diagnostics import clopper_pearson_lower, with_precision_bounds
from src.settings import RulesConfig, load_rules, save_rules
from src.timeline.split import PERIODS
from src.ui.common import (
    PERIOD_LABELS, current_settings, num, pct, read_json, reports_dir, rules_path, run_task, stale, step_header,
    validation_errors,
)

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
