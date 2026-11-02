import json
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

from src.diagnostics import clopper_pearson_lower, with_precision_bounds
from src.timeline.split import PERIODS, SplitError
from src.ui.common import (
    MATCHER_LABELS, PERIOD_LABELS, current_settings, data_split, diagnostic, interim_dir, num, pct, read_json,
    render_model, reports_dir, run_task, save_section, stale, step_header,
)

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
