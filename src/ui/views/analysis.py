from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

from src.evaluation.analysis import (
    AUTO_KO, AUTO_OK, ERROR_TYPES, IN_REVIEW, NO_DECISION, NO_TRUTH, REASONS, SEGMENTS, compare_outcomes,
    error_table, rate_by, reason_table,
)
from src.timeline.split import PERIODS
from src.ui.common import (
    MATCHER_LABELS, PERIOD_LABELS, current_settings, num, pct, read_json, read_parquet, reports_dir, run_task, stale,
    step_header,
)

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
