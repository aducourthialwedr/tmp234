import altair as alt
import pandas as pd
import streamlit as st

from src.ui.common import (
    current_settings, diagnostic, field_widget, interim_dir, model_dir, num, pct, read_json, render_model, run_task,
    save_section, stale, step_header,
)

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
