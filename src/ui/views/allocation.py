import json

import pandas as pd
import streamlit as st

from src.timeline.split import PERIODS
from src.ui.common import (
    PERIOD_LABELS, current_settings, diagnostic, interim_dir, load_meta, num, render_model, reports_dir, run_task,
    save_section, step_header,
)

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
