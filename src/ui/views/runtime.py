import json

import pandas as pd
import streamlit as st

from src import memory, profiling
from src.ui.common import (
    MATCHER_LABELS, PERIOD_LABELS, current_settings, load_meta, model_dir, reports_dir, runtime_dir, step_header,
)

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
