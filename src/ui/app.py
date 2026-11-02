"""UI de pilotage de la pipeline.

    python -m streamlit run src/ui/app.py

Tourne en local ; la télémétrie Streamlit est désactivée (.streamlit/config.toml).
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from src.ui.common import REAL, STEPS, SYNTHETIC, load_meta  # noqa: E402

st.set_page_config(page_title="Rapprochement automatique", page_icon=":material/join:", layout="wide")

VIEWS = Path(__file__).parent / "views"


def page(key: str, file: str, default: bool = False) -> st.Page:
    step = STEPS[key]
    return st.Page(VIEWS / file, title=step.title, icon=step.icon, url_path=key, default=default)


nav = st.navigation({
    "": [st.Page(VIEWS / "home.py", title="Vue d'ensemble", icon=":material/home:", default=True)],
    "Préparation": [page("config", "config.py"), page("ingestion", "ingestion.py"), page("load", "load.py"),
                    page("split", "split.py")],
    "Rapprochement": [page("allocation", "allocation.py"), page("rules", "rules.py"), page("ml", "ml.py")],
    "Mesure": [page("evaluation", "evaluation.py"), page("analysis", "analysis.py"), page("runtime", "runtime.py")],
})

with st.sidebar:
    # <synthétique>
    st.radio("Jeu de données", options=[SYNTHETIC, REAL], key="dataset", horizontal=True,
             help="Synthétique : données générées (seed fixe). Réel : sources de config/schema.yaml.")
    # </synthétique>
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
