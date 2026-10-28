"""Génère notebooks/pipeline.ipynb et sa version portable single/pipeline.ipynb (outil de développement)."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CELLS: list[tuple[str, str]] = [
    ("md", """# Rapprochement automatique paiements / factures — pipeline

Chaque étape du brief est une méthode de `Project` (`src/api.py`). Les résultats sont écrits sur disque
(`data/interim`, `reports/`, `models/`) et renvoyés sous forme de tableaux.

- **Paramètres** : `config/settings.yaml` (par étape) et `config/rules.yaml` (règles de l'étape 4),
  éditables à la main, depuis l'interface (`streamlit run src/ui/app.py`) ou depuis ce notebook.
- **Données réelles** : renseigner `config/schema.yaml`, puis `Project(dataset="real")`.
- **Données synthétiques** : `Project(dataset="synthetic")` — volume réglable (réel : ~2 M de paiements ;
  commencer par 50 000 pour une exécution de quelques minutes).
- **Temps de run** : profil par étape et sous-étape, toujours actif et sans coût, et profil par fonction des
  étapes longues avec `PROFILE = True` (§ « Profil de temps », en fin de notebook)."""),
    ("code", """import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

import altair as alt
import pandas as pd
from IPython.display import Markdown, display

from src import memory, profiling, progress
from src.api import Project

pd.set_option("display.max_columns", 50)
MEMORY_LOG = ROOT / "reports" / "memory.csv"
N_PAYMENTS = 50_000          # 2_000_000 pour le volume réel (≈ 1 h de bout en bout)
PROFILE = False              # profil par fonction des étapes longues (échantillonnage, < 3 % de surcoût)"""),
    ("md", """## Suivi de la mémoire et de la progression

Les tâches longues affichent une **barre de progression** dans la sortie de leur cellule : jours rejoués
(allocation, règles, backtest, jeu d'entraînement, calibration des seuils) et itérations LightGBM (passe 1,
plis hors échantillon, passe 2). Chaque barre donne le temps écoulé, le temps restant estimé et un détail
(jour et taille du lot, perte de validation).

La cellule suivante lance un relevé chaque seconde :
- **RSS** : mémoire du processus ;
- **pod** : mémoire du conteneur (cgroup), dont **anonyme**, ce que l'OOM killer compte, et la limite du pod ;
- **phase en cours** : étape, jour du rejeu, sous-étape (signal d'allocation, règle, candidats ML…).

Ce que la cellule produit :
- une **pastille** sous la cellule, mise à jour en continu pendant l'exécution des cellules suivantes ;
- l'état mémoire **ajouté à chaque message** de la pipeline ;
- chaque relevé **écrit immédiatement** dans `reports/memory.csv`.

**Si le pod est tué (OOM)**, le fichier reste. Relancer le noyau, exécuter la cellule des imports, puis la
cellule « Analyse mémoire » en fin de notebook : elle donne le pic par étape / sous-étape et le jour
concerné. Pour suivre en direct hors du notebook, ouvrir un terminal JupyterLab : `tail -f reports/memory.csv`.

À la fin de chaque jour de rejeu, la mémoire libérée est rendue au système (`trim_daily`, via
`malloc_trim`). Sans cela, sous Linux, le RSS ne redescend jamais. Autre réglage utile : lancer JupyterLab
avec `MALLOC_ARENA_MAX=2` dans l'environnement, qui limite la fragmentation de glibc.

**Mesurer d'où vient le temps**, sans ralentir le run :
- **par étape et sous-étape**, toujours actif : les relevés du suivi mémoire portent la phase en cours. Compter
  les secondes par phase donne un profil de temps gratuit (§ « Profil de temps »). Le temps d'inactivité entre
  deux cellules est exclu ;
- **par fonction**, avec `PROFILE = True` : chaque étape longue est enveloppée dans `with prof("…")`, qui relève
  la pile d'appels 10 fois par seconde. Avec **py-spy** installé (`pip install py-spy`), le relevé se fait depuis
  l'extérieur du processus, sans le suspendre ; sinon, un fil Python interne prend le relais. Surcoût mesuré
  inférieur à 3 % dans les deux cas, contre 1,5 à 3 fois plus lent avec un profileur classique (cProfile)."""),
    ("code", """monitor = memory.MemoryMonitor(MEMORY_LOG, interval=1.0, trim_daily=True).start()
monitor.live()                                     # pastille mise à jour chaque seconde
progress.use_notebook()                            # barres de progression (rejeux, apprentissage)
project = Project(dataset="synthetic", log=monitor.log)
prof = profiling.Profiler(ROOT / "reports" / "profiles", enabled=PROFILE, log=monitor.log)"""),
    ("md", """### Réglages mémoire

Ces tailles de blocs bornent la mémoire d'une journée, sans changer les résultats. Autre conseil : **redémarrer
le noyau après l'étape 1** (chargement), puis reprendre à l'étape 2 : la mémoire du chargement n'est pas
toujours rendue au système. Pour le chargement lui-même, `load.workers` (processus parallèles de
normalisation) multiplie la mémoire ; le réduire si l'étape 1 est tuée. Il faut les réduire si
une sous-étape dépasse la limite du pod (au prix d'un peu de temps). D'autres leviers changent les
résultats :
- `split.retention_days` : taille du reliquat ;
- `reconcile_ml.training.payment_sample` : part des paiements du jeu d'entraînement."""),
    ("code", """import src.allocation.allocator as _allocation
import src.reconcile_ml.features as _features
import src.reconcile_ml.pipeline as _pipeline

_allocation.CHUNK_ROWS = 20_000                 # paiements par bloc d'allocation
_allocation.NAME_PAIR_BUDGET = 2_000_000        # paires (paiement, débiteur) du signal « nom » par paquet
_allocation.CF_PAIR_BUDGET = 2_000_000          # couples (paiement, client file de même montant) par paquet
_features.CANDIDATE_PAIR_BUDGET = 2_000_000     # paires (paiement, facture) examinées par bloc de candidats ML
_pipeline.FEATURE_BLOCK_PAIRS = 500_000         # paires par bloc de calcul des features"""),
    ("md", "## Paramètres\n\nLecture et modification des paramètres depuis le notebook (écrits dans `config/settings.yaml`)."),
    ("code", """from src.settings import save_settings

settings = project.settings
print(settings.split)
print(settings.allocation.signals)
# Exemple : settings.split.purge_days = 7 ; save_settings(settings, project.settings_path)"""),
    ("md", "## Étape 1 — chargement, normalisation, journal"),
    ("code", """with prof("étape 1 · chargement"):
    step1 = project.load(n_payments=N_PAYMENTS)
display(step1["volumes"])
display(step1["issues"][step1["issues"]["count"] > 0])
display(step1["missing_fields"][~step1["missing_fields"]["mapped"] | (step1["missing_fields"]["null"] > 0)])
step1["meta"]"""),
    ("md", "## Étape 2 — découpage temporel et boucle quotidienne"),
    ("code", """split = project.split()
display(pd.DataFrame([{"période": p.name, "début": p.start, "fin": p.end, "jours": p.days} for p in split.periods]))

with prof("étape 2 · rejeu vide"):
    replay = project.replay("test", "null")    # rapprocheur vide : taille des lots et reliquat
daily = replay["daily"].melt(id_vars="day", value_vars=["new", "carried"], var_name="origine", value_name="paiements")
alt.Chart(daily).mark_bar().encode(x="day:T", y="paiements:Q", color="origine:N").properties(height=240)"""),
    ("md", "## Étape 3 — allocation (rappel cible ≥ 99 %)"),
    ("code", """with prof("étape 3 · allocation"):
    alloc = project.measure_allocation("validation")
display(pd.Series(alloc["summary"]))
display(alloc["by_route"])
display(alloc["found_by"].head(10))"""),
    ("md", "## Étape 4 — réconciliation par règles (baseline)"),
    ("code", """with prof("étape 4 · règles"):
    rules = project.backtest("test", matcher="rules")
display(pd.Series(rules["summary"]))
display(rules["by_rule_alone"])
display(rules["by_rule"])"""),
    ("md", """## Étape 5 — réconciliation ML (entraînement sur train, seuils sur la validation)

`project.train()` rejoue entraînement + validation, entraîne le modèle et calibre les seuils sur la boucle
réelle de validation. Après un changement des paramètres de décision (marge δ, précision cible),
`project.calibrate()` recalcule seulement les seuils, sans réentraîner.

Le jeu d'entraînement (rejeu train + validation, la phase la plus longue) est enregistré dans
`data/interim/ml/dataset`. Si l'apprentissage est interrompu après sa construction (pendant LightGBM
ou la calibration des seuils), `project.train(reuse_dataset=True)` repart de ce jeu. La reprise est
refusée si les données, la featurisation ou les réglages dont il dépend ont changé.

**Scores OOF de la pass 2** (`reconcile_ml.training.oof_scheme`) :
- `walk_forward` (par défaut) : chaque mois du train est scoré par un modèle appris sur le seul passé, avec les
  labels connus à cette date, comme en production. `walk_forward_step_months` (un modèle tous les N mois)
  accélère l'apprentissage sur un long historique ; `walk_forward_window_months` borne la fenêtre ;
- `random_folds` : version précédente (3 folds tirés par paiement), qui voit le futur du train (fuite F4).
  Elle reste disponible par ce réglage.

Le bilan du walk-forward (mois, paires apprises, modèles) est dans `meta["oof"]`."""),
    ("code", """with prof("étape 5 · entraînement"):
    meta = project.train()    # reuse_dataset=True : reprend le jeu déjà construit (run interrompu)
if meta.get("oof", {}).get("mois"):
    display(pd.DataFrame(meta["oof"]["mois"]))
display(pd.Series(meta["metrics"]["validation"]))
display(pd.Series(meta["thresholds"]["kinds"], name="τ_high par type"))
pd.read_csv(project.model_dir / "feature_importance.csv").head(15)"""),
    ("md", "## Étape 6 — backtest complet et tableau en cascade"),
    ("code", """with prof("étape 6 · backtest"):
    ev = project.backtest("test", matcher="pipeline")
display(pd.Series(ev["summary"]))
display(ev["cascade"])
display(ev["by_group"])
display(ev["by_step"])"""),
    ("code", """curve = ev["curve"].replace([float("inf")], None).dropna()
target = ev["summary"]["précision_cible"]
line = alt.Chart(curve).mark_line().encode(x=alt.X("taux_automatisation:Q", axis=alt.Axis(format="%")),
                                           y=alt.Y("précision:Q", scale=alt.Scale(zero=False), axis=alt.Axis(format="%")))
rule = alt.Chart(pd.DataFrame({"y": [target]})).mark_rule(strokeDash=[4, 4]).encode(y="y:Q")
(line + rule).properties(height=260, title="Automatisation / précision (règles acquises, seuil ML variable)")"""),
    ("code", """for name in ("by_month", "by_client_file", "ml_calibration"):
    if name in ev:
        display(Markdown(f"**{name}**"))
        display(ev[name])
display(Markdown((project.reports_dir / "evaluation_pipeline_test.md").read_text(encoding="utf-8")))"""),
    ("md", """## Export des imputations proposées

Une ligne par couple (paiement, facture) proposé par le moteur, plus les couples réellement imputés qu'il
n'a pas proposés (`statut_paire`). Le fichier se joint à la table d'imputation d'origine, et à ses
métadonnées, sur (payment_id, invoice_id). Colonnes détaillées dans `RESULTATS.md`, §6.11."""),
    ("code", """path = project.export_decisions("test", "pipeline")          # fmt="csv" pour un CSV
pairs = pd.read_parquet(path)
display(pairs["statut_paire"].value_counts())
# Exemple de jointure avec la table d'origine (colonnes réelles : voir config/schema.yaml) :
# source = pd.read_csv("imputation_source.csv", dtype=str)
# stats = pairs.merge(source.rename(columns={"<col paiement>": "payment_id", "<col facture>": "invoice_id"}),
#                     on=["payment_id", "invoice_id"], how="left")
pairs.head()"""),
    ("md", """## Profil de temps : d'où vient la durée d'un run

**Par étape et sous-étape**, toujours disponible, y compris après un redémarrage du noyau (relevés de
`memory.csv`, toutes sessions). Les sous-étapes portent le signal d'allocation, la règle, la phase du ML ou de
l'état. Les parties variables (numéro de bloc, nombre de paires, date) sont regroupées. Précision : l'intervalle
de relevé, une seconde."""),
    ("code", """profile = memory.time_profile(MEMORY_LOG)
display(profile.style.format({"minutes": "{:.1f}", "part de l'étape": "{:.0%}", "part du total": "{:.0%}"})
        .hide(axis="index"))
memory.time_chart(MEMORY_LOG)"""),
    ("md", """**Par fonction**, pour les étapes lancées avec `PROFILE = True`. Deux temps par fonction du projet :
- `inclus` : la fonction et tout ce qu'elle appelle ;
- `propre` : le temps où c'est la fonction du projet la plus profonde de la pile, appels pandas, numpy ou
  LightGBM compris. C'est là qu'il faut optimiser.

Les fichiers `reports/profiles/*.txt` (piles repliées) s'ouvrent aussi dans speedscope.app ou flamegraph.pl, et
`profiling.stack_summary(fichier)` les relit dans un nouveau noyau."""),
    ("code", """if not prof.runs:
    print("Profil par fonction désactivé : PROFILE = True dans la cellule des imports, puis relancer les étapes.")
for name, run in prof.runs.items():
    display(Markdown(f"**{name}** · {run['secondes'] / 60:.1f} min · {run['moteur']} · `{run['fichier']}`"))
    display(prof.summary(name).style.format({"propre (s)": "{:.0f}", "inclus (s)": "{:.0f}", "part propre": "{:.0%}"})
            .hide(axis="index"))"""),
    ("md", """## Analyse mémoire

Pic par étape et sous-étape de la dernière session, et courbe dans le temps (limite du pod en rouge).
Après un arrêt du noyau, cette cellule fonctionne seule : il suffit d'avoir exécuté la cellule des imports.
`memory.by_phase(MEMORY_LOG, session=None)` couvre toutes les sessions."""),
    ("code", """display(memory.by_phase(MEMORY_LOG))
memory.chart(MEMORY_LOG)"""),
]


# Version portable (single/) : mêmes cellules, imports et chemins de reconciliation.py, données réelles.
PORTABLE = [
    ("Chaque étape du brief est une méthode de `Project` (`src/api.py`)",
     "Chaque étape du brief est une méthode de `Project` (`reconciliation.py`)"),
    ("""  éditables à la main, depuis l'interface (`streamlit run src/ui/app.py`) ou depuis ce notebook.
- **Données réelles** : renseigner `config/schema.yaml`, puis `Project(dataset="real")`.
- **Données synthétiques** : `Project(dataset="synthetic")` — volume réglable (réel : ~2 M de paiements ;
  commencer par 50 000 pour une exécution de quelques minutes).""",
     """  éditables à la main, depuis l'interface (`streamlit run reconciliation_ui.py`) ou depuis ce notebook.
- **Données** : renseigner `config/schema.yaml` (sources réelles), puis `Project()`."""),
    ('ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()',
     "ROOT = Path.cwd()                     # dossier contenant reconciliation.py"),
    ("from src import memory, profiling, progress\nfrom src.api import Project",
     "from reconciliation import Project, memory, profiling, progress"),
    ("N_PAYMENTS = 50_000          # 2_000_000 pour le volume réel (≈ 1 h de bout en bout)\n", ""),
    ('Project(dataset="synthetic"', 'Project(dataset="real"'),
    ("import src.allocation.allocator as _allocation\nimport src.reconcile_ml.features as _features\n"
     "import src.reconcile_ml.pipeline as _pipeline",
     "import reconciliation as _allocation\nimport reconciliation as _features\nimport reconciliation as _pipeline"),
    ("from src.settings import save_settings", "from reconciliation import save_settings"),
    ("project.load(n_payments=N_PAYMENTS)", "project.load()"),
]


def portable(cells: list[tuple[str, str]]) -> list[tuple[str, str]]:
    out, used = [], set()
    for kind, text in cells:
        for i, (old, new) in enumerate(PORTABLE):
            if old in text:
                text = text.replace(old, new)
                used.add(i)
        out.append((kind, text))
    missing = [PORTABLE[i][0][:60] for i in range(len(PORTABLE)) if i not in used]
    assert not missing, f"substitutions portables sans effet : {missing}"
    return out


def build(cells: list[tuple[str, str]]) -> dict:
    out = []
    for kind, text in cells:
        lines = text.splitlines(keepends=True)
        if kind == "md":
            out.append({"cell_type": "markdown", "metadata": {}, "source": lines})
        else:
            out.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines})
    return {"cells": out, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}}}


def main() -> None:
    for path, cells in ((ROOT / "notebooks" / "pipeline.ipynb", CELLS),
                        (ROOT / "single" / "pipeline.ipynb", portable(CELLS))):
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(build(cells), indent=1, ensure_ascii=False), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
