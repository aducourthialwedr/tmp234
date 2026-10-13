"""Génère notebooks/deep_dive.ipynb : visite technique guidée du moteur, en direct sur les données
(outil de développement, hors livrable).

    python tools/make_deep_dive_notebook.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CELLS: list[tuple[str, str]] = [
    ("md", """# Deep dive technique — le moteur, ouvert étape par étape

Ce notebook sert de support à une revue technique de la solution. Chaque section appelle **les vrais objets
du code** sur les données chargées, et renvoie au fichier où regarder. On suit un jour rejoué et un paiement,
de la donnée source jusqu'à la décision, puis on mesure et on vérifie les garanties.

| § | Thème | Code |
|---|---|---|
| 1 | Données canoniques et normalisation | `src/load/` |
| 2 | Journal d'événements | `src/load/events.py` |
| 3 | État daté et boucle quotidienne : pas de fuite | `src/timeline/` |
| 4 | Allocation d'une journée | `src/allocation/` |
| 5 | Règles déterministes | `src/reconcile_rules/` |
| 6 | ML : candidats, features, score, propositions, décision | `src/reconcile_ml/` |
| 7 | Rejeu court et évaluation | `src/timeline/loop.py`, `src/evaluation/` |
| 8 | Démonstration anti-fuite | `src/timeline/state.py` |
| 9 | Coût et mémoire d'une journée | `src/memory.py` |

**Prérequis** : l'étape 1 faite (`project.load()`) et, pour le §6, un modèle entraîné (`project.train()`).
Pour une démonstration rapide, le jeu synthétique à 50 000 paiements se charge et s'entraîne en quelques
minutes (cellule suivante, `PREPARE_DEMO = True`). Sur les données réelles, les §3 à §8 prennent plus de temps :
chaque construction d'état ou d'index lit toute l'historique.

Documentation de référence : `docs/codebase/` (architecture, flux, modèles, zones d'ombre)."""),
    ("code", """import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

import json
import time

import numpy as np
import pandas as pd
from IPython.display import Markdown, display

from src import memory, progress
from src.allocation.allocator import Allocator
from src.api import Project
from src.evaluation.metrics import evaluate, ground_truth
from src.load.interim import load_interim
from src.load.normalize import label_numbers, normalize_text, reference_keys
from src.reconcile_ml.decision import decide, propose
from src.reconcile_ml.model import PairModel
from src.reconcile_ml.pipeline import PipelineMatcher
from src.reconcile_rules.matcher import RulesMatcher
from src.timeline.loop import DailyIterator, run_replay
from src.timeline.state import LedgerState, TemporalError

pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 200)
progress.use_notebook()

DATASET = "synthetic"          # "real" pour les données réelles
PREPARE_DEMO = False           # True : génère et charge 50 000 paiements synthétiques, puis entraîne (≈ 5 min)
project = Project(dataset=DATASET, log=print)
settings, rules = project.settings, project.rules

if PREPARE_DEMO and DATASET == "synthetic":
    project.load(n_payments=50_000)
    project.train()"""),
    ("md", """## 1. Données canoniques et normalisation

Les sources sont relues depuis `data/interim/` : tables **canoniques** (noms internes du modèle,
`src/load/canonical.py`), montants en **centimes**, dates UTC, colonnes normalisées ajoutées au chargement
(`label_norm`, `label_numbers`, `*_reference_keys`…). `load_interim` vérifie l'empreinte du journal et garde les
colonnes de listes en Arrow (compactes)."""),
    ("code", """data, journal, meta = load_interim(project.interim_dir)
display(pd.Series(meta, name="journal_meta.json").drop(labels=["mapped_fields"], errors="ignore"))
display(pd.DataFrame({name: {"lignes": len(df), "colonnes": df.shape[1]} for name, df in data.tables.items()}).T)
pay, inv = data.tables["payment"], data.tables["invoice"]
display(pay[["payment_id", "value_date", "amount", "currency", "iban_debtor", "label", "label_norm"]].head(3))
display(inv[["invoice_id", "client_reference", "client_reference_keys", "initial_amount", "due_date", "debtor_id"]].head(3))"""),
    ("md", """**Normalisation en direct** (`src/load/normalize.py`). Un match de référence = une intersection non vide entre
les clés du libellé (`label_numbers`) et les clés de la référence (`reference_keys`) : variantes sans préfixe,
sans zéros de tête, inversions, concaténations de tokens voisins. Changer les deux chaînes pour expérimenter."""),
    ("code", """LABEL = "VIR SEPA Dupont Bâtiment fact. FA-2024-0012345"
REFERENCE = "FA-2024-0012345"

norm = normalize_text(LABEL)
keys_label, keys_ref = label_numbers(norm.split()), reference_keys(REFERENCE)
display(pd.Series({"label_norm": norm, "label_numbers": keys_label, "reference_keys": keys_ref,
                   "intersection (match)": sorted(set(keys_label) & set(keys_ref))}, name="normalisation"))"""),
    ("md", """## 2. Journal d'événements

Tout ce qui se passe est un événement daté : création de facture, arrivée de paiement, réception de client file,
imputation (avec le montant déduit de la chaîne des soldes), ouverture et fermeture de parties et de contrats.
Ordre déterministe : date, puis type (ce qui ouvre avant ce qui l'utilise), puis identifiant
(`build_journal`, `src/load/events.py`). Son empreinte SHA-256 est recopiée dans tous les artefacts."""),
    ("code", """display(journal.groupby("event_type").agg(événements=("seq", "size"), premier=("ts", "min"), dernier=("ts", "max")))
example_payment = journal.loc[journal["event_type"] == "IMPUTATION_APPLIED", "entity_id"].iloc[0]
display(journal[(journal["entity_id"] == example_payment) | (journal["related_id"] == example_payment)])"""),
    ("md", """## 3. État daté et boucle quotidienne

`LedgerState` (`src/timeline/state.py`) rejoue le journal **jour par jour** dans des tableaux numpy. Au jour D,
il ne contient que les événements **strictement antérieurs** à D. Toute lecture exige `as_of` égal à l'instant de
l'état ; il ne recule jamais. Le lot du jour (`DailyIterator`, `src/timeline/loop.py`) = paiements arrivés le jour D
\\+ **reliquat** des paiements non résolus des 60 derniers jours."""),
    ("code", """state = LedgerState(data, journal, settings.reconcile_ml.features.behavioral_window_days)
test_start, test_end = project._period(state, "test")
DAY = test_start + pd.Timedelta(days=7)               # un jour de la période de test
ctx = next(iter(DailyIterator(state, DAY.date(), DAY.date(), settings.split.retention_days)))

inv_pos, balance = state.open_invoice_positions(ctx.as_of)
print(f"Jour {DAY.date()} · état au {state.as_of} · factures ouvertes : {len(inv_pos):,}".replace(",", " "))
print(f"Lot : {len(ctx.batch)} paiements dont {int(ctx.batch['is_new'].sum())} nouveaux "
      f"et {int((~ctx.batch['is_new']).sum())} en reliquat")
try:
    state.open_invoice_positions(ctx.as_of + pd.Timedelta(days=1))
except TemporalError as exc:
    print("Lecture à une autre date refusée :", exc)
display(ctx.batch[["payment_id", "value_date", "amount", "label", "is_new", "days_pending"]].head())"""),
    ("md", """## 4. Allocation d'une journée

Pour chaque paiement du lot, une liste classée de **débiteurs candidats** (`Allocator.allocate`,
`src/allocation/allocator.py`). Cinq signaux, du plus fort au plus faible : client file (1,0), référence (0,95),
IBAN direct (0,9), nom (0,7), montant exact (0,5), combinés par « ou » probabiliste 1 − Π(1 − sᵢ). L'allocation
est **ferme** si un seul débiteur porte un signal fort. Les index sont construits une fois ; tout ce qui dépend
du temps est filtré à la date du jour."""),
    ("code", """t = time.perf_counter()
allocator = Allocator(state, settings.allocation)
print(f"index construits en {time.perf_counter() - t:.1f} s")
t = time.perf_counter()
alloc = allocator.allocate(ctx)
print(f"allocation du lot en {time.perf_counter() - t:.1f} s")
display(alloc.payments["status"].value_counts().rename("paiements").to_frame())
display(alloc.payments["iban_route"].value_counts().rename("paiements").to_frame())

PAYMENT = alloc.payments.loc[alloc.payments["status"] == "multiple", "payment_id"].iloc[0]   # un paiement ambigu
display(Markdown(f"**Paiement suivi : `{PAYMENT}`**"))
display(ctx.batch.loc[ctx.batch["payment_id"] == PAYMENT, ["payment_id", "amount", "label", "iban_debtor"]])
display(alloc.candidates[alloc.candidates["payment_id"] == PAYMENT])"""),
    ("md", """## 5. Règles déterministes (étape 4)

R1 à R5 (`config/rules.yaml`, `src/reconcile_rules/matcher.py`), appliquées par priorité ; validation **seulement si
la solution est unique**. Arbitrage du lot : une facture disputée est servie par priorité. Chaque
auto-validation **réserve** son montant jusqu'à l'imputation réelle. Ici on exécute l'étape 4 d'un rapprocheur
complet, pour réutiliser ensuite exactement le même état au §6."""),
    ("code", """model_dir = project.model_dir
has_model = (model_dir / "model.json").exists()
model = PairModel.load(model_dir) if has_model else None
matcher = PipelineMatcher(state, settings, rules, model) if has_model else RulesMatcher(state, settings, rules)

rules_decisions = RulesMatcher.process(matcher, ctx)                 # étape 4 seule
decided = rules_decisions.drop_duplicates("payment_id")
print(f"{decided['payment_id'].nunique()} paiements résolus par les règles sur {len(ctx.batch)}")
display(decided["rule_id"].value_counts().rename("paiements").to_frame())
display(rules_decisions.groupby("rule_id").head(1))
print("Le paiement suivi est-il résolu par les règles ?", PAYMENT in set(rules_decisions["payment_id"]))"""),
    ("md", """## 6. ML sur le résiduel (étape 5)

Les mêmes étapes que `PipelineMatcher.process`, déroulées une à une :
1. **candidats** : factures des débiteurs alloués, référence citée, montant exact, client file ; filtres et plafond ;
2. **features** : 54 colonnes par paire (montant, temporel, textuel, identité, comportement, contrat,
   allocation, client file) ;
3. **score** : LightGBM passe 1 → features de compétition (rang, marge) → passe 2 ;
4. **propositions** : facture seule (`single` / `partial`) ou ensemble (`set`, DFS borné), avec une marge ;
5. **décision** : auto si score ≥ τ_high du type et marge ≥ δ ; revue si score ≥ τ_low ; sinon attente."""),
    ("code", """if not has_model:
    print("Aucun modèle : lancer project.train() (ou PREPARE_DEMO = True) pour cette section.")
else:
    batch_ids, pos, alloc_ml, cands, X = matcher._residual(ctx, rules_decisions)
    raw = model.raw(X, cands["row"].to_numpy())
    scored = cands.assign(p=raw, p_cal=model.calibrate(raw), invoice_id=matcher._inv_ids[cands["inv"].to_numpy()],
                          payment_id=batch_ids[cands["row"].to_numpy()])
    props = propose(scored, matcher._pay_amount[pos], settings.reconcile_ml.sets)
    actions = decide(props, matcher.thresholds) if len(props) else np.array([], dtype=object)
    props = props.assign(action=actions, payment_id=batch_ids[props["row"].to_numpy()])
    print(f"résiduel : {cands['row'].nunique()} paiements · {len(cands)} paires candidates · {X.shape[1]} features")
    display(props["action"].replace("", "attente").value_counts().rename("paiements").to_frame())
    display(pd.Series(matcher.thresholds["kinds"], name="τ_high par type"))"""),
    ("md", """**Le paiement suivi, vu par le ML** : ses candidats triés par score, la proposition retenue, et les
features qui ont le plus pesé pour le meilleur candidat (contributions SHAP natives de LightGBM, passe 1)."""),
    ("code", """if has_model:
    follow = PAYMENT if PAYMENT in set(scored["payment_id"]) else scored["payment_id"].iloc[0]
    display(Markdown(f"**Paiement : `{follow}`**"))
    view = scored[scored["payment_id"] == follow].sort_values("p", ascending=False)
    cols = ["invoice_id", "balance", "p", "p_cal", "src", "amount_diff_rel", "ref_key_in_label", "alloc_rank", "days_to_due"]
    display(view.join(X[["amount_diff_rel", "ref_key_in_label", "alloc_rank", "days_to_due"]])[cols].head(10))
    display(props[props["payment_id"] == follow])
    best = view.index[0]
    contrib = model.pass1.predict(X.loc[[best], model.features], pred_contrib=True)[0][:-1]
    top = pd.Series(contrib, index=model.features).sort_values(key=np.abs, ascending=False).head(10)
    display(pd.DataFrame({"contribution (log-odds)": top, "valeur": X.loc[best, top.index]}))"""),
    ("md", """## 7. Rejeu court et évaluation

Le rapprocheur complet sur quelques jours (`run_replay`), puis l'évaluation contre les imputations réelles
(`evaluate`, `src/evaluation/metrics.py`) : un paiement auto-validé est correct si l'ensemble de factures décidé
est **exactement** celui réellement imputé. Périmètre = paiements arrivés pendant la période rejouée.
Pour un backtest complet : `project.backtest("test", "pipeline")`."""),
    ("code", """N_DAYS = 7
replay_state = LedgerState(data, journal, settings.reconcile_ml.features.behavioral_window_days)
replay_matcher = PipelineMatcher(replay_state, settings, rules, model) if has_model else RulesMatcher(replay_state, settings, rules)
start, end = test_start.date(), (test_start + pd.Timedelta(days=N_DAYS - 1)).date()
result = run_replay(replay_state, replay_matcher, start, end, settings.split.retention_days, label=f"rejeu {N_DAYS} jours")
display(result.daily)

arrivals = journal[(journal["event_type"] == "PAYMENT_RECEIVED") & (journal["ts"] >= pd.Timestamp(start))
                   & (journal["ts"] < pd.Timestamp(end) + pd.Timedelta(days=1))]
scope = pd.DataFrame({"payment_id": arrivals["entity_id"].to_numpy(), "arrival_day": arrivals["ts"].dt.normalize().to_numpy()})
ev = evaluate(result.decisions, ground_truth(data.tables["imputation"]), scope,
              settings.evaluation.target_precision, settings.evaluation.current_automation_rate)
display(pd.Series(ev.summary, name="résumé").to_frame())
display(ev.cascade)
display(ev.by_rule)"""),
    ("md", """## 8. Démonstration anti-fuite

On supprime du journal **tout ce qui suit le jour D** et on recalcule l'allocation de D : le résultat doit être
identique. C'est la propriété vérifiée par les tests `test_removing_events_after_d_changes_nothing_computed_up_to_d`,
`test_allocation_does_not_leak` et `test_pipeline_runs_without_leak`. Coûteux sur données réelles (deux états,
deux jeux d'index)."""),
    ("code", """cutoff = DAY + pd.Timedelta(days=1)
truncated = journal[journal["ts"] < cutoff].reset_index(drop=True)
print(f"journal complet : {len(journal):,} événements · tronqué après {DAY.date()} : {len(truncated):,}".replace(",", " "))


def allocation_of_day(j: pd.DataFrame) -> pd.DataFrame:
    s = LedgerState(data, j, settings.reconcile_ml.features.behavioral_window_days)
    c = next(iter(DailyIterator(s, DAY.date(), DAY.date(), settings.split.retention_days)))
    a = Allocator(s, settings.allocation).allocate(c)
    return a.candidates.sort_values(["payment_id", "rank"]).reset_index(drop=True)


full_alloc, cut_alloc = allocation_of_day(journal), allocation_of_day(truncated)
pd.testing.assert_frame_equal(full_alloc, cut_alloc)
print(f"allocation du {DAY.date()} identique avec et sans le futur : {len(full_alloc)} candidats comparés")"""),
    ("md", """## 9. Coût et mémoire d'une journée

Le code est écrit pour environ 2 M de paiements par an : tout est vectorisé, les jointures sont bornées
(jamais « lot × toutes les factures d'une clé fréquente ou d'un gros débiteur »), les lots sont traités par
blocs. Les tailles de blocs se règlent sans changer les résultats (tests d'équivalence). Le moniteur relève la
mémoire du processus et du pod, avec la phase en cours."""),
    ("code", """import src.allocation.allocator as allocation_module
import src.reconcile_ml.features as features_module
import src.reconcile_ml.pipeline as pipeline_module

display(pd.Series({
    "CHUNK_ROWS (paiements par bloc d'allocation)": allocation_module.CHUNK_ROWS,
    "NAME_PAIR_BUDGET (paires du signal nom)": allocation_module.NAME_PAIR_BUDGET,
    "CF_PAIR_BUDGET (couples paiement / client file)": allocation_module.CF_PAIR_BUDGET,
    "CANDIDATE_PAIR_BUDGET (paires de candidats ML)": features_module.CANDIDATE_PAIR_BUDGET,
    "FEATURE_BLOCK_PAIRS (paires par bloc de features)": pipeline_module.FEATURE_BLOCK_PAIRS,
    "CPU utilisables (quota du pod)": memory.available_cpus(),
}, name="réglage"))
display(result.daily[["day", "batch", "new", "carried", "auto", "review", "seconds"]])
print(f"mémoire du processus : {memory.process_rss() / 2**30:.2f} Go · pod : {memory.pod_memory()}")"""),
    ("md", """## Où regarder dans le code

| Question | Fichier | Point d'entrée |
|---|---|---|
| Comment lancer chaque étape ? | `src/api.py` | `Project` |
| Comment une source réelle est-elle mappée ? | `config/schema.yaml`, `src/load/loader.py` | `load_all` |
| Pourquoi aucune fuite n'est possible ? | `src/timeline/state.py`, `src/timeline/loop.py` | `LedgerState._check`, `_validate` |
| Comment un paiement trouve-t-il ses débiteurs ? | `src/allocation/allocator.py` | `Allocator.allocate` |
| Comment ajouter une règle ? | `docs/ajouter_une_regle.md` | `RulesMatcher.process` |
| Comment le ML décide-t-il ? | `src/reconcile_ml/` | `PipelineMatcher.process`, `propose`, `decide` |
| Pourquoi ces choix ML ? | `docs/choix_etape_ml.md` | — |
| Comment lire les résultats ? | `RESULTATS.md` | `evaluate` |
| Que reste-t-il de fragile ? | `docs/codebase/08_zones_dombre_et_dette.md` | — |"""),
]


def build() -> dict:
    cells = []
    for kind, text in CELLS:
        lines = text.splitlines(keepends=True)
        if kind == "md":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": lines})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines})
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}}}


def main() -> None:
    out = ROOT / "notebooks" / "deep_dive.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(build(), indent=1, ensure_ascii=False), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
