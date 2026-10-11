# 06 — Configuration et exécution

> **Résumé.** Trois fichiers YAML pilotent tout :
> - `config/schema.yaml` : d'où viennent les données ;
> - `config/settings.yaml` : un bloc de réglages par étape ;
> - `config/rules.yaml` : les règles de l'étape 4.
>
> Ils sont validés par pydantic (`src/settings.py`) et modifiables à la main, depuis le notebook ou
> depuis l'interface. Il n'y a ni CLI, ni Docker, ni CI : on installe avec pip ou uv, puis on exécute
> le notebook, l'interface Streamlit ou les méthodes de `Project`. Quelques constantes de code
> (tailles de blocs mémoire) se règlent depuis le notebook.

---

## 1. Inventaire des configurations

| Source | Rôle | Lu par | Modifiable par |
|---|---|---|---|
| `config/schema.yaml` | mapping sources → modèle canonique | `Project.load` (`src/api.py:134`) → `load_all` | à la main ; page Configuration de l'interface (`src/load/schema_io.py`) |
| `config/schema.synthetic.yaml` | mapping du jeu synthétique | `Project._synthetic_source` (`src/api.py:120`) | à la main |
| `config/settings.yaml` | réglages de toutes les étapes | `load_settings` (`src/settings.py:313`), relu à chaque accès à `Project.settings` | à la main ; `save_settings` (`src/settings.py:318`) ; pages de l'interface |
| `config/rules.yaml` | règles R1–R5, version | `load_rules` (`src/settings.py:322`) | à la main ; `save_rules` ; page Règles |
| `.streamlit/config.toml` | Streamlit : écoute locale, pas de télémétrie | Streamlit | à la main |
| Variables d'environnement | §4 | | |
| Constantes de code | §5 | | notebook (cellule « Réglages mémoire ») |

L'écriture passe par `dump_model` (`src/yaml_io.py:59`) : le YAML est **régénéré avec les
descriptions pydantic en commentaires**. Une modification par l'interface ou `save_settings` réécrit
donc tout le fichier ; les commentaires ajoutés à la main sont perdus.

Validation : `extra="forbid"` (`src/settings.py:26`). Une clé inconnue ou mal orthographiée lève
une erreur au chargement.

---

## 2. `config/schema.yaml`

Décrit en détail dans [03_gestion_des_donnees.md](03_gestion_des_donnees.md) §1.1 : unité des
montants, séparateur décimal, fuseau, formats de date, `base_dir`, puis pour chaque table `source`,
`read_options`, `columns` et `value_maps`. Le gabarit du dépôt contient des `null` partout ;
`(requis)` marque les champs obligatoires. Exemple complet : `config/schema.synthetic.yaml`.

---

## 3. `config/settings.yaml` — référence des réglages

Valeurs **du fichier du dépôt**. Quand elles diffèrent du défaut du code, celui-ci est indiqué. Le
modèle pydantic de chaque bloc est entre parenthèses.

### 3.1 `paths` (`PathsSettings`, `src/settings.py:32`)

| Clé | Valeur | Effet |
|---|---|---|
| `interim_dir` | `data/interim` | tables normalisées, journal, décisions, jeu ML ; `synthetic/` ajouté pour le jeu synthétique |
| `reports_dir` | `reports` | rapports ; idem |
| `synthetic_dir` | `data/synthetic` | jeux synthétiques générés, un sous-dossier par volume et seed |
| `models_dir` | `models` | modèles, sous-dossier `synthetic/` ou `real/` |

### 3.2 `load` (`LoadSettings`, `src/settings.py:39`)

| Clé | Valeur | Effet |
|---|---|---|
| `workers` | 12 | processus de normalisation ; sans effet sur le résultat, multiplie la mémoire |

### 3.3 `split` (`SplitSettings`, `src/settings.py:47`)

| Clé | Valeur | Effet |
|---|---|---|
| `test_months` | 2 | durée du test, ancré sur la fin |
| `validation_months` | 2 | durée de la validation |
| `train_months` | `null` | `null` = tout l'historique restant |
| `purge_days` | 5 | jours exclus entre deux périodes |
| `anchor_end` | `null` | dernier jour du test ; `null` = dernier jour de paiement |
| `retention_days` | 60 | durée de vie d'un paiement non résolu dans le reliquat |

### 3.4 `allocation` (`AllocationSettings`, `src/settings.py:120`)

| Clé | Valeur | Effet |
|---|---|---|
| `signals.client_file.enabled` / `date_tolerance_days` | `true` / 5 | rattachement des client files ; écart de date toléré |
| `signals.reference.enabled` | `true` | signal référence |
| `signals.reference.min_key_length` | 4 | clés plus courtes ignorées (années, numéros de rue) |
| `signals.reference.strong_min_key_length` | 6 | longueur minimale d'une clé « forte » (allocation ferme) |
| `signals.reference.max_debtors_per_key` | 5 | une clé partagée par plus de débiteurs à D est ignorée |
| `signals.iban.enabled` / `max_debtors_per_iban` | `true` / 20 | routage IBAN ; un IBAN porté par plus de 20 débiteurs ne propose pas de candidats |
| `signals.name.enabled` | `true` | signal nom |
| `signals.name.min_similarity` | 0,5 | score de nom minimal |
| `signals.name.max_token_share` / `max_debtors_per_term` | 0,01 / 200 | élagage des mots trop fréquents |
| `signals.name.min_token_length` | 3 | longueur minimale d'un mot indexé |
| `signals.amount.enabled` | `true` | signal montant exact |
| `signals.amount.max_debtors_per_amount` | 3 | au-delà, montant ignoré… |
| `signals.amount.max_debtors_with_name_hint` | 30 | … sauf si un mot du nom figure au libellé, jusqu'à 30 débiteurs |
| `max_candidates` | 10 | taille de la liste de débiteurs transmise |
| `target_recall` | 0,99 | cible affichée par `measure_allocation` (`cible_atteinte`) |

### 3.5 `reconcile_ml` (`MLSettings`, `src/settings.py:208`)

| Clé | Valeur | Effet |
|---|---|---|
| `candidates.allocated_debtors` | `true` | source « factures des débiteurs alloués » |
| `candidates.max_debtors` | 3 | seulement les 3 premiers débiteurs de l'allocation |
| `candidates.debtor_window_before_days` / `after_days` | 180 / 30 | fenêtre d'échéance des factures du débiteur |
| `candidates.reference_no_window` | `true` | source « référence dans le libellé » |
| `candidates.amount_exact` / `amount_window_days` | `true` / 90 | source « montant exact », factures créées à ±90 jours |
| `candidates.client_file_cited` | `true` | source « factures citées par le client file » |
| `candidates.max_per_payment` | **50** (60 par défaut dans le code) | plafond de candidats par paiement |
| `features.<famille>` | `true` ×8 | familles actives : `amount`, `temporal`, `textual`, `identity`, `behavioral`, `contract`, `allocation`, `client_file` |
| `features.behavioral_window_days` | 180 | fenêtre des agrégats du débiteur ; **sert aussi** à construire tout `LedgerState` (`src/api.py:180-185`) |
| `second_pass` / `calibration` | `true` / `true` | passe 2 ; calibration isotonique |
| `training.payment_sample` | 0,5 | part des paiements du résiduel gardés pour l'apprentissage |
| `training.num_boost_round` / `learning_rate` / `num_leaves` / `min_data_in_leaf` / `seed` | 400 / 0,05 / 63 / 50 / 42 | LightGBM |
| `training.num_threads` | 0 | 0 = CPU utilisables du pod (quota cgroup) |
| `sets.enabled` / `near_sum` / `single_candidate` / `n_to_n` | `true` | passes de résolution des ensembles |
| `sets.n_to_n_window_hours` | 72 | fenêtre des paiements groupés n↔n (convertie en jours : 3) |
| `sets.max_invoices` / `max_candidates` / `node_budget` | 5 / 25 / 100 000 | bornes du DFS |
| `sets.tolerance_abs_cents` / `tolerance_rel` | 500 / 0,03 | tolérance sur la somme : 5 € ou 3 % |
| `decision.review_min_score` | 0,3 | τ_low |
| `decision.min_margin` | 0,05 | δ |
| `decision.segmented_thresholds` / `segments` / `min_segment_volume` | `false` / … / 1000 | seuils par segment, **sans effet sur les décisions finales** (voir [08](08_zones_dombre_et_dette.md)) |
| `llm_labels.enabled` / `endpoint` / `cache_dir` | `false` / `null` / … | **non implémenté** |

### 3.6 `evaluation` (`EvaluationSettings`, `src/settings.py:223`)

| Clé | Valeur | Effet |
|---|---|---|
| `target_precision` | 0,995 | cible des seuils et de `taux_automatisation_à_précision_cible` |
| `current_automation_rate` | `null` | taux de l'algorithme actuel ; alimente la cascade. **À renseigner** |
| `reference_total_volume` / `reference_manual_volume` | `null` | lus nulle part dans le code |

---

## 4. Variables d'environnement

| Variable | Effet | Code |
|---|---|---|
| `RECON_SETTINGS`, `RECON_SCHEMA`, `RECON_RULES` | chemins des fichiers de configuration **pour l'interface** | `src/ui/common.py:89`, `:93`, `:97` |
| `PYTHONIOENCODING=utf-8`, `PYTHONUNBUFFERED=1` | posées par l'interface pour le sous-processus | `src/ui/common.py:241` |
| `OMP_NUM_THREADS` | non lue par le code ; limite les threads OpenMP si posée avant le démarrage | conseil donné dans le notebook |
| `MALLOC_ARENA_MAX` | non lue par le code ; limite la fragmentation glibc | conseil donné dans le notebook |

Il n'y a **aucun argument de ligne de commande** : la CLI a été retirée.

---

## 5. Constantes de code réglables depuis le notebook

Elles bornent la mémoire d'une journée **sans changer les résultats** (tests d'équivalence dans
[07](07_tests_et_qualite.md)) :

| Constante | Valeur | Rôle |
|---|---|---|
| `CHUNK_ROWS` | 20 000 | paiements par bloc d'allocation (`src/allocation/allocator.py`) |
| `NAME_PAIR_BUDGET` | 2 000 000 | paires (paiement, débiteur) du signal nom par paquet |
| `CF_PAIR_BUDGET` | 2 000 000 | couples (paiement, client file de même montant) par paquet |
| `CANDIDATE_PAIR_BUDGET` | 2 000 000 | paires examinées par bloc de candidats ML (`src/reconcile_ml/features.py`) |
| `FEATURE_BLOCK_PAIRS` | 500 000 | paires par bloc de calcul des features (`src/reconcile_ml/pipeline.py`) |

---

## 6. Installer, lancer, entraîner, tester

### 6.1 Installation (tirée de `README.md`)

```bash
pip install -r requirements.txt                 # pipeline + notebook + interface
# ou
uv sync --extra notebook --extra ui             # uv.lock fourni ; ajouter --extra dev pour pytest
```

Python ≥ 3.11 (`pyproject.toml`).

### 6.2 Notebook

```bash
jupyter lab                                     # puis notebooks/pipeline.ipynb
```

Après une modification de `tools/make_notebook.py`, régénérer le notebook avec
`python tools/make_notebook.py`.

### 6.3 API Python (ordre canonique)

```python
from src.api import Project
from src import memory, progress

monitor = memory.MemoryMonitor("reports/memory.csv").start()   # optionnel : suivi mémoire
progress.use_notebook()                                        # optionnel : barres de progression
project = Project(dataset="real", log=monitor.log)             # "synthetic" pour le jeu de test

project.load()                                   # 1 · sources de config/schema.yaml
project.split()                                  # 2 · périodes
project.replay("test", "null")                   #     taille des lots et du reliquat
project.measure_allocation("validation")         # 3 · rappel ≥ 99 %
project.backtest("test", "rules")                # 4 · baseline
project.train()                                  # 5 · modèle + seuils ; train(reuse_dataset=True) pour reprendre
project.calibrate()                              #     seuils seuls
project.backtest("test", "pipeline")             # 6 · évaluation complète
project.export_decisions("test", "pipeline")     #     couples proposés et manqués
project.report("pipeline", "test")               #     relecture sans recalcul
```

Jeu synthétique : `Project(dataset="synthetic").load(n_payments=50_000)` génère le jeu au besoin
(`seed=42`, `regenerate=False`).

### 6.4 Interface

```bash
python -m streamlit run src/ui/app.py
```

Les chemins de configuration peuvent être changés avec `RECON_SETTINGS`, `RECON_SCHEMA` et
`RECON_RULES`.

### 6.5 Tests

```bash
python -m pytest                                 # environ 2 minutes
python -m pytest tests/test_timeline.py -q       # un fichier
```

Voir [07_tests_et_qualite.md](07_tests_et_qualite.md).

### 6.6 Livrables

| Commande | Produit |
|---|---|
| `python tools/build_bundle.py` | `dist/reconciliation_poc.zip` : code, configuration, notebook, documentation ; sans tests ni données |
| *(script hors dépôt)* | `single/` : version portable en 4 fichiers + `RESULTATS.md` ; voir [outils_et_livrables.md](05_modules/outils_et_livrables.md) |

### 6.7 Durées et volumes indicatifs

Sur le jeu synthétique de 2 M de paiements, 16 cœurs (d'après `GUIDE.md` §8 et les métadonnées du
modèle synthétique) :

| Étape | Durée |
|---|---|
| Chargement | environ 2 min |
| Mesure de l'allocation sur 2 mois | 3 à 4 min |
| Backtest des règles | environ 3 min |
| Apprentissage complet | environ 1 h (jeu d'entraînement ≈ 29 min, LightGBM ≈ 6 min, calibration en ligne ≈ 25 min) |
| Backtest complet | environ 30 min |

---

## 7. Infrastructure

- **Docker, CI/CD, jobs planifiés** : aucun dans le dépôt.
- **Cible d'exécution** : un notebook JupyterLab sur un pod Kubernetes à mémoire et CPU limités. Le
  code s'y adapte : il lit le quota CPU et la mémoire du pod via les cgroups (`src/memory.py`) et
  ne lance pas plus de threads LightGBM que de CPU disponibles.
- **Réseau** : aucun appel sortant. Streamlit est configuré en écoute locale, sans télémétrie
  (`.streamlit/config.toml`).
- **Dépôt distant** : le code est poussé sur un dépôt GitHub de travail. La version portable sert
  aux environnements où seul le copier-coller est possible.
