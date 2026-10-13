# Outils et livrables — `tools/`, `notebooks/`, `single/`, `dist/`

> **Résumé.** Il y a trois façons de transporter et d'exécuter le projet :
> 1. **le notebook** `notebooks/pipeline.ipynb`, généré par `tools/make_notebook.py` ;
> 2. **le zip** `dist/reconciliation_poc.zip`, produit par `tools/build_bundle.py` ;
> 3. **la version portable** `single/` : tout le code concaténé en quelques fichiers à copier-coller,
>    données réelles uniquement.
>
> Le script qui produit `single/` **n'est pas dans le dépôt** ; c'est le principal risque de dérive
> de ces livrables.

---

## 1. `tools/make_notebook.py` → `notebooks/pipeline.ipynb`

Le notebook n'est pas édité à la main. Il est **généré** à partir de la liste `CELLS`
(`tools/make_notebook.py:8`), en paires `("md" | "code", texte)`. `main()` (`tools/make_notebook.py:166`)
l'écrit au format nbformat 4, sans sorties.

Après modification de `CELLS`, lancer `python tools/make_notebook.py`.

| # | Type | Contenu |
|---|---|---|
| 1 | md | présentation : `Project`, paramètres, données réelles ou synthétiques |
| 2 | code | `sys.path`, imports (`altair`, `pandas`, `memory`, `progress`, `Project`), `MEMORY_LOG`, `N_PAYMENTS` |
| 3 | md | suivi de la mémoire et de la progression (pastille, CSV, analyse après un kill OOM, `MALLOC_ARENA_MAX`) |
| 4 | code | `MemoryMonitor(...).start()`, `monitor.live()`, `progress.use_notebook()`, `project = Project(dataset="synthetic", log=monitor.log)` |
| 5–6 | md, code | réglages mémoire : `CHUNK_ROWS`, `NAME_PAIR_BUDGET`, `CF_PAIR_BUDGET`, `CANDIDATE_PAIR_BUDGET`, `FEATURE_BLOCK_PAIRS` |
| 7–8 | md, code | lecture et modification des réglages (`save_settings`) |
| 9–10 | md, code | étape 1 : `project.load(n_payments=N_PAYMENTS)`, volumes, anomalies, champs manquants |
| 11–12 | md, code | étape 2 : `project.split()`, rejeu vide, graphique des lots |
| 13–14 | md, code | étape 3 : `project.measure_allocation("validation")` |
| 15–16 | md, code | étape 4 : `project.backtest("test", matcher="rules")` |
| 17–18 | md, code | étape 5 : `project.train()` (option `reuse_dataset=True`), métriques, seuils, importance des features |
| 19–22 | md, code ×3 | étape 6 : `project.backtest("test", matcher="pipeline")`, cascade, courbe (altair), tableaux et rapport markdown |
| 23–24 | md, code | export : `project.export_decisions("test", "pipeline")` et exemple de jointure avec la table d'origine |
| 25–26 | md, code | analyse mémoire : `memory.by_phase(MEMORY_LOG)`, `memory.chart(MEMORY_LOG)` ; utilisable après un redémarrage du noyau |

Le notebook se lance depuis la racine ou depuis `notebooks/`. La cellule 2 ajoute la racine du
dépôt à `sys.path`.

---

### 1 bis. `tools/make_ml_notebook.py` → `notebooks/exploration_ml.ipynb`

Même principe (liste `CELLS`), pour un notebook d'**exploration ML** : structure des candidats et sources, pouvoir de chaque feature (AUC, top-1 par paiement), distributions, redondances, contributions SHAP, ablation par famille, gabarit de nouvelle feature, analyse d'erreurs, calibration, propositions et seuils. Il lit le jeu `data/interim/ml/dataset/`, le modèle et `validation_daily_proposals.parquet`. Il est écrit aussi en `single/exploration_ml.ipynb` : un repli d'import (`from reconciliation import ...`) le rend utilisable dans la version portable.

### 1 ter. `tools/make_deep_dive_notebook.py` → `notebooks/deep_dive.ipynb`

Support de revue technique : données canoniques et normalisation, journal, état daté et lot du jour, allocation d'une journée, règles, ML déroulé étape par étape sur un paiement (candidats, score, proposition, décision, contributions SHAP), rejeu court et évaluation, démonstration anti-fuite (journal tronqué → allocation identique), coût et mémoire. Environ 8 minutes sur le synthétique à 2 M de paiements ; option `PREPARE_DEMO` pour un jeu de 50 000 paiements.

## 2. `tools/build_bundle.py` → `dist/reconciliation_poc.zip`

`main()` (`tools/build_bundle.py:28`) zippe sous le préfixe `reconciliation_poc/` :
- `INCLUDE` (`tools/build_bundle.py:13-14`) : `src`, `config`, `notebooks`, `.streamlit`,
  `pyproject.toml`, `requirements.txt`, `README.md`, `GUIDE.md`, `RESULTATS.md`, `CLAUDE.md`,
  `spec_rapprochement_automatique.md` ;
- sans les caches (`__pycache__`, `.pytest_cache`, `.ipynb_checkpoints`) ;
- avec des dossiers vides `data/`, `reports/`, `models/` (fichier `.keep`).

**Exclus** : tests, outils, données, rapports, modèles. Le zip contient le mode synthétique complet
(`src/synthetic/`), contrairement à `single/`.

---

## 3. `single/` — version portable

### 3.1 Pourquoi

Certains environnements cibles (pod sans accès git) ne permettent que le copier-coller. `single/`
regroupe tout dans un petit nombre de fichiers.

| Fichier | Contenu |
|---|---|
| `reconciliation.py` | tout `src/` hors interface et générateur, concaténé (environ 7 000 lignes) |
| `reconciliation_ui.py` | l'interface : chaque vue devient une fonction `page_<vue>()`, plus `main()` |
| `pipeline.ipynb` | le notebook, adapté (`from reconciliation import ...`, `Project(dataset="real")`, `project.load()`) |
| `GUIDE.md`, `RESULTATS.md` | la documentation, avec un avertissement « version portable » ; les liens vers `src/` sont retirés |

### 3.2 Structure de `reconciliation.py`

- **En-tête** : docstring d'usage, puis tous les imports externes dédoublonnés.
- **Une section par module**, précédée d'une bannière `# src/<module>.py`, dans l'ordre des
  dépendances. Exemples dans le fichier actuel : `src/config.py` ligne 61, `src/memory.py` 563,
  `src/load/loader.py` 1500, `src/timeline/state.py` 2407, `src/allocation/allocator.py` 3411,
  `src/reconcile_ml/pipeline.py` 5701, `src/api.py` 6175. Les imports internes `from src... import`
  sont supprimés : tout partage le même espace de noms.
- **Espaces de noms reconstitués.** Le code du dépôt appelle `memory.mark(...)` et
  `progress.task(...)`. La version portable crée donc, après les sections correspondantes,
  `memory = SimpleNamespace(...)` (ligne 902) et `progress = SimpleNamespace(...)` (ligne 1100).
- **Configuration embarquée** : `_EMBEDDED_FILES` (ligne 6521) contient `config/settings.yaml`,
  `config/rules.yaml`, `config/schema.yaml` et `.streamlit/config.toml`. Au premier import,
  `ensure_config()` (lignes 6953 et 6962) les écrit à côté du fichier **s'ils n'existent pas**,
  sans jamais écraser.
- **`REPO_ROOT`** = dossier du fichier (et non plus le parent de `src/`).

### 3.3 Différences avec le dépôt (commit `7dae7fd`)

- **Pas de mode synthétique.** Le code entre marqueurs `# <synthétique>` … `# </synthétique>`, et
  les lignes marquées `# synthétique seulement`, sont retirés. Les lignes marquées
  `# portable: <code>` sont remplacées par `<code>`. Exemples dans `src/api.py` : `DATASETS = (REAL,)`,
  `DEFAULT_DATASET = REAL`, `def load(self) -> dict:`. Dans `src/ui/common.py` : `dataset()` renvoie
  `REAL`.
- **Absents** : le générateur, `config/schema.synthetic.yaml`, le choix du jeu dans l'interface,
  `N_PAYMENTS` dans le notebook.

### 3.4 Vérifier une copie

Le copier-coller peut altérer le texte (fermeture automatique des parenthèses, auto-complétion). Une
empreinte SHA-256 du contenu normalisé (fins de ligne `\n`) permet de comparer une copie à
l'original. La procédure a été donnée en accompagnement des livraisons, mais elle n'est pas outillée
dans le dépôt.

---

## 4. `dist/`

Ignoré par git (`.gitignore:5`). Il contient le zip de `build_bundle.py` et, localement, d'anciennes
copies de `single/`. **Ne pas utiliser `dist/single/` sans le régénérer** : une copie périmée y a
déjà été prise pour la version à jour.

---

## 5. Tests

Aucun. Ni `tools/make_notebook.py`, ni `tools/build_bundle.py`, ni le contenu de `single/` ne sont
couverts par `tests/`. La version portable a été vérifiée **à la main** lors de sa production
(import dans un dossier vierge, chargement d'un petit jeu via `schema.yaml`, allocation et règles),
mais rien ne le rejoue.

---

## ⚠️ Zones d'ombre

1. **Générateur de `single/` hors dépôt.** `tools/` ne contient que `make_notebook.py` et
   `build_bundle.py`. Le script qui concatène `src/`, applique les marqueurs, réécrit les imports et
   embarque la configuration n'est pas versionné. Toute évolution de `src/` doit donc être
   reportée par ce script hors dépôt, sinon `single/` dérive sans que rien ne le signale.
2. **Les marqueurs n'ont pas d'effet dans le dépôt** : leur traitement n'existe que dans ce script.
   Un marqueur mal fermé ou une ligne `portable:` erronée ne serait détecté qu'à la génération.
3. **Le notebook versionné utilise le jeu synthétique** (`Project(dataset="synthetic")`, cellule 4).
   Pour les données réelles, il faut l'éditer, ou utiliser la version de `single/`.
4. **Le zip et `single/` divergent** : le zip garde le mode synthétique et la structure `src/` ; `single/`
   non. Deux livrables, deux comportements.
5. **Pas d'outil de vérification d'intégrité** d'une copie dans le dépôt (empreintes, contrôle par
   blocs).
