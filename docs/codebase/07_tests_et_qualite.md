# 07 — Tests et qualité

> **Résumé.** 16 fichiers pytest, 155 fonctions de test (166 cas à l'exécution, en environ 2 minutes).
> Ils couvrent surtout les **garanties du moteur** :
> - absence de fuite temporelle (troncature du journal) ;
> - déterminisme ;
> - exactitude des conversions et de la normalisation ;
> - règles sur des cas figés ;
> - équivalence des traitements par blocs (mémoire) avec un passage unique.
>
> Les données de test sont construites à la main ou générées avec une seed fixe (5 000 paiements),
> jamais réelles, comme le demande le brief (`CLAUDE.md:231`). Le dépôt n'a ni CI, ni linter, ni
> mesure de couverture.

---

## 1. Lancer les tests

```bash
python -m pytest                           # tout : environ 2 minutes
python -m pytest tests/test_ml.py -q       # un fichier (test_ml : environ 1 minute, entraîne un petit modèle)
python -m pytest -k leak                   # par mot-clé
```

- **Configuration** : `pyproject.toml:22-24` (`testpaths = ["tests"]`, `pythonpath = ["."]`).
- **Dépendances** : extra `dev` (pytest). `tests/test_ui.py` exige streamlit, installé avec l'extra
  `ui`.
- **Avertissement connu** : un `RuntimeWarning: invalid value encountered in cast` (voir
  [08](08_zones_dombre_et_dette.md)).

### Données de test

- **`tests/conftest.py`** :
  - `synthetic_dir` (portée session) génère un jeu synthétique de 5 000 paiements, seed 7
    (`SMALL`, `tests/conftest.py:14`) ;
  - `make_data(**rows)` (`tests/conftest.py:43`) construit un `LoadedData` typé à partir de lignes
    écrites à la main.
- **`tests/test_timeline.py`** : `enriched`, `ledger`, `truncate`, réutilisés par d'autres fichiers
  (`tests/test_allocation.py:19`).

---

## 2. Ce qui est testé, fichier par fichier

| Fichier | Tests | Ce qui est vérifié |
|---|---|---|
| `test_loader.py` | 14 | centimes depuis texte et numérique, rejet des valeurs invalides ; gabarit de schéma ; erreurs de mapping (colonne requise, colonne absente du fichier) ; noms réels ; types et conventions du jeu synthétique ; champ et table optionnels ; conversion UTC ; consolidation des parties multi-lignes (fermeture seulement si toutes fermées) |
| `test_normalize.py` | 15 | accents, ponctuation, espaces ; tokens ; variantes (préfixe, zéros, inversion) ; correspondances référence ↔ libellé (exacte, sans préfixe, inversion entre et dans les tokens, découpage par ponctuation) ; absence de faux positif ; déterminisme ; IBAN |
| `test_events.py` | 10 | montant imputé par chaîne de résidus ; départage des ex æquo ; fermetures en dernier ; `booking_date` ordonne ; journal déterministe et indépendant de l'ordre des lignes ; cohérence sur le synthétique ; contrôles qualité (anomalies, solde reconstruit ≠ audit) |
| `test_timeline.py` | 13 | visibilité des factures et client files ; imputation appliquée le lendemain ; `as_of` obligatoire, état qui ne recule pas ; fenêtre comportementale strictement antérieure ; lot = nouveaux + reliquat ; expiration ; auto-validation retirée du lot ; **`LeakError`** (facture future, paiement hors lot) ; **troncature du journal sans effet avant D** ; deux exécutions identiques ; encours conforme à une formule de référence |
| `test_split.py` | 6 | ancrage sur la fin, non-chevauchement, purge, `train_months` / `anchor_end`, avertissement, découpages impossibles, étiquetage des jours |
| `test_allocation.py` | 19 | chaque signal et son caractère ferme ou non ; routes IBAN ; facture future jamais utilisée ; client file visible seulement après réception ; signaux désactivés ; métriques ; **pas de fuite** ; déterminisme ; **rappel ≥ 99 % et précision ferme ≥ 99,5 % sur le synthétique** (`test_synthetic_recall`) ; second IBAN d'un débiteur ; vocabulaire à empreintes ; **blocs = passage unique** ; clé fréquente gardée seulement par montant exact ; IBAN partagé |
| `test_rules.py` | 8 | cas figés de non-régression (`test_frozen_case`, paramétré) ; règle et version tracées ; ex æquo de même règle rejetés ; réservations contre la double imputation et leur libération ; règle désactivée ; propositions de chaque règle seule ; `exact_subset` |
| `test_ml.py` | 12 | propositions `single`/`partial`/`set`, ambiguïté escompte contre ensemble exact ; décision par type ; calibration en ligne « premier franchissement » ; features de compétition ; apprentissage versionné (AUC de validation > 0,9) ; sauvegarde et rechargement ; **pipeline sans fuite** ; **candidats et features par blocs = passage unique** ; **reprise du jeu d'entraînement** et refus si les réglages changent |
| `test_evaluation.py` | 8 | types de groupes ; comptes, précision, taux ; première auto-validation retenue plutôt qu'une revue antérieure ; décisions vides ; courbe coupée seulement entre scores distincts ; rejeu oracle 100 % précis ; erreurs détectées ; rejeu vide à 0 |
| `test_export.py` | 1 | `decision_pairs` : statuts de paire, jours, scores, décision retenue, couples manqués, hors périmètre |
| `test_arrow_ops.py` | 4 | `isin` et `lookup` (valeurs manquantes, référence vide, dates) |
| `test_memory.py` | 4 | relevés écrits avec leurs phases, sessions séparées, marqueurs sans moniteur, `available_cpus` borné |
| `test_progress.py` | 2 | progression d'un rejeu et d'un entraînement LightGBM dans un journal ; tâches muettes sans afficheur |
| `test_synthetic.py` | 5 | générateur déterministe, volume atteint, vérité cohérente ; normalisation parallèle = séquentielle |
| `test_ui.py` | 14 | chaque page s'affiche (Streamlit `AppTest`) ; navigation ; configuration et sauvegardes ; **pages qui lancent les tâches** (chargement, rejeu, évaluation, mesure de l'allocation, baseline des règles), donc `Project.load`, `replay`, `evaluate`, `measure_allocation` et `backtest("rules")` de bout en bout sur un petit jeu |

### Les tests qui portent les garanties du brief

| Garantie (`CLAUDE.md`) | Test |
|---|---|
| Supprimer les événements après D ne change rien avant D (§4.3) | `test_removing_events_after_d_changes_nothing_computed_up_to_d` (`tests/test_timeline.py:231`), `test_allocation_does_not_leak` (`tests/test_allocation.py:202`), `test_pipeline_runs_without_leak` (`tests/test_ml.py:134`) |
| Deux exécutions donnent le même résultat (§4.3) | `test_two_runs_on_same_journal_are_identical` (`tests/test_timeline.py:245`), `test_allocation_is_deterministic` (`tests/test_allocation.py:213`) |
| Rappel d'allocation ≥ 99 % (§5) | `test_synthetic_recall` (`tests/test_allocation.py:218`), sur le synthétique seulement |
| Non-régression des règles sur un jeu figé (§6) | `test_frozen_case` (`tests/test_rules.py:73`) |
| Normalisation déterministe (§3.3) | `test_label_numbers_sorted_and_deterministic` (`tests/test_normalize.py:84`) |

---

## 3. Ce qui n'est pas testé

| Zone | Constat |
|---|---|
| `Project.train` et `Project.calibrate` (`recalibrate_ml`) | `fit_ml` est testé directement (`tests/test_ml.py`), mais `recalibrate_ml` n'est appelé par aucun test |
| `Project.backtest("test", "pipeline")` et l'évaluation du rapprocheur complet | `PipelineMatcher` est testé dans un rejeu, mais `evaluate` n'est testé qu'avec les rapprocheurs `null` et `rules` (via l'interface), et avec des décisions construites |
| `Project.export_decisions` | seul `decision_pairs` est testé, pas l'écriture du fichier |
| Paiements groupés n↔n (`_n_to_n`, `src/reconcile_ml/pipeline.py:205`) | aucun test dédié |
| `near_subsets` | testé indirectement par les tests de `propose`, pas directement |
| `NotebookReporter`, `MemoryMonitor.live`, `chart`, `by_phase` sur une vraie session | non testés (nécessitent Jupyter ou altair) |
| Lecture des cgroups (`pod_memory`, quota de `available_cpus`) | non testée : l'environnement de test n'est pas un conteneur |
| Résultat de LightGBM selon le nombre de threads | non testé |
| `render_markdown`, `build_profile`, `write_reports` | exécutés via l'interface, sans assertion sur le contenu |
| `tools/make_notebook.py`, `tools/build_bundle.py` | non testés |
| **Version portable `single/`** | **aucun test** : elle est générée par un script hors dépôt, et sa cohérence avec `src/` n'est vérifiée par rien dans le dépôt |
| Comportement sur données réelles | par construction : les tests n'utilisent que du synthétique. Les tests de charge mémoire (gros débiteurs, références et montants fréquents) ont été menés avec des scripts hors dépôt ; le dépôt n'en garde que les tests d'équivalence par blocs |

---

## 4. Qualité du code : observations

- **Style homogène** : docstrings en français, commentaires qui expliquent le *pourquoi*,
  type hints, `from __future__ import annotations` partout.
- **Défense en profondeur sur les fuites** : contrôles à l'exécution (`TemporalError`,
  `LeakError`) **et** tests de troncature.
- **Versionnement** des artefacts :
  - `NORMALIZATION_VERSION` (`src/load/normalize.py:34`) ;
  - `FEATURIZATION_VERSION` (`src/reconcile_ml/features.py:25`) ;
  - version des règles (`config/rules.yaml`) ;
  - empreinte du journal ;
  - empreinte des réglages.

  Ces versions sont **incrémentées à la main** : rien n'empêche de modifier la featurisation sans
  changer `FEATURIZATION_VERSION`.
- **Pas d'outillage qualité** : ni linter, ni formateur, ni typage statique configurés, ni CI.
  Les tests sont lancés à la main.
- **Tests d'équivalence comme filet de sécurité des optimisations** : chaque découpage par blocs
  a un test qui force des blocs minuscules et compare au passage unique. C'est le bon réflexe pour
  toute optimisation future.
