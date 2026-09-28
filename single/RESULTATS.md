# Comment les résultats sont produits

Ce document explique, pour chaque rapport de `reports/` et chaque métrique renvoyée par le notebook,
**ce qui est compté, sur quelle population, avec quel dénominateur, et par quel code**. `GUIDE.md`
explique le fonctionnement du moteur ; ce document explique comment lire ce qu'il mesure.

---

## 1. Les principes communs à tous les résultats

### 1.1 Rejeu de l'historique, jour par jour

Aucun résultat n'est calculé « à plat » sur la table des paiements. Chaque mesure vient d'un
**rejeu** de l'historique :
- pour chaque jour D de la période, le moteur voit l'état tel qu'il était la veille à minuit
  (factures ouvertes, restants dus, client files reçus…) ;
- il traite le **lot du jour** : les paiements arrivés ce jour-là, **plus le reliquat**, c'est-à-dire les
  paiements des 60 derniers jours (`split.retention_days`) ni imputés en réalité, ni auto-validés par le
  moteur ;
- ses décisions sont enregistrées, puis l'état avance d'un jour.

Code : `run_replay` et `DailyIterator` dans src/timeline/loop.py. Un même
paiement peut donc être examiné plusieurs jours de suite tant qu'il reste dans le reliquat.

### 1.2 Les périodes

L'historique est coupé en trois périodes consécutives (`split` dans `config/settings.yaml`), séparées
par quelques jours de purge :

| Période | Sert à |
|---|---|
| train (~8 mois) | construire le jeu d'entraînement du ML |
| validation (~2 mois) | mesurer l'allocation, choisir les seuils du ML |
| test (2 derniers mois) | **mesure finale** : jamais vue par l'apprentissage ni par le choix des seuils |

### 1.3 La vérité : les imputations réelles

La table `imputation` dit quelles factures ont réellement été imputées à chaque paiement, par
l'algorithme actuel ou à la main. C'est la seule référence. Pour chaque paiement
(`ground_truth`) :
- **`truth_invoices`** : l'ensemble des factures qui lui ont été imputées, **toutes dates confondues**
  (une imputation prononcée après la fin de la période compte aussi) ;
- **`group_type`** : on relie paiements et factures qui partagent des imputations et on regarde le
  groupe obtenu :

  | Type | Signification |
  |---|---|
  | 1↔1 | un paiement, une facture |
  | 1↔n | un paiement règle plusieurs factures |
  | n↔1 | une facture réglée par plusieurs paiements |
  | n↔n | plusieurs paiements et plusieurs factures emmêlés |
  | sans imputation | le paiement n'a jamais été imputé |

### 1.4 Unité de mesure : le paiement, jugé sur l'ensemble exact

Sauf mention contraire (tableaux marqués « paires »), **on compte des paiements**.

Une auto-validation est **correcte** si l'ensemble des factures décidé par le moteur est **exactement**
l'ensemble réellement imputé. Une facture en trop ou en moins rend la décision fausse. Un paiement
**sans imputation réelle** qui est auto-validé compte comme **une erreur**.

### 1.5 Le périmètre : les paiements arrivés pendant la période

Le **périmètre** d'une évaluation est l'ensemble des paiements **arrivés** pendant la période
(événement `PAYMENT_RECEIVED` du journal). Les paiements du reliquat de démarrage, arrivés avant le
début de la période, sont traités par le rejeu mais **exclus de la mesure**. C'est le dénominateur
`paiements_périmètre`.

### 1.6 Une décision par paiement

Un paiement peut recevoir plusieurs décisions au fil des jours. L'évaluation en retient une seule
(`_proposals`) : **la première auto-validation** s'il y en a une, sinon
**la première mise en revue**. Un paiement sans aucune des deux n'a pas de décision.

---

## 2. Qui produit quoi

Tous les rapports sont écrits dans `reports/` (sous-dossier `synthetic/` pour le jeu synthétique).
`<p>` désigne la période (`validation`, `test`…) et `<m>` le rapprocheur (`null`, `rules`, `pipeline`).

| Méthode de `Project` | Fichiers produits | Section |
|---|---|---|
| `load()` | `load_profile.md`, `load_volumes.csv`, `load_missing_fields.csv`, `load_date_ranges.csv`, `load_events.csv`, `load_issues.csv` | §3 |
| `replay(p, m)` | `replay_<m>_<p>.json`, `replay_<m>_<p>_daily.csv` ; décisions dans `data/interim/replay/` | §4 |
| `measure_allocation(p)` | `allocation_<p>.json`, `allocation_<p>_by_route.csv`, `_by_status.csv`, `_found_by.csv`, `_misses.csv` | §5 |
| `evaluate(p, m)` | `evaluation_<m>_<p>.json`, `.md` et un CSV par tableau | §6 |
| `backtest(p, m)` | `replay` puis `evaluate` | §4, §6 |
| `train()` | `models/<jeu>/pair_model/` : `model.json`, `feature_importance.csv`, `calibration_validation.csv` ; jeu dans `data/interim/ml/` | §7 |
| `report(m, p)` | relit une évaluation déjà calculée, sans recalcul | — |

---

## 3. Étape 1 — rapports de chargement

Produits par src/load/profile.py et src/load/quality.py
sur les tables telles que chargées, avant tout rejeu.

| Fichier | Contenu |
|---|---|
| `load_volumes.csv` | nombre de lignes par table (dont `party_iban`, les couples partie-IBAN consolidés) |
| `load_missing_fields.csv` | pour chaque champ du modèle : obligatoire ou non, renseigné dans `schema.yaml` ou non, nombre et part de valeurs vides |
| `load_date_ranges.csv` | date minimale et maximale de chaque champ date |
| `load_events.csv` | nombre d'événements du journal par type, premier et dernier horodatage |
| `load_issues.csv` | anomalies détectées, une ligne par type et par table, avec leur nombre : clés dupliquées (`cle_dupliquee`), valeurs illisibles (`valeur_invalide`), montants nuls ou négatifs, dates incohérentes (`echeance_avant_creation`, `fermeture_avant_ouverture`…), imputations incohérentes (`full_avec_residu_non_nul`, `avant_creation_facture`, `solde_reconstruit_different`…), orphelins, parties sur plusieurs lignes (`lignes_multiples_par_partie`, avec les colonnes qui varient) |
| `load_profile.md` | le tout en un document |

---

## 4. Rejeu — `replay_<m>_<p>_daily.csv`

Une ligne par jour rejoué (`run_replay`) :

| Colonne | Signification |
|---|---|
| `batch` | taille du lot du jour (nouveaux + reliquat) |
| `new` | paiements arrivés ce jour |
| `carried` | paiements repris du reliquat |
| `left_imputed` | paiements sortis du reliquat parce qu'imputés en réalité |
| `left_engine` | paiements sortis parce qu'auto-validés par le moteur |
| `expired` | paiements sortis après 60 jours sans solution |
| `auto`, `review` | paiements auto-validés / mis en revue ce jour |
| `seconds` | durée de traitement du jour |

Avec le rapprocheur `null` (aucune décision), ce fichier montre la taille naturelle des lots et du
reliquat : c'est ce qui dimensionne le temps et la mémoire des étapes suivantes.

---

## 5. Étape 3 — allocation (`measure_allocation`)

Code : src/allocation/evaluate.py. La sonde rejoue la période en
n'appliquant **que l'allocation** (aucune décision). Elle note, pour chaque paiement, sa liste de
débiteurs candidats (10 au plus, `allocation.max_candidates`) :
- au **premier passage** : le jour de son arrivée ;
- au **dernier passage** : le dernier jour où il était encore dans le lot (un client file a pu arriver
  entre-temps).

**Vérité** : les débiteurs des factures réellement imputées au paiement.

**Périmètre** : les paiements arrivés pendant la période **et** ayant une imputation réelle
(`avec_imputation_réelle`) ; c'est le dénominateur de toutes les métriques ci-dessous.

| Métrique (`allocation_<p>.json`) | Calcul |
|---|---|
| `rappel_premier_passage` | part des paiements dont **tous** les vrais débiteurs sont dans la liste au premier passage. **Métrique clé, cible ≥ 99 %** (`cible_atteinte`) |
| `rappel_dernier_passage` | idem au dernier passage |
| `top1` | paiements à un seul vrai débiteur, classé premier / tous les paiements |
| `taux_ferme` | part des paiements en allocation **ferme** : un seul débiteur porte un signal fort (client file univoque, référence désignant un seul débiteur, IBAN direct unique) |
| `précision_ferme` | parmi les allocations fermes : débiteur ferme = vrai débiteur (paiements à un seul débiteur) |
| `sans_candidat` | part des paiements sans aucun débiteur candidat |
| `candidats_moyens` | taille moyenne de la liste |
| `avec_client_file` | part des paiements avec un client file rattaché au dernier passage |

| Tableau | Contenu |
|---|---|
| `_by_route.csv` | les mêmes taux par route IBAN : `DEBTOR_DIRECT`, `ASSIGNOR`, `TECHNICAL_ACCOUNT`, `UNKNOWN` |
| `_by_status.csv` | les mêmes taux par statut : `ferme`, `multiple`, `aucun` |
| `_found_by.csv` | pour les paiements à un seul vrai débiteur : **quels signaux** l'ont fait entrer dans la liste (`reference`, `iban+amount`…), ou `non trouvé` / `plusieurs débiteurs` |
| `_misses.csv` | 200 exemples de paiements dont le vrai débiteur manque, avec libellé, route et candidats |

Un débiteur manqué ici est perdu pour les étapes 4 et 5 : ni les règles ni le ML ne regardent les
factures d'un débiteur absent de la liste.

---

## 6. Étapes 4, 5 et 6 — évaluation d'un rejeu (`evaluate` / `backtest`)

Code : `evaluate`, appelé par `Project.evaluate` dans
src/api.py. Toutes les mesures portent sur le périmètre (§1.5) et la décision retenue
par paiement (§1.6).

### 6.1 Le résumé (`evaluation_<m>_<p>.json`, en tête du `.md`)

| Champ | Calcul |
|---|---|
| `paiements_périmètre` | paiements arrivés dans la période |
| `avec_imputation_réelle` / `sans_imputation_réelle` | partage du périmètre selon l'existence d'une imputation |
| `auto` | paiements dont la décision retenue est une auto-validation |
| `auto_corrects` | auto-validations dont l'ensemble de factures est exact |
| `revue` | paiements dont la décision retenue est une mise en revue |
| **`taux_automatisation`** | `auto / paiements_périmètre` |
| **`précision`** | `auto_corrects / auto` |
| `taux_automatisation_correct` | `auto_corrects / paiements_périmètre` |
| `précision_cible`, `cible_atteinte` | cible de `evaluation.target_precision` (99,5 %) ; atteinte si `précision ≥ cible` |
| `taux_automatisation_à_précision_cible` | point le plus à droite de la courbe (§6.9) encore au-dessus de la cible |
| `précision_paires`, `rappel_paires` | au niveau des couples (paiement, facture) : parmi les couples auto-validés, part des vrais ; parmi tous les vrais couples du périmètre, part auto-validée |
| `taux_actuel_référence` | taux de l'algorithme actuel, **à renseigner** dans `evaluation.current_automation_rate` |
| `ml_*` | diagnostics du ML (§6.10), présents pour le rapprocheur `pipeline` |

### 6.2 Attention aux dénominateurs

C'est la première source de confusion : tous les tableaux n'ont pas le même.

| Tableau | `taux_automatisation` = auto de la ligne / … | Les lignes s'additionnent ? |
|---|---|---|
| `by_step`, `by_rule` | **tous** les paiements du périmètre | **oui**, la somme est le taux global |
| `by_group`, `by_month`, `by_client_file` | les paiements **de la ligne** (du groupe, du mois…) | non, ce sont des taux internes à chaque ligne |
| `by_rule_alone` (`couverture`) | tous les paiements du périmètre | **non**, les règles se recoupent |
| `cascade` | tous les paiements du périmètre | non, chaque niveau inclut le précédent |

Dans tous les tableaux, **`précision` = corrects / auto de la ligne**. La précision d'un regroupement
de lignes n'est pas la moyenne des précisions : c'est `Σ corrects / Σ auto`.

### 6.3 `cascade`

Trois niveaux, chacun mesuré sur le même rejeu :

| Niveau | Calcul |
|---|---|
| Algorithme actuel | `evaluation.current_automation_rate` (vide tant qu'il n'est pas renseigné) |
| Étape 4 — règles seules | paiements auto-validés **par une règle** / périmètre, et leur précision |
| Étapes 4 + 5 — règles puis ML | toutes les auto-validations (règles + ML) / périmètre, et leur précision |

`gain_points` = écart en points avec l'algorithme actuel, une fois celui-ci renseigné.

Le niveau « règles seules » est mesuré dans le même rejeu que le ML. Une facture prise par le ML un
jour donné n'est plus disponible pour les règles les jours suivants. D'où de petits écarts avec un
rejeu du rapprocheur `rules` seul (`backtest(p, "rules")`).

### 6.4 `by_step` et `by_rule` : qui a auto-validé

- **`by_step`** : auto-validations par étape, `rules` ou `ml`.
- **`by_rule`** : auto-validations par règle ou type de proposition ML.

Chaque paiement n'y apparaît qu'une fois. Côté règles, c'est la règle de plus haute priorité (R1 → R5)
qui a trouvé une solution unique ce jour-là (`RulesMatcher.process`).

Quand plusieurs paiements du jour réclament la même facture, ils sont servis par priorité de règle
(puis par identifiant de paiement). Chacun n'est accepté que s'il reste assez de restant dû sur ses
factures. Si plusieurs paiements soldent chacun la facture par la même règle (ex æquo), tous sont
rejetés et restent dans le reliquat.

| `rule` | Origine |
|---|---|
| `R1_CLIENT_FILE` … `R5_PARTIAL_REFERENCED` | règles de l'étape 4 (`config/rules.yaml`) |
| `ML_SINGLE` | ML : une facture seule qui solde le paiement |
| `ML_SET` | ML : un ensemble de factures dont la somme égale le paiement à la tolérance près |
| `ML_PARTIAL` | ML : paiement partiel d'une facture. Auto-validé seulement si son propre seuil le permet (§7.4) ; souvent jamais, les partiels étant peu fiables |

### 6.5 `by_rule_alone` : chaque règle seule

Code : `rule_alone_metrics`. Chaque jour, **toutes** les règles
sont calculées, même si une règle prioritaire l'emporte. Pour chaque règle et chaque paiement, on garde
la **première** solution unique trouvée.

| Colonne | Calcul |
|---|---|
| `propositions` | paiements du périmètre pour lesquels la règle a trouvé une solution unique, qu'elle ait été retenue ou non |
| `correctes` | propositions dont l'ensemble de factures est exact |
| `précision` | `correctes / propositions` |
| `couverture` | `propositions / paiements_périmètre` |

Ce tableau sert à **juger chaque règle** : une règle sous `min_precision` (`config/rules.yaml`, 99,5 %)
est signalée par l'interface. Elle n'est **pas** désactivée automatiquement ; on la désactive à la
main (`enabled: false`). Les couvertures se recoupent (un paiement peut avoir une référence unique
**et** un montant unique) et ne s'additionnent donc pas.

### 6.6 `by_group`

Les taux par type de groupe réel (§1.3). Chaque ligne a pour dénominateur les paiements de ce type :
« 1↔n : 32 % » veut dire que 32 % des paiements qui règlent plusieurs factures sont auto-validés. La
ligne `sans imputation` doit rester proche de 0 : toute auto-validation y est une erreur.

### 6.7 `by_month`

Les mêmes taux par mois d'**arrivée** du paiement, chaque mois ayant ses propres paiements pour
dénominateur. Une baisse d'un mois à l'autre signale une dérive des données. Le dernier mois est
souvent partiel et peu représentatif.

### 6.8 `by_client_file`

Deux lignes : paiements **avec** et **sans** client file rattaché **le jour de leur arrivée**.
Dénominateur : les paiements de la ligne. Rapprocheur `pipeline` uniquement.

### 6.9 `curve` : automatisation / précision

Code : `precision_curve`.

1. **Point de départ** (`seuil = inf`) : les paiements auto-validés par les règles, comptés quel que soit
   le seuil (« règles acquises »).
2. On ajoute ensuite les autres paiements qui ont une proposition du ML (auto-validée ou en revue),
   **par score décroissant**. Chaque point correspond à un seuil τ : les règles + toutes les propositions
   de score ≥ τ.
   - `taux_automatisation` = (règles + propositions ≥ τ) / périmètre.
   - `précision` = part correcte parmi elles.

La courbe est tracée **a posteriori sur la période mesurée**, avec un seuil unique : c'est un plafond
optimiste. Le moteur réel utilise des seuils fixés à l'avance sur la validation, un par type de
proposition, et exige une marge (§7.4). Son point réel (`taux_automatisation`) est donc à gauche de
`taux_automatisation_à_précision_cible`. Un grand écart entre les deux signale des seuils qui se
transposent mal de la validation au test.

### 6.10 Diagnostics du ML (`ml_*` et `ml_calibration`)

Code : `ml_diagnostics`. Population : les paiements du périmètre
**passés en ML** (non résolus par les règles) et ayant une imputation réelle, au **premier jour** où
ils sont passés en ML, avec **tous leurs candidats** (60 au plus).

| Champ | Calcul |
|---|---|
| `ml_paiements_en_ml` | nombre de ces paiements |
| `ml_rappel_candidats` | part d'entre eux dont **toutes** les vraies factures figurent parmi les candidats. Plafond de ce que le ML peut trouver |
| `ml_precision_at_1` | part dont le candidat le mieux classé est une vraie facture |
| `ml_mrr` | moyenne de 1 / rang de la première vraie facture (0 si aucune) : 1 = toujours premier, 0,5 = en moyenne deuxième |

**`ml_calibration`** porte sur des **paires** (paiement, facture candidate), rangées en 10 tranches de
score calibré :

| Colonne | Calcul |
|---|---|
| `paires` | nombre de paires de la tranche |
| `score_moyen` | probabilité prédite en moyenne |
| `taux_observé` | part des paires qui sont de vraies imputations |

Bien calibré : `score_moyen ≈ taux_observé`. Si `taux_observé < score_moyen` dans les tranches
hautes, le modèle est trop confiant. La tranche 0,0–0,1 contient l'immense majorité des paires,
puisque presque tous les candidats sont faux. Ce tableau ne décide rien : les seuils portent sur le
score brut et sur des paiements (§7.4).

---

## 7. Étape 5 — apprentissage (`train`)

Code : `fit_ml` dans src/reconcile_ml/pipeline.py,
src/reconcile_ml/model.py,
src/reconcile_ml/decision.py.

### 7.1 Construction du jeu d'entraînement

1. **Rejeu** de train + validation avec les règles de l'étape 4 (phase la plus longue).
2. Chaque jour, pour les paiements **arrivés ce jour-là** et **non résolus par les règles**, on génère les
   candidats et leurs features, calculées sur l'état de la veille. Le jeu ne garde qu'une part des
   paiements (`reconcile_ml.training.payment_sample`, 50 % par défaut), tirée de façon déterministe,
   mais **tous leurs candidats** : pas d'échantillonnage des négatifs.
3. **Étiquetage**, dans une passe séparée et après le rejeu : une paire vaut 1 si la facture figure
   dans les imputations réelles du paiement, sinon 0.
4. Le jeu est enregistré dans `data/interim/ml/dataset`, découpé par mois, avec
   `dataset_meta.json`. `train(reuse_dataset=True)` le reprend sans rejouer.

Les paires des paiements arrivés en train servent à l'apprentissage, celles de la validation à
l'arrêt anticipé, à la calibration et aux seuils.

### 7.2 Le modèle

- **Passe 1** : LightGBM binaire sur les features des paires ; arrêt anticipé sur la validation.
- **Scores hors échantillon** : la passe 1 est réentraînée sur 3 plis (partage par paiement), pour
  scorer chaque paire d'entraînement par un modèle qui ne l'a pas vue.
- **Passe 2** : LightGBM avec, en plus, les **features de compétition** calculées à partir des scores de
  passe 1 : rang du candidat, marge au suivant, nombre de candidats du paiement.
- **Calibration** : régression isotonique du score de passe 2 sur la validation, qui donne la
  probabilité affichée dans `ml_calibration`.

### 7.3 Les propositions (chaque jour, pour chaque paiement)

`propose` construit au plus quelques propositions par paiement :
- **facture seule** : les deux meilleurs candidats capables d'absorber le paiement. C'est `single` si la
  facture est soldée à la tolérance près (5 € ou 3 %), sinon `partial` ;
- **ensemble** : sous-ensembles des 25 meilleurs candidats dont la somme approche le paiement à la
  tolérance près (5 factures au plus, cardinalité minimale préférée). Score = moyenne des scores des
  factures.

La meilleure proposition l'emporte. Sa **marge** est l'écart de score avec la meilleure proposition
différente. Cas particulier : une facture seule qui n'explique le paiement qu'à un escompte près, face
à un ensemble qui l'explique exactement, reçoit une marge de 0, donc part en revue.

Les paiements non auto-validés d'un même débiteur, proches dans le temps, peuvent recevoir une
proposition n↔n (`ML_NN`), **toujours en revue**.

### 7.4 La décision et le choix des seuils

`decide` :
- **auto-validation** si score ≥ τ_high **du type de proposition** **et** marge ≥ δ
  (`reconcile_ml.decision.min_margin`) ;
- **revue** si score ≥ τ_low (`review_min_score`) ;
- sinon, pas de proposition.

Le score utilisé ici est le **score brut** de la passe 2 : même ordre que la probabilité calibrée, mais
sans ses paliers.

**τ_high est calibré en ligne sur la validation** (`calibrate_online`).
On rejoue la validation avec le ML en mode « propositions seulement », en notant chaque jour la
proposition de chaque paiement. En production, un paiement en attente est re-scoré chaque jour et
serait auto-validé **le premier jour** où son score franchit τ. Pour chaque type de proposition, τ_high
est donc le plus petit seuil qui tient la précision cible (99,5 %) **en simulant ce premier
franchissement**, avec au moins 20 cas. Sans volume suffisant, le seuil vaut 1,01 : jamais
d'auto-validation pour ce type.

`project.calibrate()` refait ce calcul sur le modèle existant, sans réentraîner.

### 7.5 Ce que contient `models/<jeu>/pair_model/`

| Fichier / champ | Contenu |
|---|---|
| `model.json` → `metrics.validation` | sur les **paires** de validation : `auc`, `precision_at_1` et `mrr` (sur les paiements ayant au moins une vraie facture candidate), `groupes_avec_positif` |
| `metrics.rappel_candidats_validation` | part des paiements de validation dont toutes les vraies factures sont candidates |
| `metrics.auto_validation`, `précision_auto_validation` | estimation **hors boucle** sur le jeu de validation, avec les seuils hors boucle (indicatif) |
| `thresholds.kinds` | **τ_high retenus**, par type (`single`, `set`, `partial`), issus de la calibration en ligne |
| `thresholds.offline` | seuils hors boucle, pour comparaison |
| `timings_s` | durées : construction du jeu, apprentissage, calibration en ligne (`jeu repris` si le jeu a été repris) |
| `journal_sha256`, `featurization_version`, `settings_fingerprint`, `periods` | traçabilité : sur quelles données et avec quels réglages le modèle a été construit |
| `feature_importance.csv` | gain de chaque feature dans la **passe 1** (les features de compétition de la passe 2 n'y figurent pas) |
| `calibration_validation.csv` | comme `ml_calibration`, sur les paires de validation. À comparer au tableau du test : un écart signale une dérive |
| `data/interim/ml/validation_daily_proposals.parquet` | les propositions quotidiennes de la calibration en ligne, avec leur exactitude |

---

## 8. Pièges d'interprétation, en résumé

1. **Dénominateurs** : `by_step` / `by_rule` sur tout le périmètre (additionnables) ; `by_group` /
   `by_month` / `by_client_file` sur la ligne (pas additionnables) ; `by_rule_alone` se recoupe (§6.2).
2. **Exactitude stricte** : une facture en trop ou en moins = erreur ; un paiement auto-validé sans
   imputation réelle = erreur.
3. **Paires ou paiements** : `ml_calibration`, `précision_paires`, `metrics.validation` comptent des
   paires ; tout le reste compte des paiements.
4. **La courbe est un plafond** mesuré a posteriori ; le vrai résultat est `taux_automatisation`.
5. **Le reliquat de démarrage** est traité mais exclu de la mesure.
6. **La vérité, c'est l'historique** : si l'algorithme actuel ou un gestionnaire s'est trompé, le moteur
   qui « se trompe pareil » est compté juste, et celui qui a raison contre l'historique est compté faux.
7. **Taux actuel** : la cascade ne montre pas de gain tant que `evaluation.current_automation_rate`
   n'est pas renseigné.
