# 08 — Zones d'ombre et dette technique

> **Résumé.** Le moteur est cohérent et bien protégé contre les fuites temporelles. Il reste
> toutefois :
> - des **fonctionnalités prévues mais non implémentées** (LLM, seuils par segment effectifs,
>   production au fil de l'eau) ;
> - des **réglages ou colonnes sans effet** ;
> - quelques **bugs probables** (comparaison des propositions à plusieurs factures dans
>   `by_rule_alone`, réservations jamais levées) ;
> - des **points fragiles** (version portable générée hors dépôt, contrôles de version manuels) ;
> - des **questions métier** qui conditionnent la validité des résultats sur données réelles.
>
> Les fuites de données de l'étape ML (apprentissage, sélection des seuils, attributs non datés,
> indicateur principal) sont détaillées dans [docs/fuites_ml.md](../fuites_ml.md).
>
> Chaque point est référencé. « Certain » signifie vérifié dans le code ; « déduit » signifie
> raisonné à partir du code, sans exécution sur données réelles.

---

## 1. Bugs probables

| # | Constat | Références | Statut |
|---|---|---|---|
| B1 | **`by_rule_alone` peut compter fausse une proposition exacte à plusieurs factures (R1, R4).** Les factures d'une proposition sont ordonnées par **position** dans la table (`_record`, tri `["row", "inv"]`), la vérité par **identifiant** (`ground_truth`), et la comparaison est une égalité de tuples. Sur le synthétique, positions et identifiants ont le même ordre, d'où l'invisibilité ; sur données réelles, la précision « seule » de R1/R4 serait sous-estimée. L'évaluation principale (`evaluate`) n'est pas touchée : `_proposals` trie par `invoice_id` | `src/reconcile_rules/matcher.py:316-327`, `:344` ; `src/evaluation/metrics.py:52` ; `src/evaluation/metrics.py:79` | certain (code) ; impact déduit |
| B2 | **Réservation jamais levée.** Une auto-validation du moteur réserve le montant jusqu'à ce que le paiement reçoive **une** imputation réelle. Un paiement auto-validé à tort et jamais imputé bloque sa facture jusqu'à la fin du rejeu. Une imputation réelle partielle (une seule ligne) libère tout | `_release_realised`, `src/reconcile_rules/matcher.py:80-87` | certain ; impact déduit |
| B3 | **Avertissement `invalid value encountered in cast`** dans `offline_proposals` : `astype(int)` sur des colonnes de segment avec `NaN` (catégorie inconnue). Les clés de segment deviennent arbitraires, mais seulement dans les seuils **hors boucle**, qui sont indicatifs | `src/reconcile_ml/pipeline.py:319` | certain ; sans effet sur les décisions |
| B5 | **Une imputation datée avant la création de sa facture est perdue.** Le jour de création, le solde est remis à `initial_amount` et efface le débit antérieur. Le contrôle qualité compte le cas (`avant_creation_facture`) ; l'état ne le corrige pas | `src/timeline/state.py:221-222`, `src/load/quality.py:91` | certain (vérifié par script lors de l'exploration) |
| B6 | **Une imputation dont le paiement est absent de la table est ignorée** par l'état (le solde de la facture ne baisse pas), mais comptée dans la vérité | `src/timeline/state.py:203-215` (`sel` exige `pos ≥ 0`) | certain |
| B7 | **Un identifiant dupliqué de facture ou de paiement fait échouer `LedgerState`.** `_Positions` s'appuie sur `pd.Index.get_indexer`, qui refuse un index non unique. Le chargement signale `cle_dupliquee` mais ne dédoublonne que les parties | `src/timeline/state.py:61-69`, `src/load/loader.py:295` | certain (comportement pandas) ; à confirmer sur données réelles |
| B8 | **Revues fusionnées dans l'évaluation.** `_proposals` réunit les factures de **toutes** les revues d'un paiement le même jour. Une revue `ML_SINGLE`/`ML_SET` et une revue `ML_NN` émises ensemble deviennent une seule proposition, presque toujours jugée fausse. Touche la courbe, l'export et la décision retenue d'un paiement en revue | `src/evaluation/metrics.py:79-96`, `src/reconcile_ml/pipeline.py:233-248` | certain (vérifié par script) |
| B9 | **Affichage des gains en points** : `_fmt` affiche tout décimal ≤ 1 en pourcentage, donc un `gain_points` de −3,2 s'affiche « −320,00 % » | `src/evaluation/report.py:10-15` | certain |
| B4 | **`RulesMatcher` suppose l'index des références.** Si les signaux `reference` **et** `client_file` sont désactivés, `allocator.ref` vaut `None` et les règles R1, R2 et R5 échouent | `src/allocation/allocator.py:80`, `src/reconcile_rules/matcher.py:64` | déduit (rapporté par l'exploration du module) |

---

## 2. Prévu mais non implémenté, ou sans effet

| # | Élément | Constat | Références |
|---|---|---|---|
| N1 | **Extraction LLM des références** (brief §7.6) | réglages `llm_labels` (`enabled`, `endpoint`, `cache_dir`) présents, **aucun code** ne les lit (hors empreinte et interface) | `src/settings.py:201` |
| N2 | **Seuils par segment** | `decision.segmented_thresholds` produit des seuils par segment dans `calibrate_thresholds` (hors boucle). Les seuils **retenus** viennent de `calibrate_online`, qui renvoie `segments: {}`. Le réglage n'a donc aucun effet sur les décisions | `src/reconcile_ml/decision.py:124`, `:154` ; `src/reconcile_ml/pipeline.py:451` |
| N3 | **Désactivation automatique des règles sous `min_precision`** | `config/rules.yaml` annonce « une règle mesurée en dessous est désactivée ». En réalité, seule l'interface compare et propose de désactiver | `config/rules.yaml:7-8`, `src/ui/views/rules.py:106` |
| N4 | **Action `reject`** | déclarée et acceptée par `_validate`, jamais produite | `src/timeline/loop.py:36-37` |
| N5 | **`reference_total_volume`, `reference_manual_volume`** | lus nulle part | `src/settings.py:223` |
| N6 | **Feature `debtor_client_file_rate`** (brief §7.2) | absente des familles de features | `CLAUDE.md:168`, `src/reconcile_ml/features.py:29-44` |
| N7 | **`client_file_total_matches_payment`** | copie exacte de `has_client_file` : elle n'apporte aucune information | `Featurizer.features`, `src/reconcile_ml/features.py:294` (fin de la méthode) |
| N8 | **Colonnes chargées mais inutilisées** | `iban_creditor` (normalisé seulement), `payment_type`, `contract_number`, `source_format`, `issuer_name`, `gap_reason` ; `label_tokens` et `name_tokens` calculés au chargement mais non relus | `src/load/canonical.py:48-172`, `src/load/interim.py` (`UNUSED_COLUMNS`) |
| N9 | **Mode production** | le brief veut « un seul code » pour le rejeu et la production. Seul le rejeu existe : aucun point d'entrée ne traite les paiements d'un jour réel au fil de l'eau | `CLAUDE.md:35` |
| N10 | **Code sans appelant** | `Postings.from_lists`, `ReferenceIndex.key_ids`, `name_terms`, `signal_order` ; `LedgerState._pay_imputed`, `payment_imputed`, `invoice_created_at` ; `ReplayResult.extra` ; `PERIOD_NAMES` ; `progress.active` (plus utilisé depuis `chosen`) | `src/allocation/indexes.py:55`, `:182`, `:186` ; `src/allocation/evaluate.py:139` ; `src/timeline/state.py:342`, `:384` ; `src/timeline/loop.py:137` ; `src/api.py:44` ; `src/progress.py:178` |

---

## 3. Incohérences et points fragiles

| # | Constat | Références |
|---|---|---|
| F1 | **Valeur de fichier ≠ valeur de code** : `candidates.max_per_payment` vaut 50 dans `config/settings.yaml` et 60 par défaut dans le code. `GUIDE.md` et `RESULTATS.md` parlent de 60 | `config/settings.yaml:125`, `src/settings.py:145` |
| F2 | **`features.behavioral_window_days` sert à tout** : c'est la fenêtre de **tout** `LedgerState`, construit dans `Project._state`, pas seulement des features | `src/api.py:180-185` |
| F3 | **`n_to_n_window_hours` est arrondi à des jours entiers** (`// 24`, minimum 1) et compare des dates de valeur : 72 h = 3 jours calendaires | `src/reconcile_ml/pipeline.py:215` |
| F4 | **Aucun contrôle de cohérence modèle ↔ données à l'inférence.** `Project._matcher("pipeline")` charge le modèle sans vérifier l'empreinte du journal ni `FEATURIZATION_VERSION`. Seul `recalibrate_ml` vérifie le journal | `src/api.py:202-215`, `src/reconcile_ml/pipeline.py:470-476` |
| F5 | **Versions incrémentées à la main** (`NORMALIZATION_VERSION`, `FEATURIZATION_VERSION`, version des règles) : rien ne détecte un oubli | `src/load/normalize.py:34`, `src/reconcile_ml/features.py:25` |
| F6 | **Version portable générée hors dépôt.** Le script qui produit `single/` n'est pas versionné. `single/` peut donc dériver de `src/` sans que rien ne le détecte, et aucun test ne le couvre. Une ancienne copie dans `dist/single/` a déjà été utilisée par erreur | `single/`, `.gitignore:5` |
| F7 | **Réécriture des YAML** : `save_settings` / `save_rules` régénèrent tout le fichier à partir des descriptions pydantic ; les commentaires ajoutés à la main sont perdus | `src/yaml_io.py:59`, `src/settings.py:318` |
| F8 | **Calibrateur en pickle** : dépend des versions de scikit-learn, et exécute du code au chargement | `src/reconcile_ml/model.py:145`, `:154` |
| F9 | **Importance des features** limitée à la passe 1 | `src/reconcile_ml/pipeline.py:446-448` |
| F10 | **« Règles seules » de la cascade** mesuré dans le même rejeu que le ML : peut différer d'un `backtest("rules")` | `src/evaluation/metrics.py:143` |
| F11 | **`psutil` non déclaré** dans les dépendances, mais utilisé s'il est présent ; repli sur `/proc` | `src/memory.py:47` |
| F12 | **Départage alphabétique** : à règle égale, les paiements qui se disputent une facture sont servis par ordre de `payment_id`, sans critère métier | `src/reconcile_rules/matcher.py:298` |
| F13 | **Rattachement des client files glouton et définitif** : unicité vérifiée seulement dans le lot du jour ; un fichier rattaché reste consommé, même si le paiement expire | `src/allocation/allocator.py:410-418` |
| F14 | **L'allocation ignore les réservations** : elle lit les restants dus bruts, alors que les règles et le ML lisent le net des réservations | `src/allocation/allocator.py:132`, `:341` ; `src/reconcile_rules/matcher.py:89` |
| F15 | **Signal nom muet sous 100 débiteurs connus** : plafond `min(0,01 × n, 200)` < 1. Sans effet sur données réelles volumineuses, mais il fausse les petits jeux ; les tests relâchent `max_token_share` | `src/allocation/allocator.py:224-225`, `tests/test_allocation.py:24-28` |
| F17 | **L'empreinte du journal ne couvre pas le contenu des tables.** `journal_hash` porte sur les colonnes du journal : dates, types, identifiants, montants imputés. Une modification des libellés, références, noms ou montants de facture ne la change pas. `dataset_key` n'inclut pas `NORMALIZATION_VERSION`. Une donnée source corrigée peut donc être relue avec un jeu ML ou un modèle construit sur l'ancienne | `src/load/events.py:135-149`, `src/reconcile_ml/pipeline.py:323-332` |
| F18 | **Budget de recherche épuisé ignoré** : le drapeau « parcours complet » de `near_subsets` est jeté par `propose` et `_n_to_n`. Si le budget de nœuds est atteint, un ensemble concurrent peut manquer : marge surestimée, ou solution n↔n jugée unique à tort. Le réglage promet pourtant « au-delà → revue » | `src/reconcile_ml/decision.py:57`, `src/reconcile_ml/pipeline.py:224`, `src/settings.py:172` |
| F19 | **Courbe réduite à 200 points** : le taux à précision cible peut perdre le point optimal (écart de l'ordre du dixième de point mesuré lors de l'exploration) | `src/evaluation/metrics.py:132-134` |
| F20 | **La même validation sert à tout** : arrêt anticipé des deux passes, calibration isotonique, et les deux calibrations de seuils. Les mesures de validation sont donc optimistes ; seul le test est propre | `src/reconcile_ml/model.py:115-133`, `src/reconcile_ml/pipeline.py:451` |
| F21 | **Libération mémoire liée au moniteur** : `end_day` n'appelle `malloc_trim` que si un `MemoryMonitor` est actif. Les tâches lancées par l'interface ne libèrent donc jamais la mémoire en fin de journée | `src/memory.py:159-162` |
| F22 | **Rétention réelle de `retention_days + 1` jours** : expiration si `today − first > retention_days` | `src/timeline/loop.py:117` |
| F23 | **`Project.report` peut relire des CSV d'un calcul précédent** : il lit tous les fichiers du motif, et `evaluate` ne supprime pas les anciens. Son ordre d'arguments `(matcher, period)` est l'inverse des autres méthodes | `src/api.py:379-386` |
| F24 | **`progress.task` marque la tâche terminée (« ✓ ») même si elle échoue** : le marquage est dans le `finally` | `src/progress.py:76-87` |
| F25 | **Deux définitions de precision@1** : `ml_diagnostics` compte au dénominateur les paiements sans vraie facture candidate, `pair_metrics` non | `src/evaluation/metrics.py:248`, `src/reconcile_ml/model.py:158-170` |
| F26 | **Documentation interne** : `CLAUDE.md` et `RESULTATS.md` classent « nombre de candidats » parmi les features de compétition, alors que c'est une feature de base (`n_candidates`) | `src/reconcile_ml/features.py:45-46` |
| F27 | **Dépendance cachée à pandas 3.** La normalisation utilise l'expression `\p{M}`, que seul le moteur de chaînes Arrow de pandas 3 accepte (`astype("str")`). Avec pandas 2 et le module `re`, elle lève `bad escape \p`. Or `pyproject.toml` déclare `pandas>=2.2` ; seul `uv.lock` (pandas 3.0.6) protège | `src/load/normalize.py:58-61`, `pyproject.toml:7` |
| F28 | **Bases temporelles mélangées** : les horodatages sont convertis du fuseau source vers UTC, les dates restent dans leur fuseau. Une imputation à 00 h 30 heure de Paris tombe la veille. Une heure ambiguë au passage à l'heure d'hiver devient `NaT`, et l'événement disparaît du journal | `src/load/loader.py:79-95`, `src/load/events.py:81` |
| F29 | **Parquets périmés relus** : `Project.load` n'efface pas les anciens fichiers de `data/interim/`, et `load_interim` relit toute table présente. Une table optionnelle retirée de la configuration serait relue sans détection (déduit) | `src/api.py:162-166`, `src/load/interim.py:48-60` |
| F30 | **Réglages manquants silencieux** : sans fichier, `load_settings` et `load_rules` renvoient les valeurs par défaut sans avertir. La version des règles vaut 1 par défaut, 2 dans le fichier | `src/settings.py:313-325` |
| F31 | **Paramètres de règles non validés** : n'importe quelle clé est acceptée dans `params`, et un `id` de règle inconnu ne lève une erreur qu'à l'exécution du rejeu | `src/settings.py:244-251`, `src/reconcile_rules/matcher.py:255-256` |
| F32 | **Formats de montant refusés** : `1,234.56` avec séparateur `.`, espace fine insécable, notation comptable `(12,00)`. Ces valeurs deviennent NA, comptées en `valeur_invalide` | `_to_cents`, `src/load/loader.py:102` |
| F33 | **`base_dir` résolu différemment** : par rapport au répertoire courant dans `load_all`, à la racine du dépôt dans `Project.load` | `src/load/loader.py:345`, `src/api.py:151` |
| F34 | **Références purement alphabétiques** : `label_numbers` ne garde que des tokens contenant des chiffres, donc une référence de facture sans chiffre ne peut jamais correspondre à un libellé (déduit) | `src/load/normalize.py:125-167` |
| F16 | **R2 et R1 avec tolérance non nulle** : R2 impute le restant dû et non le montant payé ; en R1, une ligne de client file sans montant est toujours concordante | `src/reconcile_rules/matcher.py:172-177`, `:132-170` |

---

## 4. Risques sur les données et la mesure

| # | Risque | Références |
|---|---|---|
| D1 | **Attributs statiques non datés** : `party_iban` (tous les IBAN d'une partie), `name_variants` et le référentiel des comptes techniques n'ont pas de date. Un IBAN ou un nom ajouté tardivement sert dès le début du rejeu : c'est une fuite légère, que le test de troncature ne couvre pas | `src/load/loader.py:295-333`, `tests/test_timeline.py:199-214` |
| D2 | **Imputation sans `updated_at`** : absente du journal (le rejeu l'ignore) mais présente dans la vérité | `src/load/events.py:68-81`, `src/evaluation/metrics.py:52` |
| D3 | **Catégories du modèle construites sur toutes les données**, période de test comprise. Fuite d'une liste de valeurs, sans étiquette | `src/reconcile_ml/features.py:99-106` |
| D4 | **La vérité est l'historique** : erreurs passées apprises et comptées justes ; paiements jamais imputés comptés comme erreurs s'ils sont auto-validés | `src/evaluation/metrics.py:143` |
| D5 | **Écart entre apprentissage et production** : apprentissage sur des paiements au jour d'arrivée, sans ML actif ; inférence aussi sur le reliquat re-scoré chaque jour | `src/reconcile_ml/pipeline.py:104-120`, `:145` |
| D6 | **Collisions d'empreintes 64 bits** dans les vocabulaires (probabilité ~10⁻⁵ pour 30 M de clés) | `src/allocation/indexes.py:87` |
| D7 | **Le synthétique ne reproduit pas les distributions réelles** : gros débiteurs, références génériques, montants ronds, IBAN partagés. Il a masqué les explosions mémoire, qui n'ont été diagnostiquées que sur données réelles | `src/synthetic/generate.py` |

---

## 5. Questions ouvertes à poser à l'équipe

**Données**
1. Format réel des **client files** (structuré ou non, rattachement au paiement, lignes) : le brief
   le donne « à confirmer » ; seule la voie tabulaire est branchée (`src/load/canonical.py:145-146`).
2. Existe-t-il une **date de comptabilisation** distincte de la date de valeur ? Si oui, il faut la
   mapper (`booking_date`) : c'est elle qui ordonne le journal.
3. Part des **paiements jamais imputés** (rejets, remboursements, imputations hors système) : elle
   pèse directement sur le taux et la précision.
4. Les imputations historiques sont-elles **fiables** comme vérité ? Y a-t-il des corrections ou
   annulations d'imputation, et comment apparaissent-elles dans la table ?
5. Les identifiants de **parties** : quelle clé unique, pourquoi plusieurs lignes par débiteur, et
   les IBAN sont-ils datés quelque part ?
6. Existe-t-il un **référentiel des comptes techniques** et des IBAN de centralisation ?

**Métier**
7. **Taux d'automatisation actuel** (`evaluation.current_automation_rate`) et volumes de référence :
   sans eux, la cascade ne montre pas de gain.
8. **Précision cible** : 99,5 % mesurée au paiement et à l'ensemble exact, est-ce la bonne
   définition ? Une facture en trop ou en moins doit-elle compter comme une erreur complète ?
9. Faut-il auto-valider les **paiements partiels** ? Ils sont aujourd'hui presque toujours en revue
   (seuil par type).
10. Coût d'une **revue** comparé à celui d'une erreur : il fixe τ_low et δ.

**Technique**
11. Un **mode production** (traitement quotidien réel) est-il attendu de ce POC, ou le rejeu
    suffit-il ?
12. Faut-il **versionner le générateur de `single/`** dans le dépôt, et ajouter un test qui vérifie
    sa cohérence avec `src/` ?
13. L'**endpoint LLM on-premise** est-il disponible, et la piste vaut-elle d'être menée (brief §7.6) ?
