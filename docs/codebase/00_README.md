# Documentation de la codebase — point d'entrée

> **Résumé.** Ce dépôt est un POC qui rapproche automatiquement des **paiements** reçus et les
> **factures** qu'ils règlent, dans une activité d'affacturage. Il rejoue un an d'historique
> **jour par jour**, sans jamais regarder le futur. Chaque jour, il cherche pour chaque paiement le
> débiteur concerné, applique des règles déterministes, puis un modèle LightGBM sur ce qui reste. Il
> mesure enfin le **taux d'automatisation obtenu à précision fixée** (cible 99,5 %), en comparant ses
> décisions aux imputations réellement prononcées.

---

## 1. Ce que fait le projet

### En cinq lignes

1. Il charge les tables métier (paiements, factures, imputations, débiteurs, cédants, contrats, client
   files) à partir de fichiers décrits par `config/schema.yaml`. Il les normalise, puis les convertit en
   un **journal d'événements** daté.
2. Il **rejoue** ce journal jour par jour. L'état du grand livre au jour D est reconstruit à partir des
   seuls événements antérieurs à D.
3. Chaque jour, pour le lot de paiements à traiter, il **alloue** chaque paiement à des débiteurs
   candidats (client file, référence, IBAN, nom, montant).
4. Il **rapproche** ensuite paiement et factures : d'abord des règles déterministes (R1–R5), puis un
   modèle de ML sur le résiduel. Chaque proposition est auto-validée, mise en revue ou laissée en
   attente.
5. Il **évalue** les décisions contre les imputations réelles, par étape, règle, type de groupe et
   mois. Il produit des rapports CSV et markdown, un modèle versionné et un export des décisions.

### En détail

L'activité : un **cédant** vend ses créances (factures) à la banque ; le **débiteur** paie, parfois
directement, parfois via des comptes techniques. Un algorithme existant rapproche une partie des
paiements, le reste part en traitement manuel. Le POC doit montrer qu'on peut automatiser davantage
**sans dégrader la précision** (≥ 99,5 % de décisions automatiques exactes). C'est le brief :
`CLAUDE.md:9-17`.

Le projet est organisé en six étapes, qui sont aussi les méthodes de `Project` dans
[src/api.py](../../src/api.py) (`src/api.py:60`) :

| Étape | Rôle | Code principal |
|---|---|---|
| 1. Chargement | lire, typer, normaliser, construire le journal, contrôler | `src/load/` |
| 2. Temps | état daté, boucle quotidienne, découpage train / validation / test | `src/timeline/` |
| 3. Allocation | paiement → débiteurs candidats classés | `src/allocation/` |
| 4. Règles | rapprochement déterministe, baseline | `src/reconcile_rules/` |
| 5. ML | candidats, features, LightGBM, ensembles, seuils | `src/reconcile_ml/` |
| 6. Évaluation | métriques, cascade, courbe, export | `src/evaluation/` |

Quatre idées structurent tout le code ; elles sont détaillées dans [02_flux_de_donnees.md](02_flux_de_donnees.md) :

- **Aucune fuite temporelle.** `LedgerState` (`src/timeline/state.py:89`) refuse toute lecture à
  une date autre que celle de son état (`_check`, `src/timeline/state.py:274`). La boucle vérifie
  qu'aucune décision ne cite une facture encore inconnue (`_validate`, `src/timeline/loop.py:147`).
- **Le rejeu se fait « dans l'ombre » de l'historique.** Le journal contient les imputations
  réelles (`IMPUTATION_APPLIED`). Elles font baisser les restants dus et sortent les paiements du
  reliquat comme dans la réalité. Le moteur ne fait que proposer ; ses auto-validations
  **réservent** des montants (`_claimed`) jusqu'à l'arrivée de l'imputation réelle.
- **La vérité est l'historique.** Une décision automatique est correcte si l'ensemble de factures
  décidé est exactement celui réellement imputé (`evaluate`, `src/evaluation/metrics.py:143`).
- **Le volume réel (~2 M de paiements/an) dicte le style du code** : tout est vectorisé (numpy,
  pyarrow), les jointures sont bornées, les lots traités par blocs, la mémoire suivie
  (`src/memory.py`).

---

## 2. Carte de la documentation

| Document | Contenu |
|---|---|
| [00_README.md](00_README.md) | ce document : résumé, carte, parcours, glossaire |
| [01_architecture.md](01_architecture.md) | composants, dossiers, points d'entrée, ordre d'initialisation, stack |
| [02_flux_de_donnees.md](02_flux_de_donnees.md) | **le cœur** : tracé de bout en bout de chaque flux, séquences, exemple d'un paiement |
| [03_gestion_des_donnees.md](03_gestion_des_donnees.md) | sources, schéma canonique (ER), normalisation, journal, split, stockage, risques de fuite |
| [04_modeles_ml.md](04_modeles_ml.md) | le modèle de paires LightGBM (passes 1 et 2), la calibration isotonique, les seuils |
| [05_modules/](05_modules/) | une fiche par package : fonctions, paramètres, appelants, passages non triviaux |
| [06_configuration_et_execution.md](06_configuration_et_execution.md) | toutes les configurations, installation, commandes, livrables |
| [07_tests_et_qualite.md](07_tests_et_qualite.md) | ce qui est testé, ce qui ne l'est pas, comment lancer |
| [08_zones_dombre_et_dette.md](08_zones_dombre_et_dette.md) | code mort, incohérences, risques, questions ouvertes |

Fiches de modules (`05_modules/`) :

| Fiche | Périmètre |
|---|---|
| [config_et_reglages.md](05_modules/config_et_reglages.md) | `src/config.py`, `src/settings.py`, `src/yaml_io.py`, `src/arrow_ops.py`, `config/*.yaml` |
| [load.md](05_modules/load.md) | `src/load/` : modèle canonique, chargement, normalisation, journal, qualité, relecture |
| [timeline.md](05_modules/timeline.md) | `src/timeline/` : `LedgerState`, split, boucle quotidienne |
| [allocation.md](05_modules/allocation.md) | `src/allocation/` : index, signaux, mesure de l'allocation |
| [reconcile_rules.md](05_modules/reconcile_rules.md) | `src/reconcile_rules/` : R1–R5, DFS de sommes, arbitrage |
| [reconcile_ml.md](05_modules/reconcile_ml.md) | `src/reconcile_ml/` : candidats, features, modèle, décision, pipeline |
| [evaluation.md](05_modules/evaluation.md) | `src/evaluation/` : métriques, rapport, export |
| [api.md](05_modules/api.md) | `src/api.py` : la façade `Project` |
| [memory_et_progress.md](05_modules/memory_et_progress.md) | `src/memory.py`, `src/progress.py` : suivi mémoire, CPU, progression |
| [ui.md](05_modules/ui.md) | `src/ui/` : interface Streamlit |
| [synthetic.md](05_modules/synthetic.md) | `src/synthetic/` : générateur de données de test |
| [outils_et_livrables.md](05_modules/outils_et_livrables.md) | `tools/`, `notebooks/`, version portable `single/` |

Documents déjà présents dans le dépôt, complémentaires : `CLAUDE.md` (le brief, qui fait foi),
`spec_rapprochement_automatique.md` (spécification détaillée), `GUIDE.md` (guide pédagogique et
subtilités), `RESULTATS.md` (comment lire chaque rapport et chaque métrique), `README.md`
(installation rapide).

---

## 3. Parcours de lecture recommandé

**Pour comprendre le projet en une heure**
1. Ce document, puis [01_architecture.md](01_architecture.md) (diagramme global).
2. [02_flux_de_donnees.md](02_flux_de_donnees.md), en suivant l'exemple fil rouge du §7.
3. [04_modeles_ml.md](04_modeles_ml.md) §1 à §3.
4. [08_zones_dombre_et_dette.md](08_zones_dombre_et_dette.md).

**Pour modifier le moteur**
1. [03_gestion_des_donnees.md](03_gestion_des_donnees.md) : les garanties anti-fuite à ne pas casser.
2. [05_modules/timeline.md](05_modules/timeline.md), puis la fiche du module concerné.
3. [07_tests_et_qualite.md](07_tests_et_qualite.md) : les tests à relancer.

**Pour exploiter les résultats** : `RESULTATS.md` à la racine du dépôt, puis
[06_configuration_et_execution.md](06_configuration_et_execution.md).

**Pour ajouter une règle de réconciliation** : [docs/ajouter_une_regle.md](../ajouter_une_regle.md).

**Pour brancher les données réelles** : [06_configuration_et_execution.md](06_configuration_et_execution.md) §2
(`config/schema.yaml`), [05_modules/load.md](05_modules/load.md), [03_gestion_des_donnees.md](03_gestion_des_donnees.md) §1.

---

## 4. Glossaire

### Métier

| Terme | Sens dans le code |
|---|---|
| **Affacturage** | la banque (le « factor ») achète les créances d'un cédant et encaisse les paiements des débiteurs |
| **Cédant** (`assignor`) | le client de la banque, qui vend ses factures. Table `assignor` (`src/load/canonical.py:114`) |
| **Débiteur** (`debtor`) | celui qui doit payer la facture. Table `debtor` (`src/load/canonical.py:115`) |
| **Contrat** (`agreement`) | lien cédant ↔ débiteur, avec marché, produit, recours. `src/load/canonical.py:117` |
| **Facture** (`invoice`) | créance cédée : référence client, montant initial, échéance. `src/load/canonical.py:68` |
| **Paiement** (`payment`) | virement reçu : montant signé, date de valeur, libellé, IBAN. `src/load/canonical.py:48` |
| **Imputation** | affectation d'un paiement à une facture (ligne `FULL` ou `PARTIAL`, avec le solde **après** la ligne). Source de la vérité. `src/load/canonical.py:87` |
| **Client file** | fichier envoyé par le client, qui détaille un paiement : factures réglées et montants. `src/load/canonical.py:147` |
| **Compte technique** | IBAN intermédiaire (centralisation), qui ne désigne pas un débiteur. `src/load/canonical.py:134` |
| **`bankroll_code`** | code de portefeuille, qui sert à arbitrer quand un IBAN est à la fois débiteur et cédant |
| **Restant dû** (`balance`) | montant encore ouvert d'une facture à une date, reconstruit par les imputations antérieures |
| **Escompte, frais, retenue de garantie** | écarts typiques entre paiement et facture (≈ 0,5–3 %, 5–40 €, ≈ 5 %), captés par des features |
| **1↔1, 1↔n, n↔1, n↔n** | types de groupes d'imputation réels : un paiement pour une facture, pour plusieurs, etc. |

### Technique

| Terme | Sens dans le code |
|---|---|
| **Modèle canonique** | noms internes des tables et colonnes (`TABLES`, `src/load/canonical.py:175`), reliés aux noms réels par `config/schema.yaml` |
| **Journal** | DataFrame d'événements datés et ordonnés (`build_journal`, `src/load/events.py:89`) ; son empreinte SHA-256 identifie les données dans tous les artefacts (elle couvre les événements, pas les libellés ni les attributs des tables : voir 08, F17) |
| **`as_of`** | instant de l'état : minuit du jour D. L'état reflète les événements strictement antérieurs |
| **`LedgerState`** | état du grand livre à une date, en tableaux numpy (`src/timeline/state.py:89`) |
| **Lot du jour** (`batch`) | paiements arrivés le jour D **plus le reliquat** |
| **Reliquat** | paiements des jours précédents, non résolus, gardés jusqu'à `split.retention_days` (60 j) |
| **Rapprocheur** (`Matcher`) | objet qui reçoit un `DayContext` et renvoie des décisions (`src/timeline/loop.py:57`) : `NullMatcher`, `RulesMatcher`, `PipelineMatcher`, sondes |
| **Rejeu** (`run_replay`) | boucle jour par jour avec un rapprocheur (`src/timeline/loop.py:169`) |
| **Décision** | ligne (paiement, facture, montant, action, étape, règle, version, score) ; action `auto` ou `review` |
| **Allocation ferme** | un seul débiteur porte un signal fort ; sinon « multiple » ou « aucun » |
| **Portée** (`scope`) | couples (ligne du lot, débiteur) autorisés pour une règle : tous les candidats, ou le débiteur ferme |
| **Réservation** (`_claimed`) | montant qu'une auto-validation du moteur bloque sur une facture jusqu'à l'imputation réelle |
| **Candidat (ML)** | facture proposée au modèle pour un paiement, par une ou plusieurs sources (débiteur, référence, montant, client file) |
| **Passe 1 / passe 2** | deux modèles LightGBM ; la passe 2 ajoute les features de compétition (rang, marge) |
| **Proposition** | meilleure solution d'un paiement : `single`, `partial` ou `set`, avec score et marge |
| **τ_high, τ_low, δ** | seuil d'auto-validation (par type de proposition), seuil de revue, marge minimale |
| **Calibration en ligne** | choix de τ_high en simulant le premier jour où le score d'un paiement franchit le seuil |
| **Périmètre** | paiements **arrivés** dans la période évaluée ; dénominateur des taux |
| **Taux d'automatisation** | paiements auto-validés / périmètre ; **précision** = exacts / auto-validés |
| **Version portable** | `single/` : le code concaténé en 4 fichiers, sans mode synthétique |
