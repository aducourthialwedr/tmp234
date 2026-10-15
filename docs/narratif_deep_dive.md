# Narratif du deep dive — le code en mode flux

> **À quoi sert ce document.** C'est le fil de ton explication orale : on part de ce qui entre dans le
> système, et à chaque étape on dit ce qu'on reçoit, ce qu'on en fait, ce qui en sort et pourquoi. Chaque
> étape renvoie à la section du notebook `notebooks/deep_dive.ipynb` à montrer et au fichier à ouvrir.
> Les chiffres sont ceux du **jeu synthétique de 2 M de paiements** : à remplacer par les tiens sur données
> réelles (`[à compléter]`).

Durée visée : environ 50 minutes, plus les questions (§12). Devant des data scientists, condenser les §1 à
§6 et dérouler le **§7 bis** (logique ML en détail) ; minutage en fin de document.

---

## 0. L'accroche (1 min)

> « On reçoit des virements. Pour chacun, il faut retrouver les factures qu'il règle. Aujourd'hui une partie est
> automatique, le reste part en manuel. Notre objectif : automatiser **le plus possible**, sans jamais descendre
> sous **99,5 % de décisions automatiques exactes**. Je vais vous montrer le chemin d'un paiement dans le moteur,
> de la donnée brute jusqu'à la décision. »

Le fil rouge, à garder sous le coude :

> Un virement de **1 200,00 €**, libellé `VIR SEPA Dupont Bâtiment fact. FA-2024-0012345`, venant de l'IBAN du
> débiteur Dupont. Il règle la facture `FA-2024-0012345` de 1 200,00 €.

La carte du voyage, à montrer une fois au début :

```
Sources (CSV / Parquet)
  → 1. Chargement & normalisation   → tables canoniques
  → 2. Journal d'événements         → l'historique ordonné dans le temps
  → 3. Rejeu jour par jour          → pour chaque jour D : l'état de la veille + le lot du jour
       → 4. Allocation              → pour chaque paiement : ses débiteurs candidats
       → 5. Règles                  → les cas sûrs, résolus de façon déterministe
       → 6. ML                      → le reste : candidats, score, proposition, décision
  → 7. Évaluation                   → décisions comparées aux imputations réelles
(à côté) 8. Apprentissage           → comment le modèle de l'étape 6 est fabriqué
```

---

## 1. Ce qui entre : les sources (3 min)

**En entrée.** Des fichiers, CSV ou Parquet, extraits du système d'information :
- les **paiements** : montant, date de valeur, libellé, IBAN émetteur ;
- les **factures** : référence, montant, échéance, débiteur, contrat ;
- les **imputations** : quel paiement a été affecté à quelle facture, en réalité ;
- les **débiteurs**, les **cédants**, les **contrats** ;
- en option, les **client files** (le détail d'un paiement envoyé par le client) et les comptes techniques.

**Ce qu'on fait.** Rien n'est deviné. Un fichier de mapping, `config/schema.yaml`, dit pour chaque champ du
modèle interne quelle colonne réelle le contient, l'unité des montants, le fuseau horaire, et comment traduire
les statuts.

**À dire.**
> « Le code ne connaît que des noms internes, le "modèle canonique". Le lien avec vos colonnes est déclaratif.
> Brancher une nouvelle source, c'est remplir ce fichier, pas modifier du code. »

**À montrer.** `config/schema.yaml` ; `src/load/canonical.py` (les 9 tables du modèle).

**Code à ouvrir en live.**

| Où | Quoi | Ce qu'il faut pointer |
|---|---|---|
| `config/schema.yaml:27` | `tables:` | une table = `source` + `columns` (champ interne → colonne réelle) + `value_maps` ; les `null` sont à remplir |
| `src/load/canonical.py:48` | `PAYMENT = Table(...)` | le modèle canonique : nom, type (ID, TEXT, DATE, TIMESTAMP, AMOUNT), requis ou non |
| `src/load/canonical.py:175` | `TABLES` | les 9 tables connues du moteur |

---

## 2. Étape 1 — Chargement et normalisation (5 min)

**En entrée.** Les fichiers sources et le mapping.

**Ce qu'on fait.**
1. **Typage strict.** Identifiants en texte, dates en UTC, et **montants en centimes entiers**, sans jamais
   passer par un nombre à virgule (pas d'erreur d'arrondi). Une valeur illisible devient vide et est signalée,
   jamais devinée.
2. **On retire la fuite la plus évidente.** Le restant dû actuel des factures (`current_amount`) est le
   résultat final : on le met de côté pour l'audit, la pipeline ne le voit jamais.
3. **Normalisation des textes.** Majuscules, accents retirés, ponctuation en espaces. Puis on extrait des
   libellés des **clés de référence candidates** : sans préfixe, sans zéros de tête, inversées, ou faites de
   morceaux recollés.
4. **Consolidation des parties.** Un débiteur sur plusieurs lignes devient une seule entité, qui garde tous
   ses IBAN et ses variantes de nom.
5. **Contrôles qualité** : doublons, orphelins, dates incohérentes, imputations incohérentes. Tout est
   signalé dans un rapport ; rien n'est corrigé en silence.

**Fil rouge.**
```
libellé     : VIR SEPA Dupont Bâtiment fact. FA-2024-0012345
normalisé   : VIR SEPA DUPONT BATIMENT FACT FA 2024 0012345
clés        : 0012345, 12345, 2024, FA2024, 20240012345, FA20240012345, …
référence   : FA-2024-0012345  →  clés 20240012345, FA20240012345
match       : intersection non vide → le libellé cite la facture
```

**En sortie.** Des tables propres, typées, normalisées, écrites en Parquet dans `data/interim/`, plus un
rapport de chargement.

**Pourquoi.**
> « Toute la suite repose sur des données dont on maîtrise le type et l'origine. La normalisation est
> versionnée : si on la change, on relance tout depuis cette étape. »

**À montrer.** Notebook §1 (le testeur de normalisation, en direct) ; `src/load/normalize.py`.

**Code à ouvrir en live.**

| Où | Quoi | Ce qu'il faut pointer |
|---|---|---|
| `src/api.py:134` | `Project.load` | l'enchaînement de l'étape 1 : `load_all` → `build_journal` → contrôles → Parquet → empreinte → rapports |
| `src/load/loader.py:336` | `load_all` | contrôle de la config, lecture table par table, retrait de `current_amount`, normalisation, consolidation |
| `src/load/loader.py:158` | `_check_config` | toutes les erreurs de mapping remontées d'un coup (`SchemaConfigError`) |
| `src/load/loader.py:102` | `_to_cents` | montants en centimes **sans passer par des flottants** |
| `src/load/loader.py:365` | `# Jamais de current_amount brut` | la fuite la plus évidente, retirée dès le chargement |
| `src/load/normalize.py:53` | `normalize_series` | majuscules, accents, ponctuation : vectorisé |
| `src/load/normalize.py:100` | `number_variants` | variantes sans préfixe, sans zéros, inversées |
| `src/load/normalize.py:125` | `label_numbers` | clés extraites d'un libellé (fenêtres de tokens voisins) |
| `src/load/normalize.py:170` | `reference_keys` | même logique côté facture : le match est une intersection |
| `src/load/normalize.py:34` | `NORMALIZATION_VERSION` | la normalisation est versionnée |
| `src/load/loader.py:295` | `consolidate_parties` | débiteurs multi-lignes → une entité, tous les IBAN, toutes les variantes de nom |
| `src/load/quality.py:25` | `check_quality` | les anomalies sont signalées, jamais corrigées en silence |

---

## 3. Étape 2 — Le journal d'événements (3 min)

**En entrée.** Les tables normalisées.

**Ce qu'on fait.** On transforme les tables en **une seule liste d'événements datés** :
- une facture est **créée** ;
- un paiement **arrive** ;
- un client file est **reçu** ;
- une imputation est **appliquée**, avec son montant, déduit de la chaîne des soldes ;
- un contrat ou une partie **ouvre** ou **ferme**.

On les trie par date. À date égale, ce qui ouvre passe avant ce qui l'utilise.

**En sortie.** `journal.parquet`, plus une **empreinte SHA-256** qui identifie cette version des données
partout en aval.

**À dire.**
> « À partir de là, le système ne raisonne plus en tables mais en "film" : on sait exactement ce qui était
> connu à chaque instant. C'est ce qui rend possible un rejeu honnête de l'historique. »

**À montrer.** Notebook §2 (les événements d'un paiement réel) ; `src/load/events.py`.

**Code à ouvrir en live.**

| Où | Quoi | Ce qu'il faut pointer |
|---|---|---|
| `src/load/events.py:29` | `EventType` | l'ordre de déclaration **est** l'ordre de départage à date égale |
| `src/load/events.py:45` | `derive_imputed_amounts` | montant imputé = solde avant − solde après, chaîné par facture |
| `src/load/events.py:89` | `build_journal` | tables → événements datés, tri déterministe |
| `src/load/events.py:135` | `journal_hash` | empreinte SHA-256 binaire, stable entre versions de pandas |
| `src/load/interim.py:48` | `load_interim` | relecture : empreinte vérifiée, colonnes de listes en Arrow (`read_table`, ligne 31) |

---

## 4. Étape 3 — Le rejeu jour par jour, sans voir le futur (6 min)

**En entrée.** Le journal.

**Ce qu'on fait.** On rejoue l'historique **un jour à la fois**. Pour chaque jour D :
1. l'**état du grand livre** est avancé aux événements strictement antérieurs à D : factures ouvertes, restants
   dus reconstruits, contrats actifs, historique récent de chaque débiteur ;
2. on constitue le **lot du jour** : les paiements arrivés ce jour-là, **plus le reliquat**, c'est-à-dire les
   paiements des 60 derniers jours encore non résolus ;
3. on passe ce lot au **rapprocheur** (étapes 4 à 6), qui renvoie des décisions ;
4. on **contrôle** les décisions, puis on avance au jour suivant.

**Les garde-fous, à appuyer.**
- Toute lecture de l'état exige la date ; lire à une autre date est **refusé** (`TemporalError`).
- Une décision qui cite une facture pas encore créée, ou un paiement hors du lot, **arrête le rejeu**
  (`LeakError`).
- Le moteur tourne « dans l'ombre » de l'historique : les imputations réelles s'appliquent comme dans la
  réalité. Quand le moteur auto-valide, il **réserve** le montant de la facture jusqu'à ce que l'imputation
  réelle arrive.

**Chiffres (synthétique).** Un jour type de test : environ 22 800 paiements dans le lot, dont 6 000 nouveaux
et 16 700 en reliquat. Environ 15 à 17 secondes par jour pour le moteur complet.

**À dire.**
> « C'est exactement ce qui se passerait en production : chaque matin, l'état d'hier et les paiements à
> traiter. Un seul code sert au backtest et à la production. »

**À montrer.** Notebook §3 (la lecture refusée, en direct) ; `src/timeline/state.py`, `src/timeline/loop.py`.

**Code à ouvrir en live.**

| Où | Quoi | Ce qu'il faut pointer |
|---|---|---|
| `src/timeline/state.py:89` | `LedgerState` | l'état en tableaux numpy indexés par position |
| `src/timeline/state.py:178` | `advance_to` | événements strictement antérieurs, appliqués jour par jour ; refuse de reculer |
| `src/timeline/state.py:194` | `_apply` | création de facture, imputation → restant dû reconstruit, encours par débiteur |
| `src/timeline/state.py:274` | `_check` | **le garde-fou** : toute lecture à une autre date lève `TemporalError` |
| `src/timeline/loop.py:46` | `DayContext` | ce que reçoit un rapprocheur : le jour, le lot, l'état |
| `src/timeline/loop.py:101` | `DailyIterator.__iter__` | le lot = nouveaux du jour + reliquat ; sorties : imputés, auto-validés, expirés (ligne 117) |
| `src/timeline/loop.py:147` | `_validate` | `LeakError` : paiement hors du lot (ligne 159) ou facture inconnue à D |
| `src/timeline/loop.py:169` | `run_replay` | la boucle : un rapprocheur interchangeable par jour |
| `src/reconcile_rules/matcher.py:80` | `_release_realised` | réservations du moteur, libérées à l'arrivée de l'imputation réelle |

---

## 5. Étape 4 — L'allocation : à quel débiteur est ce paiement ? (5 min)

**En entrée.** Le lot du jour et l'état de la veille.

**Ce qu'on fait.** Pour chaque paiement, on cherche **qui paie**, avant de chercher **quoi**. Cinq signaux, du
plus fort au plus faible :

| Signal | Idée | Poids |
|---|---|---|
| Client file | le client a détaillé ce paiement | 1,0 |
| Référence | une référence de facture ouverte est citée dans le libellé | 0,95 |
| IBAN | l'IBAN émetteur appartient à un débiteur (et non à un compte technique) | 0,9 |
| Nom | le nom du débiteur apparaît dans le libellé (mots rares pondérés) | 0,7 |
| Montant | une facture ouverte a exactement ce montant | 0,5 |

Les signaux se combinent en « ou » probabiliste. L'allocation est **ferme** quand un seul débiteur porte un
signal fort.

**En sortie.** Pour chaque paiement, une **liste classée de débiteurs candidats** avec leurs scores et leurs
signaux, plus le client file rattaché.

**Fil rouge.** Référence citée + IBAN de Dupont + nom retrouvé + montant exact : allocation **ferme** sur
Dupont.

**Pourquoi.**
> « Ça réduit le problème. On ne cherche plus parmi des millions de factures, mais parmi celles de quelques
> débiteurs. Un débiteur raté ici est perdu pour la suite : la cible est un rappel d'au moins 99 %. »

**Chiffres (synthétique).** Rappel 99,16 % ; 78 % d'allocations fermes, justes à 99,64 %.

**À montrer.** Notebook §4 (un paiement ambigu et ses candidats) ; `src/allocation/allocator.py`.

**Code à ouvrir en live.**

| Où | Quoi | Ce qu'il faut pointer |
|---|---|---|
| `src/allocation/allocator.py:41` | `WEIGHTS` | le poids de chaque signal |
| `src/allocation/allocator.py:436` | `Allocator.allocate` | client files sur tout le lot, puis les autres signaux par blocs de `CHUNK_ROWS` (ligne 44) |
| `src/allocation/allocator.py:110` | `_reference` | clés du libellé → factures **ouvertes à D** → débiteurs |
| `src/allocation/allocator.py:118` | `_debtors_of_keys` | jointures bornées : jamais « lot × toutes les factures d'une clé fréquente » |
| `src/allocation/allocator.py:165` | `_iban` | routage DEBTOR_DIRECT / ASSIGNOR / TECHNICAL_ACCOUNT / UNKNOWN, arbitrage bankroll |
| `src/allocation/allocator.py:218` | `_name` | index inversé des noms, mots rares pondérés (idf) |
| `src/allocation/allocator.py:306` | `_amount` | montant exact, recherche dichotomique dans les montants ouverts triés |
| `src/allocation/allocator.py:511` | `c_score = 1 − Π(1 − s)` | la combinaison « ou » probabiliste |
| `src/allocation/allocator.py:523` | `firm_mask` | allocation ferme = un seul débiteur fort |
| `src/allocation/indexes.py:35` | `Postings` | index CSR (clé → valeurs), construits une fois, sans information temporelle |

---

## 6. Étape 5 — Les règles : les cas sûrs (4 min)

**En entrée.** Le lot, l'allocation, les factures ouvertes des débiteurs candidats.

**Ce qu'on fait.** Cinq règles déterministes, par priorité :

| Règle | Cas traité |
|---|---|
| R1 | le client file désigne chaque facture sans ambiguïté |
| R2 | une seule facture ouverte correspond à une référence du libellé, montant exact |
| R3 | une seule facture ouverte du débiteur a exactement le montant |
| R4 | une seule combinaison de factures du débiteur somme exactement au paiement |
| R5 | référence unique, paiement inférieur au restant dû : imputation partielle |

**Le principe** : on ne décide **que si la solution est unique**. Le moindre doute envoie le paiement au ML.

Si deux paiements visent la même facture le même jour, ils sont servis par priorité de règle ; s'ils sont à
égalité, aucun des deux n'est validé.

**En sortie.** Des décisions automatiques tracées : règle, version des règles, montant imputé.

**Fil rouge.** R2 : une seule facture référencée, montant égal → auto-validé, réservé.

**Chiffres (synthétique).** Règles seules : **62,75 %** des paiements automatisés à 99,61 %.

**À dire.**
> « C'est la baseline à battre, totalement auditable. Ajouter une règle, c'est une méthode et trois lignes de
> configuration (`docs/ajouter_une_regle.md`). »

**À montrer.** Notebook §5 ; `config/rules.yaml`, `src/reconcile_rules/matcher.py`.

**Code à ouvrir en live.**

| Où | Quoi | Ce qu'il faut pointer |
|---|---|---|
| `config/rules.yaml:11` | `rules:` | règles déclaratives : priorité, paramètres, version |
| `src/reconcile_rules/matcher.py:228` | `RulesMatcher.process` | allocation → portées → chaque règle → cascade → arbitrage → décisions |
| `src/reconcile_rules/matcher.py:50` | `_unique_per_row` | **le principe** : une seule facture possible, sinon rien |
| `src/reconcile_rules/matcher.py:172` | `_r2` | référence unique + montant exact |
| `src/reconcile_rules/matcher.py:197` | `_r4` | somme exacte par DFS borné (`exact_subset`, `src/reconcile_rules/subset.py:28`) |
| `src/reconcile_rules/matcher.py:261` | `# Cascade` | la règle la plus prioritaire l'emporte |
| `src/reconcile_rules/matcher.py:283` | `_arbitrate` | factures disputées servies par priorité ; ex æquo rejetés |
| `src/reconcile_rules/matcher.py:89` | `_effective` | restant dû net des réservations du moteur |

---

## 7. Étape 6 — Le ML : le résiduel (8 min)

**En entrée.** Les paiements que les règles n'ont pas résolus, reliquat compris : c'est-à-dire les cas
difficiles.

**Ce qu'on fait, en cinq temps.**
1. **Candidats.** Pour chaque paiement, les factures plausibles, par quatre portes d'entrée : factures des
   débiteurs alloués, référence citée, montant exact, client file. Puis un plafond d'environ 50 par paiement.
2. **Features.** 54 caractéristiques par couple (paiement, facture) :
   - écarts de montant (escompte, frais, retenue) ;
   - dates (retard, ancienneté) ;
   - texte (la référence est-elle citée ?) ;
   - IBAN, allocation, contrat ;
   - habitudes du débiteur sur les 6 mois **précédents** ;
   - client file.
3. **Score.** Un modèle LightGBM en deux passes. La première note chaque couple. La seconde ajoute la
   « compétition » au sein du paiement : rang, et écart avec le deuxième.
4. **Proposition.** Une facture seule, un paiement partiel, ou un **ensemble** de factures dont la somme
   approche le paiement (recherche bornée). On garde la meilleure, avec sa marge sur la suivante.
5. **Décision.**
   - **auto** si le score dépasse le seuil de son type **et** que la marge est suffisante ;
   - **revue** si le score est moyen ;
   - **attente** sinon : le paiement sera revu demain.

**Le point clé : comment les seuils sont choisis.** On rejoue la période de validation en notant chaque jour la
proposition de chaque paiement. Un paiement en attente est re-noté chaque jour, et c'est **le premier jour où
il franchit le seuil** qui compte. On choisit le seuil qui tient 99,5 % dans ces conditions réelles. Un seuil
par type de proposition : les partiels n'atteignent pas la précision requise, donc ils ne sont jamais
auto-validés.

**En sortie.** Des décisions `auto` ou `review` tracées (`ML_SINGLE`, `ML_SET`…, avec le score).

**Chiffres (synthétique).**
- Rappel des candidats : environ 80 % (le plafond du ML).
- precision@1 : 95,7 %.
- Apport : **+7,4 points** d'automatisation, à 99,64 % de précision.

**À dire.**
> « Le ML n'invente rien : il classe des candidats qu'on sait expliquer. Et il ne décide seul que quand
> l'historique montre qu'à ce niveau de score, on a raison 99,5 % du temps. »

**À montrer.** Notebook §6 : un paiement, ses candidats triés, la proposition, et les features qui ont pesé.

**Code à ouvrir en live.**

| Où | Quoi | Ce qu'il faut pointer |
|---|---|---|
| `src/reconcile_ml/pipeline.py:145` | `PipelineMatcher.process` | **la journée ML en 15 lignes** : règles → résiduel → score → propose → decide → arbitrage → n↔n |
| `src/reconcile_ml/pipeline.py:59` | `_residual` | paiements non résolus → candidats → features |
| `src/reconcile_ml/pipeline.py:153` | `self.model.raw` | score brut (passe 1 → compétition → passe 2) |
| `src/reconcile_ml/pipeline.py:158` | `propose` | une proposition par paiement |
| `src/reconcile_ml/pipeline.py:165` | `decide` | auto / revue / attente |
Fichiers : `src/reconcile_ml/` ; et `docs/choix_etape_ml.md` pour le pourquoi.

---

## 7 bis. La logique ML en détail — pour un public de data scientists (15 min)

> Section à dérouler quand l'auditoire est data science. Elle reprend l'étape 6 avec le vocabulaire du métier
> de DS : formulation, données, features, modèle, empilement, calibration, décision, validation, limites. Les
> réponses aux questions probables sont au §12.

### 7 bis.1 Formulation : un problème de *retrieval + ranking*, décidé à précision fixée

*Code :* `src/reconcile_ml/pipeline.py:290` `label_pairs` (l'étiquette) · `src/reconcile_ml/model.py:158` `pair_metrics` (métriques par groupe).

| Élément | Choix |
|---|---|
| **Unité statistique** | le couple (paiement, facture candidate) |
| **Étiquette** | 1 si la facture fait partie des imputations réelles du paiement, sinon 0 |
| **Groupe** | le paiement (toutes les métriques de classement sont par paiement) |
| **Tâche** | classification binaire de paires, utilisée comme **classement intra-paiement** |
| **Prévalence** | environ **2 %** de positifs (≈ 1 vraie facture pour 50 candidats) |
| **Volumes (tes données réelles)** | 2,23 M paires d'entraînement, 0,78 M de validation |
| **Objectif métier** | maximiser le volume de paiements auto-validés **sous contrainte** de précision ≥ 99,5 % au niveau du **paiement** (ensemble de factures exact) |

> « Architecture en deux étages, comme un moteur de recommandation : un étage de *retrieval* (les candidats),
> qui fixe le plafond de rappel, puis un étage de *ranking* (le modèle). La décision finale n'est pas un argmax :
> c'est un seuil calibré pour une précision cible. »

**Pourquoi binaire plutôt que multi-classe ou ranking pur.**
- Le nombre de candidats varie d'un paiement à l'autre : une formulation multi-classe ne tient pas.
- Un paiement groupé a **plusieurs** bonnes réponses : une formulation « une seule bonne réponse » non plus.
- On a besoin d'un **score absolu**, comparable à un seuil de précision. Un score de ranking pur
  (`lambdarank`) n'a de sens que relativement aux autres candidats du paiement.

### 7 bis.2 Données d'entraînement : construites par rejeu, étiquetées à part

*Code :* `src/reconcile_ml/pipeline.py:92` `DatasetRecorder` (son `process`, ligne 104 : nouveaux paiements, résiduel, échantillon) · `src/reconcile_ml/pipeline.py:43` `_sampled` · `src/reconcile_ml/pipeline.py:290` `label_pairs` · `src/reconcile_ml/pipeline.py:323` `dataset_key` (versionnement du jeu).

- **Rejeu** de l'entraînement et de la validation avec les **règles actives**. Chaque jour, pour chaque
  paiement **nouveau** non résolu par les règles (la population exacte que le ML verra en production), on
  enregistre **tous** ses candidats et leurs features, calculées sur l'état de la veille
  (`DatasetRecorder`, `src/reconcile_ml/pipeline.py`).
- **Pas d'échantillonnage des négatifs.** Les probabilités restent cohérentes avec la prévalence réelle, ce qui
  est indispensable pour fixer des seuils à 99,5 %. On échantillonne seulement des **paiements entiers**
  (`payment_sample = 0,5`, tirage par hachage de l'identifiant), ce qui garde les groupes intacts.
- **Déséquilibre de classes** : pas de pondération. La perte est la log-loss binaire. Le déséquilibre ne pèse
  pas sur la décision, qui se fait par seuil de précision. Il fausse en revanche l'AUC, très optimiste ici,
  d'où le suivi de la precision@1 et du MRR par paiement.
- **Étiquetage dans une seconde passe**, après le rejeu, à partir de la table d'imputation : aucune
  information de vérité ne peut atteindre les features.
- **Biais de sélection assumé** : le modèle n'apprend que le résiduel des règles. C'est voulu (même
  population qu'en inférence), mais il faut **réentraîner après toute modification des règles**.
- **Versionnement** : le jeu enregistré porte l'empreinte du journal, la version de featurisation et
  l'empreinte des réglages dont il dépend. Une reprise est refusée si l'un d'eux change.

### 7 bis.3 Retrieval : la génération de candidats (le vrai plafond)

*Code :* `src/reconcile_ml/features.py:136` `Featurizer.candidates` (découpage par charge) · `src/reconcile_ml/features.py:178` `_candidates_rows` (les 4 sources et les filtres) · `src/reconcile_ml/features.py:234` `_cap` (le plafond).

**Sources réunies** (`Featurizer.candidates`) :
1. factures ouvertes des **3 premiers débiteurs** de l'allocation, avec une échéance entre −180 et +30 jours ;
2. factures dont une **clé de référence** est citée dans le libellé ;
3. factures ouvertes de **montant exact**, créées à ±90 jours ;
4. factures **citées par le client file**.

**Filtres durs** : devise, facture ouverte nette des réservations du moteur, contrat actif.

**Plafond par paiement** (50, `_cap`) :
- les sources précises (référence, montant, client file) passent d'abord ;
- puis, parmi les factures du débiteur, l'union des plus proches **en montant** et des échéances les plus
  proches **parmi celles qui tiennent dans le paiement**. C'est ce qui garde les vraies factures des
  paiements groupés.

**Métrique** : le *rappel des candidats* (toutes les vraies factures sont-elles candidates ?), environ
**80 %** sur le synthétique. C'est le plafond de tout l'étage ML.

> « 20 % des paiements du résiduel ne peuvent pas être trouvés quel que soit le modèle. C'est le premier
> levier, avant tout réglage du modèle. »

Les bits de source (`src_debtor`, `src_reference`…) sont **aussi des features** : le modèle sait par quelle
porte la facture est arrivée.

### 7 bis.4 Features : 54 colonnes, toutes calculables à la date du jour

*Code :* `src/reconcile_ml/features.py:29` `FAMILIES` (la liste) · `src/reconcile_ml/features.py:294` `Featurizer.features` (la construction) · ligne 322 `is_typical_discount` · ligne 367 `debtor_stats` (agrégats datés, `src/timeline/state.py:348`) · ligne 374 `days_to_due_zscore` · `src/reconcile_ml/features.py:393` `_text`.

Une ligne = une paire (paiement, facture candidate). Toutes les features sont calculées à la date du jour,
sur l'état de la veille. Ce sont les **entrées communes aux deux passes LightGBM** (§7 bis.6).

| Famille | Features | Ce que ça mesure |
|---|---|---|
| **Base** (6) | `src_debtor`, `src_reference`, `src_amount`, `src_client_file` | par quelle « porte » la facture est devenue candidate (débiteur alloué, référence citée, montant exact, client file) |
| | `n_candidates` | nombre de candidates du paiement |
| | `amount_rank_in_payment` | rang de la facture parmi les candidates, par écart de montant |
| **Montant** (10) | `amount_diff` (€), `amount_diff_rel`, `amount_ratio`, `amount_exact`, `payment_covers` | écart et ratio entre le paiement et le restant dû |
| | `is_typical_discount`, `is_bank_fee_gap`, `is_retention_gap` | l'écart ressemble-t-il à un escompte (0,5–3 %), à des frais (5–40 €), à une retenue de garantie (4–6 %) ? |
| | `ratio_to_initial`, `balance_is_partial` | paiement / montant initial ; facture déjà partiellement payée |
| **Temporel** (4) | `days_to_due`, `days_since_creation`, `is_before_creation` | date de valeur comparée à l'échéance et à la création de la facture |
| | `days_to_due_zscore` | ce retard, rapporté au retard habituel **de ce débiteur** |
| **Textuel** (6) | `ref_full_in_label`, `ref_key_in_label`, `internal_ref_in_label` | la référence (complète, une clé, ou la référence interne) est-elle dans le libellé ? |
| | `label_length`, `label_has_no_alpha`, `n_label_numbers` | forme du libellé |
| **Identité** (4) | `iban_route` (cat.), `iban_matches_invoice_debtor` | route IBAN du paiement ; l'IBAN est-il celui du débiteur de la facture ? |
| | `channel`, `bankroll_code` (cat.) | canal du paiement, portefeuille |
| **Comportemental** (8) | `debtor_mean_payment_delay`, `debtor_std_payment_delay`, `debtor_partial_payment_rate`, `debtor_grouping_rate`, `debtor_ref_citation_rate`, `debtor_payment_count` | habitudes du débiteur sur la fenêtre glissante **[D − 180 j, D)**, strictement antérieure |
| | `debtor_open_invoice_count`, `debtor_open_invoice_amount` | encours du débiteur à la date du jour |
| **Contrat** (3) | `market`, `product`, `recourse` (cat.) | contrat de la facture |
| **Allocation** (9) | `alloc_rank`, `alloc_score`, `alloc_is_firm_debtor`, `debtor_in_allocation` | place du débiteur de la facture dans l'allocation du paiement (étape 3) |
| | `alloc_sig_client_file`, `alloc_sig_reference`, `alloc_sig_iban`, `alloc_sig_name`, `alloc_sig_amount` | quels signaux ont désigné ce débiteur |
| **Client file** (4) | `has_client_file`, `invoice_cited_in_client_file`, `client_file_line_amount_diff`, `client_file_total_matches_payment` | le client a-t-il détaillé ce paiement, et cite-t-il cette facture ? |

**Ce qui n'est pas une feature**, volontairement :
- **Les identifiants** (`payment_id`, `invoice_id`, débiteur) : pas de mémorisation, et une généralisation
  possible aux nouveaux débiteurs, grâce aux agrégats comportementaux.
- **Les montants bruts** (`balance`, `payment_amount`) : seulement des écarts et des ratios, pour être
  indépendant de l'échelle.
- **`label`** : c'est la cible.
- **`group`** (le paiement) : il ne sert qu'à calculer la compétition et à former les plis.

Points à souligner :
- **Catégories** : les 6 features marquées « cat. » sont déclarées catégorielles à LightGBM, avec un
  vocabulaire figé et sauvegardé avec le modèle. Une valeur inconnue devient `NaN`.
- **Valeurs manquantes** natives (débiteur sans historique, facture hors allocation) : pas d'imputation
  arbitraire.
- **Garantie anti-fuite des features** : tout est lu via l'état daté. Le test de troncature du journal
  vérifie que les scores d'un jour ne dépendent pas du futur.
- **Redondance connue** : `client_file_total_matches_payment` est une copie exacte de `has_client_file`.
  Il n'y a donc en réalité que 53 informations distinctes.

**Ablation rapide** (notebook d'exploration §8, passe 1 seule, échantillon de 3 %) : sans la famille
**montant**, la precision@1 perd 4,0 points ; sans **temporel**, 3,3 ; sans **allocation**, 1,8 ; sans
**comportemental**, 0,4. Les autres familles comptent peu sur le synthétique.

### 7 bis.5 Modèle : LightGBM, configuration sobre et déterministe

*Code :* `src/reconcile_ml/model.py:57` `_params` · `src/reconcile_ml/model.py:69` `_train` (early stopping, ligne 77) · `config/settings.yaml` section `reconcile_ml.training`.

| Paramètre | Valeur |
|---|---|
| Objectif | `binary` (log-loss) |
| `learning_rate` / `num_leaves` / `min_data_in_leaf` | 0,05 / 63 / 50 |
| `feature_fraction` | 0,9 |
| Arbres | ≤ 400, **early stopping** à 30 itérations sur la log-loss de validation |
| Reproductibilité | `deterministic=True`, `force_row_wise`, `seed=42` |
| Threads | CPU réellement disponibles du pod (quota cgroup), pas les cœurs du nœud |

> « Pas de recherche d'hyperparamètres à ce stade, et je l'assume : l'essentiel du gain se joue sur le
> retrieval et les features. Une recherche sur la validation temporelle est une étape suivante simple. »

### 7 bis.6 Empilement en deux passes : injecter le contexte du paiement

*Code :* `src/reconcile_ml/model.py:37` `competition_features` · `src/reconcile_ml/model.py:111` `PairModel.fit` (plis hors échantillon, ligne 119) · `src/reconcile_ml/model.py:93` `PairModel.raw` (enchaînement à l'inférence).

**En une phrase.** Le modèle note d'abord chaque facture **toute seule** (passe 1). Puis il réévalue chaque note
**en voyant celles des autres candidates du même paiement** (passe 2).

```
Une ligne = une paire (paiement, facture candidate)

  54 features ──► LightGBM passe 1 ──► s1 (score de chaque paire, seule)
                                           │
                    par paiement : rang, meilleure concurrente, écart
                                           ▼
  54 features + 3 features de compétition ──► LightGBM passe 2 ──► p (score brut, utilisé pour décider)
                                                                       │
                                                                 isotonique ──► p_cal (probabilité lisible)
```

#### Ce qu'utilise chaque LightGBM

Vérifié sur le modèle entraîné : `pass1.txt` attend 54 features, `pass2.txt` en attend 57.

| | Passe 1 | Passe 2 |
|---|---|---|
| **Entrée** | les 54 features du §7 bis.4 | les mêmes 54 + `rank_in_payment`, `score_margin`, `score_best_other` |
| **Cible** | `label` : 1 si la facture est réellement imputée au paiement | idem |
| **Données** | paires du **train** ; arrêt anticipé sur la **validation** | train, avec la compétition calculée **hors échantillon** ; arrêt anticipé sur la validation |
| **Sortie** | `s1` : sert seulement à calculer la compétition | `p` (score brut) : sert aux propositions et aux seuils ; `p_cal` après l'isotonique |
| **Taille (synthétique)** | 257 arbres | 140 arbres |
| **Réglages** | `_params` (§7 bis.5) | identiques |

#### Le problème : la passe 1 note chaque facture isolément

Exemple illustratif, avec deux paiements de 1 000 € :

| Paiement | Facture candidate | Ce que voit la passe 1 | Score passe 1 |
|---|---|---|---|
| P1 | **A** : 1 000 €, référence citée dans le libellé | montant exact + référence | 0,95 |
| P1 | **B** : 1 000 €, même débiteur, pas de référence | montant exact | **0,60** |
| P1 | C : 400 € | montant éloigné | 0,05 |
| P2 | **D** : 1 000 €, pas de référence | montant exact | **0,60** |
| P2 | E : 300 € | montant éloigné | 0,02 |

B et D ont **le même score (0,60)** : prises isolément, elles se ressemblent (montant exact, pas de référence).
Pourtant leur situation est opposée :
- **B** est battue par A, qui est clairement la bonne facture : B est très probablement fausse.
- **D** est la seule candidate plausible de P2 : D est très probablement juste.

La passe 1 ne peut pas faire la différence, car elle ne voit pas les concurrentes.

#### La solution : trois features calculées à partir des scores de la passe 1

Pour chaque paiement, on classe ses candidates par score `s1` et on ajoute à chaque paire :

| Feature | Définition |
|---|---|
| `rank_in_payment` | rang de la paire dans son paiement, par score `s1` décroissant (1 = meilleure) |
| `score_best_other` | score `s1` de la meilleure **autre** candidate : la 2e pour la 1re, la 1re pour toutes les autres |
| `score_margin` | `s1` − `score_best_other` : positif pour la 1re (son avance), négatif pour les autres (leur retard) |

Sur l'exemple :

| Paire | `rank_in_payment` | `score_best_other` | `score_margin` |
|---|---|---|---|
| A | 1 | 0,60 (B) | **+0,35** |
| B | 2 | 0,95 (A) | **−0,35** |
| C | 3 | 0,95 (A) | −0,90 |
| D | 1 | 0,02 (E) | **+0,58** |
| E | 2 | 0,60 (D) | −0,58 |

La passe 2 reçoit les 54 features d'origine **plus** ces trois colonnes. Elle apprend des règles du type :
- « 1re avec une grosse avance » → monte (D passe par exemple de 0,60 à 0,9) ;
- « 2e derrière une concurrente très sûre » → descend (B passe par exemple à 0,1).

> Analogie : un jury note d'abord chaque copie seule (passe 1), puis regarde le classement. Un 15/20 n'a pas
> le même sens si la meilleure autre copie a 8 ou 19 (passe 2).

Pourquoi ne pas calculer ce rang directement ? On le fait déjà pour des critères simples
(`amount_rank_in_payment`, `n_candidates`). Mais le score de passe 1 résume **toutes** les features à la fois,
donc un rang sur ce score est bien plus informatif.

#### Le point délicat : les scores « hors échantillon » sur le train

Pour entraîner la passe 2, il faut les features de compétition **sur le jeu d'entraînement**, donc des scores de
passe 1 sur ces paires.

**Le piège.** Avec la passe 1 entraînée sur ces mêmes paires, elle les a déjà vues. Elle les note de façon trop
parfaite : environ 0,99 pour les vraies, 0,01 pour les fausses, avec des marges énormes partout. La passe 2
apprendrait alors que « rang 1 avec une grosse marge = toujours juste ». Mais en production, sur des paiements
nouveaux, les scores de passe 1 sont plus hésitants et les marges plus petites. La passe 2 serait **trop
confiante**, exactement là où on veut 99,5 % de précision. C'est la fuite classique de l'empilement de modèles
(*stacking*).

**La parade : la validation croisée hors échantillon** (`PairModel.fit`, plis ligne 119) :
1. On répartit les **paiements** du train en 3 groupes, par hachage de l'identifiant (l'équivalent d'un
   `GroupKFold`). Toutes les candidates d'un même paiement restent dans le même groupe ; sinon le modèle
   verrait une partie du paiement qu'il doit noter.
2. On entraîne la passe 1 sur les groupes 1 et 2, et on note le groupe 3. Puis on recommence avec les deux
   autres combinaisons. Ces modèles n'ont pas de validation : ils utilisent le nombre d'arbres retenu par la
   passe 1 complète.
3. Chaque paire du train reçoit ainsi un score d'un modèle **qui ne l'a jamais vue**, avec le même niveau
   d'hésitation qu'en production.

| Jeu | Scores `s1` utilisés pour la compétition | Pourquoi |
|---|---|---|
| **Train** (entraîner la passe 2) | hors échantillon : 3 modèles de passe 1, chacun entraîné sur 2/3 des **paiements**, notent le tiers restant | pour que la passe 2 apprenne sur des scores aussi hésitants qu'en production |
| **Validation** (arrêt anticipé de la passe 2, calibration) | passe 1 complète | elle n'a jamais vu ces paiements |
| **Production** | passe 1 complète, sur les candidates du jour | idem |

En production, l'enchaînement est dans `PairModel.raw` : passe 1 sur les 54 features, compétition par
paiement, puis passe 2 sur les 57.

#### Coût, limite et phrase à dire

- **Coût** : 3 entraînements de plus (les plis). La passe 2 se désactive par `reconcile_ml.second_pass: false`.
- **Limite** : l'importance des features enregistrée (`feature_importance.csv`) ne porte que sur la **passe 1**.
  Pour l'effet des features de compétition, il faut regarder la passe 2, par exemple avec les valeurs SHAP du
  notebook d'exploration.
- **À dire** :

> « La passe 1 note chaque facture seule. La passe 2 la renote en sachant contre qui elle concourt : c'est un
> effet listwise, tout en gardant un score binaire calibrable. Et pour que la passe 2 apprenne sur des notes
> aussi hésitantes qu'en production, la passe 1 du train est calculée hors échantillon. »

### 7 bis.7 Calibration : isotonique pour lire, score brut pour décider

*Code :* `src/reconcile_ml/model.py:133` `IsotonicRegression` · `src/reconcile_ml/decision.py:3-5` (docstring : pourquoi le score brut).

- Régression **isotonique** ajustée sur le score brut de passe 2 de la validation. Elle sert à lire les scores
  comme des probabilités et à surveiller la fiabilité (tableau par tranche, train puis test).
- **Les décisions utilisent le score brut** : l'isotonique est une fonction en escalier, et ses paliers
  créent des ex æquo massifs près de 1, ce qui empêche de placer un seuil fin. Le score brut a le même ordre,
  sans paliers.
- Pourquoi pas Platt : pas nécessaire, puisque la décision ne passe pas par la probabilité calibrée ; et
  l'isotonique ne suppose pas de forme paramétrique.

### 7 bis.8 Des scores de paires à une décision par paiement

*Code :* `src/reconcile_ml/decision.py:30` `propose` · ligne 46 `absorbs` (facture seule) · ligne 57 `near_subsets` (ensembles) · ligne 70 `margin` · ligne 71 (règle d'ambiguïté) · `src/reconcile_ml/decision.py:136` `decide`.

**Propositions** (`propose`, `src/reconcile_ml/decision.py`) :
- **facture seule** : les 2 meilleurs candidats qui « absorbent » le paiement. Type `single` si le restant dû
  est soldé à la tolérance près (max(5 €, 3 %)), sinon `partial` ;
- **ensemble** (`set`) : sous-ensembles des 25 meilleurs candidats dont la somme approche le paiement à la
  tolérance près (DFS borné, ≤ 5 factures, cardinalité minimale préférée). Le score est la **moyenne** des
  scores des factures : comparable à un score de facture seule, alors qu'un produit pénaliserait
  mécaniquement les grands ensembles.

**Marge** : score de la meilleure proposition moins celui de la meilleure proposition **différente**.

**Règle d'ambiguïté** : une facture seule qui n'explique le paiement qu'à un escompte près, face à un ensemble
qui l'explique exactement, reçoit une marge de 0, donc part en revue.

**Décision** (`decide`) :
- `auto` si score ≥ τ_high(type) **et** marge ≥ δ = 0,05 ;
- `review` si score ≥ τ_low = 0,3 ;
- sinon le paiement reste en attente et sera re-scoré le lendemain.

### 7 bis.9 Choix des seuils : calibration « en ligne », au premier franchissement

*Code :* `src/reconcile_ml/pipeline.py:451` `online_thresholds` (rejeu de la validation) · `src/reconcile_ml/decision.py:154` `calibrate_online` · ligne 173 `setter` (le premier franchissement) · `src/reconcile_ml/decision.py:27` `MIN_KIND_VOLUME`.

**Le problème statistique.** En production, un paiement en attente est re-scoré **chaque jour**. Il a donc
plusieurs chances de franchir le seuil : c'est un problème de **tests répétés**. Un seuil choisi sur une seule
photo par paiement sous-estime le taux d'erreur.

**L'algorithme** (`calibrate_online`) :
1. On rejoue la validation avec `PipelineMatcher(record_proposals=True)` : pas d'auto-validation ML, chaque
   proposition quotidienne est enregistrée avec son exactitude.
2. Pour chaque type de proposition et chaque seuil τ candidat (jusqu'à 2 000 valeurs distinctes du score), on
   retient pour chaque paiement la proposition du **premier jour** où le maximum courant de son score
   franchit τ. Les jours dont la marge est sous δ ne comptent pas.
3. On garde le **plus petit τ** dont la précision sur ces propositions est ≥ 99,5 %, avec au moins 20 cas
   (`MIN_KIND_VOLUME`). Sinon τ = 1,01 : jamais d'auto-validation pour ce type.

**Résultat (synthétique)** : τ_high(`single`) = 0,9976, τ_high(`set`) = 0,9986, `partial` jamais. Les
seuils hors boucle, conservés pour comparaison, sont très proches ici (0,9978 / 0,9986). L'écart devrait
grandir sur données réelles, où le reliquat dure plus longtemps.

**Limite à reconnaître** : avec un minimum de 20 cas, la précision estimée est bruitée. Par exemple, 200
validations avec une erreur donnent 99,5 % estimés, mais la borne basse d'un intervalle de confiance à 95 %
est autour de 97 %. Une amélioration naturelle : choisir τ sur la **borne basse** (Wilson ou
Clopper-Pearson) plutôt que sur la précision ponctuelle.

### 7 bis.10 Protocole de validation et métriques

*Code :* `src/timeline/split.py:53` `compute_split` · `src/reconcile_ml/model.py:158` `pair_metrics` · `src/evaluation/metrics.py:223` `ml_diagnostics` · `src/evaluation/metrics.py:143` `evaluate`.

| Niveau | Métrique | Où |
|---|---|---|
| Retrieval | rappel des candidats | `model.json`, diagnostics du backtest |
| Paires | AUC, precision@1, MRR (sur les paiements ayant un positif), log-loss (early stopping) | `pair_metrics` |
| Calibration | taux observé par tranche de score | `calibration_validation.csv`, `ml_calibration` |
| Décision | précision au **paiement** (ensemble exact), taux d'automatisation, apport ML dans la cascade, courbe automatisation / précision | `evaluate` |

**Découpage temporel**, jamais aléatoire : environ 8 mois d'entraînement, 2 mois de validation, 2 mois de
test, avec 5 jours de purge entre les blocs.

**Usage de chaque période** :
- la **validation** sert à l'early stopping, à l'isotonique et aux seuils : ses métriques sont donc
  optimistes ;
- le **test** n'est vu que par le backtest final : c'est la seule mesure propre.

**Chiffres (synthétique)** :
- validation : AUC 0,9988, precision@1 95,7 %, MRR 0,971, rappel des candidats 79,8 % ;
- test : **+7,4 points** d'automatisation apportés par le ML, à 99,64 % de précision.

### 7 bis.11 Limites connues, et ce que je ferais ensuite

*Code :* `src/reconcile_ml/pipeline.py:205` `_n_to_n` (heuristique n↔n) · `docs/codebase/08_zones_dombre_et_dette.md`.

| Limite | Effet | Piste |
|---|---|---|
| Rappel des candidats ≈ 80 % | plafond du ML | diagnostiquer les candidats manqués par cause, élargir la source fautive |
| Écart entre apprentissage et production | le modèle n'apprend qu'au jour d'arrivée, alors que l'inférence re-score le reliquat avec un état qui a évolué | ajouter les jours de re-scorage au jeu, ou « jours en attente » en feature |
| Validation réutilisée | métriques de validation optimistes | une validation dédiée aux seuils, ou une validation croisée temporelle |
| Seuils sur peu de cas | précision estimée bruitée | seuil sur une borne de confiance |
| Pas de recherche d'hyperparamètres | gain possible non exploité | recherche sur la validation temporelle |
| Seuils par segment inopérants | un seuil global par type | porter la segmentation dans la calibration en ligne |
| n↔n heuristique, toujours en revue | volume non automatisé | modèle au niveau (débiteur, fenêtre de temps) |
| Dérive | seuils figés | suivi mensuel (`by_month`), recalibration (`project.calibrate()`), réentraînement périodique |

---

## 8. Ce qui sort : décisions, évaluation, rapports (4 min)

**En entrée.** Toutes les décisions du rejeu, et la vérité : les imputations réellement prononcées.

**Ce qu'on fait.**
- On ne mesure que les paiements **arrivés** pendant la période.
- Une décision automatique est **juste** si l'ensemble de factures est **exactement** le bon : une facture en
  trop ou en moins est une erreur.
- Indicateurs : taux d'automatisation, précision, cascade (actuel → règles → règles + ML), découpages par règle,
  par type de groupe et par mois, courbe automatisation / précision.

**En sortie.** `reports/` (CSV, JSON, markdown), et un export couple par couple (`export_decisions`) à croiser
avec les métadonnées métier.

**Chiffres (synthétique, période de test de 2 mois).**

| Niveau | Automatisation | Précision |
|---|---|---|
| Algorithme actuel | [à compléter] | [à compléter] |
| Règles seules | 62,75 % | 99,61 % |
| Règles + ML | **70,13 %** | **99,61 %** |

**À dire.**
> « La période de test n'a jamais servi ni à l'apprentissage ni au choix des seuils : c'est une mesure honnête,
> comme si on avait déployé le moteur il y a deux mois. »

**À montrer.** Notebook §7 (rejeu de 7 jours et évaluation) ; `RESULTATS.md` pour lire chaque tableau.

**Code à ouvrir en live.**

| Où | Quoi | Ce qu'il faut pointer |
|---|---|---|
| `src/api.py:374` | `Project.backtest` | `replay` puis `evaluate` |
| `src/evaluation/metrics.py:52` | `ground_truth` | la vérité : ensembles de factures et type de groupe (composantes connexes) |
| `src/evaluation/metrics.py:79` | `_proposals` | une décision retenue par paiement : la première auto, sinon la première revue |
| `src/evaluation/metrics.py:143` | `evaluate` | taux, précision, cascade, découpages |
| `src/evaluation/metrics.py:154` | `same = [...]` | **ensemble exact** : une facture en trop ou en moins = erreur |
| `src/evaluation/metrics.py:113` | `precision_curve` | courbe automatisation / précision, règles acquises |
| `src/evaluation/export.py:30` | `decision_pairs` | export couple par couple, à croiser avec les métadonnées |

---

## 9. À côté du flux : comment le modèle est fabriqué (3 min)

**En entrée.** Les périodes d'entraînement (environ 8 mois) et de validation (2 mois).

**Ce qu'on fait.**
1. On rejoue ces mois avec les règles actives. Chaque jour, pour les nouveaux paiements non résolus, on
   enregistre leurs candidats et leurs features, calculés sur l'état de la veille.
2. **Ensuite seulement**, dans une passe séparée, on étiquette chaque couple : est-il dans l'imputation réelle ?
3. On entraîne le modèle sur l'entraînement, et on l'arrête et le calibre sur la validation.
4. On rejoue la validation pour fixer les seuils.

**En sortie.** Le modèle, ses seuils, ses métriques, et l'empreinte des données et des réglages qui l'ont
produit.

**À dire.**
> « Les features ne voient jamais l'étiquette, et les seuils ne voient jamais le test. »

**Code à ouvrir en live.**

| Où | Quoi | Ce qu'il faut pointer |
|---|---|---|
| `src/api.py:284` | `Project.train` | `reuse_dataset=True` pour reprendre un jeu déjà construit |
| `src/reconcile_ml/pipeline.py:359` | `fit_ml` | rejeu → étiquetage → `PairModel.fit` (ligne 407) → seuils → sauvegarde |
| `src/reconcile_ml/pipeline.py:451` | `online_thresholds` | les seuils viennent de la boucle de validation |

---

## 10. Les garanties transverses (3 min)

| Garantie | Comment | Preuve |
|---|---|---|
| **Pas de fuite temporelle** | état daté, lectures contrôlées, décisions vérifiées | notebook §8 : on coupe le futur du journal, l'allocation du jour D est **identique** ; plus trois tests automatiques |
| **Reproductible** | tri déterministe, seeds fixes, empreintes | test : deux rejeux donnent le même résultat |
| **Tient le volume réel** | tout vectorisé, jointures bornées, traitement par blocs, mémoire et CPU du pod suivis | pic mémoire de l'allocation passé de 10,4 à 4,3 Go sur 2 M de paiements, avec des tests « blocs = passage unique » |
| **Auditable** | chaque décision porte sa règle ou son type, sa version et son score | export `decisions_<rapprocheur>_<période>` |

166 tests automatisés.

**À montrer.** Notebook §8 et §9.

**Code à ouvrir en live.**

| Où | Quoi | Ce qu'il faut pointer |
|---|---|---|
| `tests/test_timeline.py:231` | `test_removing_events_after_d_changes_nothing_computed_up_to_d` | le test anti-fuite : journal tronqué, résultats identiques |
| `tests/test_ml.py:134` | `test_pipeline_runs_without_leak` | le même, sur le moteur complet avec ML |
| `tests/test_allocation.py:253` | `test_chunked_allocation_matches_single_pass` | les optimisations mémoire ne changent pas le résultat |
| `src/memory.py:98` | `available_cpus` | threads limités au quota CPU du pod |
| `src/memory.py:171` | `MemoryMonitor` | mémoire du processus et du pod, phase en cours |

---

## 11. Limites et suites (3 min)

À dire franchement :
- **Le synthétique n'est pas la réalité.** Les chiffres sur données réelles : [à compléter].
- **Le plafond du ML est le rappel des candidats** (environ 80 %) : c'est le premier levier d'amélioration.
- **Les paiements groupés n↔n** ne sont proposés qu'en revue.
- **Quelques points techniques connus** sont listés et priorisés dans `docs/codebase/08_zones_dombre_et_dette.md`.

Ce dont on a besoin :
- le taux d'automatisation actuel ;
- le format réel des client files ;
- la date de comptabilisation ;
- la part des paiements jamais imputés.

---

## 12. Questions probables

**« Pourquoi pas tout en ML ? »**
> Les règles résolvent 63 % des paiements avec une précision quasi parfaite et sont auditables. Le ML se
> concentre sur ce qui est difficile ; il est mesuré par ce qu'il **ajoute**.

**« Comment être sûr qu'il n'y a pas de fuite ? »**
> Par construction : l'état refuse toute lecture à une autre date, et les décisions sont contrôlées. Et par
> test : couper le futur du journal ne change rien au passé (démonstration au §8 du notebook).

**« Que se passe-t-il si le modèle se trompe ? »**
> Les seuils sont choisis pour 99,5 % de précision sur la validation, mesurée au paiement et à l'ensemble
> exact. En dessous du seuil, c'est la revue humaine. La performance est suivie mois par mois pour détecter une
> dérive.

**« Pourquoi LightGBM ? »**
> Données tabulaires, valeurs manquantes, catégories, rapidité, explicabilité (valeurs SHAP). Les alternatives
> sont discutées dans `docs/choix_etape_ml.md`.

**« Comment ça passe en production ? »**
> Le code est déjà celui d'une journée de production : même boucle, même état daté. Il manque le branchement
> sur les flux quotidiens réels : pas de point d'entrée « au fil de l'eau » aujourd'hui.

**« Et la mémoire / le temps ? »**
> Environ 15 secondes par jour sur 2 M de paiements par an. Un apprentissage complet prend environ une heure.
> La mémoire est bornée et suivie ; les tailles de blocs se règlent sans changer le résultat.

### Questions de data scientists

**« Avec 2 % de positifs, comment gérez-vous le déséquilibre ? »**
> Pas de pondération ni de sous-échantillonnage des négatifs : on veut des scores cohérents avec la prévalence
> réelle, puisqu'on décide par seuil de précision. Le déséquilibre rend l'AUC trop flatteuse (0,999). Je
> regarde surtout la precision@1 et le MRR par paiement, et la précision au niveau du paiement après décision.

**« Pourquoi une classification binaire et pas du learning-to-rank ? »**
> Il me faut un score absolu à seuiller, et un paiement peut avoir plusieurs bonnes factures. Le contexte du
> classement est réintroduit par la passe 2 (rang, marge). `lambdarank` en passe 2 est une piste à tester, à
> comparer sur le seul critère qui compte : le volume automatisé à 99,5 % sur le test.

**« Les features de compétition ne fuient-elles pas ? »**
> Elles le feraient avec des scores in-sample. Sur le train, elles viennent d'un out-of-fold à 3 plis groupés
> par paiement ; sur la validation et en inférence, de la passe 1 entraînée sans ces paires.

**« Et les agrégats par débiteur, c'est du target leakage ? »**
> Ils sont calculés sur les imputations **déjà prononcées** avant le jour D, dans une fenêtre glissante
> [D − 180 j, D) maintenue par l'état daté. Le test de troncature du journal le vérifie : retirer le futur ne
> change aucun calcul du passé.

**« Pourquoi l'isotonique et pas Platt ? Et pourquoi décider sur le score brut ? »**
> L'isotonique ne suppose pas de forme. Mais ses paliers créent des ex æquo près de 1, qui empêchent un seuil
> fin. Comme la calibration est monotone, décider sur le score brut donne le même ordre sans les paliers. Le
> score calibré sert au suivi de fiabilité.

**« Vos seuils à 99,5 % sont-ils statistiquement solides ? »**
> C'est la vraie limite. Le minimum est de 20 cas par type, donc la précision estimée est bruitée : avec 200
> cas et une erreur, la borne basse à 95 % est autour de 97 %. La correction est simple : choisir τ sur la
> borne basse de Clopper-Pearson ou de Wilson. Et la mesure de référence reste le test, jamais vu pendant la
> calibration.

**« Pourquoi rejouer la validation pour fixer les seuils ? »**
> Parce qu'un paiement en attente est re-scoré tous les jours : il a plusieurs chances de franchir le seuil.
> On simule le **premier franchissement**, exactement comme en production. Un seuil fixé sur une seule photo
> par paiement serait optimiste.

**« Il y a un écart entre l'apprentissage et la production ? »**
> Oui, et je l'identifie : le modèle apprend au jour d'arrivée, alors que l'inférence re-score aussi le
> reliquat, avec un état qui a évolué. La calibration en ligne corrige les seuils, pas le modèle. Piste :
> intégrer les jours de re-scorage au jeu d'entraînement.

**« Comment comparez-vous deux modèles ? »**
> Sur le backtest de la période de test, au **taux d'automatisation à précision fixée** et à l'apport dans la
> cascade. Le notebook d'exploration sert à trier les idées sur la validation (ablations, nouvelles features),
> mais seul le test tranche.

**« Pas de tuning des hyperparamètres ? »**
> Pas encore, volontairement : les ablations montrent que le gain se joue sur le retrieval et les features
> (montant, temporel, allocation). Une recherche sur la validation temporelle est l'étape suivante.

**« Nouveaux débiteurs, cold start ? »**
> Pas d'identifiant en feature, donc rien à mémoriser. Un nouveau débiteur a des agrégats vides (`NaN`, gérés
> nativement) ; le modèle s'appuie alors sur le montant, la référence, l'allocation et les dates.

**« Comment expliquez-vous une décision ? »**
> Chaque décision porte son type, son score et sa marge. Pour le détail, les contributions SHAP natives de
> LightGBM, paire par paire (démonstration au §6 du notebook). Côté règles, la règle et sa version suffisent.

**« Et la dérive dans le temps ? »**
> Le backtest donne la performance par mois. Les seuils se recalibrent sans réentraîner
> (`project.calibrate()`) ; le modèle se réentraîne en reprenant le jeu déjà construit quand rien n'a changé.

---

## Parcours de code en live : les onglets à ouvrir, dans l'ordre

Ouvre ces fichiers à l'avance dans l'éditeur, dans cet ordre. Le chemin d'un paiement se lit de haut en bas.

| # | Fichier : ligne | Objet | Phrase d'accroche |
|---|---|---|---|
| 1 | `config/schema.yaml:27` | mapping | « Le lien avec vos données est déclaratif. » |
| 2 | `src/api.py:60` | `Project` | « Une méthode par étape : c'est la façade du notebook et de l'interface. » |
| 3 | `src/load/loader.py:336` | `load_all` | « Typage strict, centimes, et la fuite retirée dès l'entrée. » |
| 4 | `src/load/normalize.py:125` | `label_numbers` | « Comment un libellé bruité retrouve une référence. » |
| 5 | `src/load/events.py:89` | `build_journal` | « On passe des tables à un film daté. » |
| 6 | `src/timeline/state.py:274` | `LedgerState._check` | « Le garde-fou : impossible de lire le futur. » |
| 7 | `src/timeline/loop.py:101` | `DailyIterator.__iter__` | « Chaque matin : l'état d'hier et le lot du jour. » |
| 8 | `src/allocation/allocator.py:436` | `Allocator.allocate` | « D'abord qui paie, ensuite quoi. » |
| 9 | `src/reconcile_rules/matcher.py:228` | `RulesMatcher.process` | « Les cas sûrs, uniquement si la solution est unique. » |
| 10 | `src/reconcile_ml/features.py:178` | `_candidates_rows` | « Le retrieval : c'est lui qui fixe le plafond. » |
| 11 | `src/reconcile_ml/features.py:294` | `Featurizer.features` | « 54 features, toutes lues à la date du jour. » |
| 12 | `src/reconcile_ml/model.py:111` | `PairModel.fit` | « Deux passes, plis hors échantillon, isotonique. » |
| 13 | `src/reconcile_ml/decision.py:30` | `propose` | « Des scores de paires à une proposition par paiement. » |
| 14 | `src/reconcile_ml/decision.py:154` | `calibrate_online` | « Les seuils au premier franchissement. » |
| 15 | `src/reconcile_ml/pipeline.py:145` | `PipelineMatcher.process` | « Toute la journée ML, assemblée. » |
| 16 | `src/evaluation/metrics.py:143` | `evaluate` | « Juste = l'ensemble exact, sur une période jamais vue. » |
| 17 | `tests/test_timeline.py:231` | test anti-fuite | « Et on le prouve par un test. » |

> Les numéros de ligne correspondent au commit poussé sur tmp234 au moment de la rédaction. Si le code
> évolue, chercher le nom de la fonction (Ctrl+Shift+O dans VS Code pour les symboles du fichier).

---

## Minutage

Deux versions selon l'auditoire. Pour des data scientists, on condense le début et on garde le §7 bis en entier.

| § | Sujet | Général | Data scientists |
|---|---|---|---|
| 0 | Accroche et carte | 1 | 1 |
| 1 | Sources | 3 | 2 |
| 2 | Chargement et normalisation | 5 | 3 |
| 3 | Journal | 3 | 2 |
| 4 | Rejeu sans fuite | 6 | 5 |
| 5 | Allocation | 5 | 4 |
| 6 | Règles | 4 | 3 |
| 7 | ML (vue d'ensemble) | 8 | 4 |
| 7 bis | Logique ML en détail | — | 15 |
| 8 | Évaluation | 4 | 4 |
| 9 | Fabrication du modèle | 3 | (dans 7 bis) |
| 10 | Garanties | 3 | 3 |
| 11 | Limites et suites | 3 | 4 |
| | **Total** | **48** | **50** |
