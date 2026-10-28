# Fuites de données dans l'étape ML (étape 5)

> **Résumé.** L'inférence est bien protégée : toute lecture d'état passe par `LedgerState` avec `as_of`,
> et le test de troncature du journal le vérifie. Les fuites se trouvent **autour** de l'inférence :
> - dans l'**apprentissage** : étiquettes lues au-delà de la fin de la période, plis hors échantillon
>   non temporels ;
> - dans la **sélection** : la validation sert à cinq choix successifs, puis à rendre compte de leur
>   qualité ;
> - dans les **données** : attributs non datés (IBAN, variantes de nom, champs de facture tels
>   qu'extraits) et imputations datées avant l'arrivée de leur paiement ;
> - dans l'**évaluation** : l'indicateur affiché en tête choisit son seuil sur le test.
>
> Aucune de ces fuites n'est couverte par `test_pipeline_runs_without_leak`. Ce test rejoue un modèle
> **déjà entraîné** sur un journal tronqué : il ne contrôle ni l'apprentissage, ni les tables non datées
> (`tests/test_timeline.py:199-214` ne tronque ni `party_iban`, ni `name_variants`, ni les attributs
> des factures).
>
> « Certain » : vérifié dans le code. « Déduit » : raisonné à partir du code, sans mesure sur les données
> réelles. Les ordres de grandeur restent à mesurer (§ 9).

| # | Fuite | Où | Gravité | Nature |
|---|---|---|---|---|
| F1 | Validation réutilisée pour cinq choix, puis pour mesurer leur qualité | `model.py`, `pipeline.py`, `decision.py` | **haute** | certain |
| F2 | Attributs statiques photographiés à l'extraction (IBAN, noms, champs de facture) | `loader.py`, `indexes.py`, `features.py` | **haute** si les référentiels sont enrichis après coup | certain (mécanisme), déduit (impact) |
| F3 | Étiquettes lues au-delà de la fin de la période ; purge de 5 jours | `pipeline.py:290-294`, `settings.yaml` | moyenne | certain |
| F4 | Plis hors échantillon de la passe 2 tirés au hasard, pas dans le temps | `model.py` | moyenne | **corrigé** : walk-forward par défaut (`oof_scheme`), ancien schéma disponible |
| F5 | Imputation datée avant l'arrivée de son paiement : l'état à D la contient déjà | `events.py:122-124`, `state.py:237-265` | moyenne, dépend des données | certain (mécanisme) |
| F6 | `taux_automatisation_à_précision_cible` : seuil choisi sur le test | `metrics.py:168-174`, `ui/views/evaluation.py:57-61` | moyenne (présentation) | certain |
| F7 | Vocabulaires de catégories construits sur toutes les données | `features.py:100-107` | faible | certain |

---

## F1 — La validation sert à choisir et à mesurer

**Constat [certain].** La même période de validation sert, dans l'ordre :

1. à l'**arrêt précoce** de la passe 1 (`_train`, `src/reconcile_ml/model.py:76-77`, appelé `:115`) ;
2. au **nombre d'itérations** des plis hors échantillon, `pass1.best_iteration` (`model.py:121`) ;
3. à l'**arrêt précoce** de la passe 2 (`model.py:129`) ;
4. à l'**ajustement de la calibration isotonique** (`model.py:131-133`) ;
5. au **choix des seuils** : hors boucle (`calibrate_thresholds`, `src/reconcile_ml/pipeline.py:414-419`),
   puis en boucle (`online_thresholds`, `pipeline.py:425`). Ce sont ces derniers qui sont utilisés.

Puis on mesure sur ces **mêmes** données :

- `metrics.validation` : AUC, precision@1, MRR, après une calibration ajustée sur ces paires
  (`pipeline.py:429`) ;
- `calibration_validation.csv` : la courbe de fiabilité de l'isotonique sur ses propres données
  d'ajustement (`pipeline.py:444`). Elle est presque parfaite par construction ;
- `précision_auto_validation` (`pipeline.py:433`) : précision des auto-validations obtenues avec des seuils
  choisis pour atteindre la cible sur ces propositions. Elle vaut au moins 99,5 % **par construction**.

**Biais de sélection sur τ_high [certain].** `calibrate_online` retient le seuil **le plus permissif** dont
la précision observée atteint la cible, sur au moins `MIN_KIND_VOLUME = 20` propositions
(`src/reconcile_ml/decision.py:27`, `:182-185`). À 99,5 %, une tranche de 20 à 199 propositions doit être
correcte à 100 %. Parmi les quelque 2 000 seuils testés, celui qui passe est souvent celui qui a eu de la
chance. On choisit le maximum d'une quantité bruitée, donc la précision réelle de ce seuil est en espérance
plus basse que celle observée.

**Effet attendu [déduit].**
- Les métriques de `model.json` sont optimistes. Elles ne prouvent pas que le modèle généralise.
- Sur le test, la précision du ML à seuil fixé sera **inférieure** à la cible, d'autant plus que le volume
  d'un type (`single`, `partial`, `set`) est faible.
- Le test reste propre : le problème est la confiance accordée aux chiffres de validation, et un τ_high
  trop permissif.

**Correctifs.**
- Couper la validation en deux sous-périodes consécutives : **V1** pour l'arrêt précoce et l'isotonique,
  **V2** pour les seuils. Ne rapporter comme « hors échantillon » que des métriques calculées sur une
  période qui n'a servi à aucun choix.
- Choisir τ_high sur une **borne inférieure** de la précision, et non sur la précision observée. Par
  exemple, la borne de Clopper-Pearson à 95 % doit être ≥ 99,5 %, ce qui impose environ 600 propositions
  toutes correctes. Relever `MIN_KIND_VOLUME` en conséquence.
- Renommer ou retirer `précision_auto_validation`. En l'état, c'est une tautologie.

---

## F2 — Attributs statiques photographiés à l'extraction

**Constat [certain].** Plusieurs informations n'ont pas de date et sont visibles dès le premier jour du
rejeu, dans l'état où elles ont été **extraites** (en fin d'historique) :

| Attribut | Où il est construit | Ce qu'il alimente dans le ML |
|---|---|---|
| tous les IBAN d'une partie (`party_iban`) | `consolidate_parties`, `src/load/loader.py:322-333` ; `IbanIndex`, `src/allocation/indexes.py:236-253` | `iban_route`, `alloc_sig_iban`, `alloc_is_firm_debtor`, `iban_matches_invoice_debtor`, `alloc_score`, `alloc_rank` (`src/reconcile_ml/features.py:338-361`), et le périmètre des candidats K1 |
| toutes les variantes de nom (`name_variants`) | `loader.py:327-330` ; `NameIndex`, `indexes.py:204-224` | `alloc_sig_name`, `alloc_score`, `alloc_rank`, candidats K1 |
| référentiel des comptes techniques | `IbanIndex`, `indexes.py:242-253` | `iban_route` |
| champs de la facture : `client_reference`, `debtor_id`, `due_date`, `agreement_id` | table `invoice` telle que chargée | `ref_*_in_label`, candidats K2, `days_to_due`, `market`/`product`/`recourse`, débiteur de la facture |
| attributs du contrat : `market`, `product`, `recourse` | table `agreement` | famille `contract` |

L'index ne filtre que l'**existence** de la partie à D (`party_known_at`). Il ne filtre pas la date à
laquelle un IBAN ou un nom lui a été rattaché.

**Pourquoi c'est probablement la fuite la plus coûteuse [déduit].** En affacturage, le référentiel se
complète souvent **à partir des paiements eux-mêmes**. Un gestionnaire rapproche à la main un virement
venu d'un IBAN inconnu, puis rattache cet IBAN au débiteur. Il peut aussi corriger une référence de
facture ou réaffecter une facture à un autre débiteur. Dans le rejeu, l'IBAN est alors connu **avant** le
paiement qui l'a fait découvrir. Pour ce paiement, la feature `alloc_sig_iban` ou `iban_route =
DEBTOR_DIRECT` est en partie une **étiquette déguisée** : elle n'existe que parce que la bonne réponse a
été trouvée.

C'est la même cause que le point D1 de `docs/codebase/08_zones_dombre_et_dette.md`. Son effet sur le ML
n'y était pas décrit : le modèle apprend que « IBAN direct » prédit presque parfaitement, et il calibre ses
seuils là-dessus. En production, les IBAN encore inconnus sont précisément ceux des paiements difficiles.

**Correctifs.**
- Demander à l'équipe si `party_iban`, les noms et les champs de facture ont un **historique** (date
  d'ajout ou de modification). Si oui, les dater et en faire des événements du journal
  (`PARTY_IBAN_ADDED`, etc.).
- Sinon, **estimer** la date d'apparition d'un IBAN par le premier paiement imputé à ce débiteur depuis
  cet IBAN, et ne le rendre visible qu'à partir de ce jour.
- Mesurer la dépendance : voir § 9, mesure 2.

---

## F3 — Étiquettes lues au-delà de la fin de la période

**Constat [certain].** `label_pairs` (`src/reconcile_ml/pipeline.py:290-294`) étiquette les paires à partir
de **toute** la table `imputation`, sans condition sur `updated_at`. Un paiement arrivé à la fin de la
validation est donc étiqueté par une imputation prononcée pendant le test, ou après.

La purge entre périodes est de 5 jours (`split.purge_days`, `config/settings.yaml`), alors qu'un paiement
reste en attente jusqu'à 60 jours (`retention_days`) et que l'imputation manuelle prend souvent plusieurs
semaines. La purge ne remplit donc pas son rôle déclaré : « éviter qu'un groupe d'imputations chevauche
deux blocs ».

**Effet [déduit].**
- Les seuils sont calibrés en partie sur des décisions de gestionnaires prises **pendant la période de
  test**.
- Le jeu d'entraînement contient plus d'étiquettes que n'en aurait un réentraînement réel à la fin de la
  validation. En production, les paiements récents non encore imputés seraient inconnus ; ici, ils ont
  leur vérité finale.
- Les paiements difficiles, imputés tard, ne sont étiquetés qu'**grâce** au futur. Le modèle les voit
  mieux qu'il ne les verrait en conditions réelles.

**Correctifs.**
- Étiqueter avec un **horizon** : `imputation.updated_at ≤ fin de la période + h`, où h est la
  purge.
- Fixer la purge à au moins le P95 du délai `updated_at − arrivée du paiement`, mesuré sur le train.
- Écarter de l'apprentissage les paiements dont l'imputation n'est pas connue à l'horizon, au lieu de les
  étiqueter comme négatifs.

---

## F4 — Plis hors échantillon non temporels (passe 2)

> **Statut : corrigé.** Les scores OOF de la pass 2 sont désormais produits en walk-forward
> (`reconcile_ml.training.oof_scheme: walk_forward`, `walk_forward_scores`, `src/reconcile_ml/model.py`) : chaque
> mois du train est scoré par un modèle appris sur les paiements arrivés avant `Dₖ − purge`, avec les seuls labels
> connus à `Dₖ` ; les paiements d'issue encore inconnue sont écartés. `walk_forward_step_months` règle la cadence
> des modèles et `walk_forward_window_months` la fenêtre. L'ancien schéma reste disponible avec
> `oof_scheme: random_folds` (c'est aussi le défaut d'un appel direct à `PairModel.fit`). Le constat ci-dessous
> décrit ce schéma.

**Constat [certain].** Les scores de passe 1 qui nourrissent les features de compétition du train sont
obtenus par 3 plis tirés au hasard par `payment_id` (`src/reconcile_ml/model.py:119`). Pour prédire un
paiement de janvier, un modèle de pli a donc vu des paiements de février à août.

Ce qui est évité : un paiement n'est jamais prédit par un modèle qui a vu ses propres paires (le
groupe est le paiement, et chaque paiement n'est enregistré qu'une fois, `pipeline.py:107`, `:393`).

Ce qui fuit, par ordre d'importance [déduit] :
1. **Le futur du même débiteur.** Les features comportementales (`debtor_*`), `bankroll_code`,
   `channel`, `market` et les signaux d'allocation forment une quasi-empreinte du débiteur. Les arbres
   apprennent donc des régularités propres à un débiteur à partir de ses paiements **postérieurs**.
   En production, la passe 1 n'a jamais vu ces paiements.
2. **La dérive dans le temps.** Les plis mélangent les mois. Les scores hors échantillon ne contiennent
   donc pas la dégradation qu'on observe sur des paiements futurs (nouveaux débiteurs, nouveaux
   libellés, changements de pratique).
3. **Les mêmes factures.** Paiements partiels successifs d'une facture, n↔1, une facture candidate pour
   plusieurs paiements. L'effet est faible : le modèle n'a pas d'identifiant de facture, et
   `min_data_in_leaf = 50` limite la mémorisation.
4. **Le nombre d'itérations des plis** vient de la validation (`pass1.best_iteration`, `model.py:121`,
   voir F1).

Hors fuite, il y a aussi un **décalage de distribution** : chaque modèle de pli apprend sur 2/3 du train,
alors que la passe 1 utilisée en validation et en production apprend sur tout le train, au même nombre
d'itérations. Les échelles de score diffèrent légèrement entre la passe 2 apprise et la passe 2 appliquée.

**Effet [déduit].** Les scores hors échantillon du train sont plus optimistes que ceux que la passe 1
produit en réalité sur des paiements futurs (validation, test, production). La passe 2 apprend une
relation entre la marge ou le rang et la vérité qui ne se transpose pas telle quelle. Elle surpondère
`score_margin` et `rank_in_payment`. C'est la fuite classique des modèles empilés, que la décision 4 de
`docs/choix_etape_ml.md` voulait justement éviter : elle n'est corrigée que pour le cas « même
paiement ».

**Portée.** La fuite est confinée aux features d'**apprentissage** de la passe 2. En validation, en test
et en production, les features de compétition viennent de la passe 1 entraînée sur le seul train,
appliquée à des paiements postérieurs : elles ne contiennent pas de futur. F4 **ne gonfle donc pas** les
métriques de test. Elle dégrade le modèle, et donc le taux d'automatisation atteignable à précision
fixée. Les seuils calibrés sur la validation en tiennent compte : ils sont pris sur les vrais scores.

**Correctifs.**
- Produire les scores de passe 1 du train **comme en production**, en suivant la boucle quotidienne
  (« walk-forward »). Au jour D, la passe 1 qui score le lot ne doit connaître que les paires
  enregistrées avant D, avec les étiquettes connues avant D : les imputations d'horodatage
  strictement antérieur à D, selon la même convention que `LedgerState`.
  - Réentraîner chaque jour serait trop coûteux. On réentraîne aux dates D₀ < D₁ < …, à la cadence
    prévue en production (par exemple chaque mois). Le modèle appris à Dₖ score les jours de
    `[Dₖ, Dₖ₊₁)`.
  - Il n'est pas nécessaire de refaire un rejeu. Les features de passe 1 ne dépendent pas du modèle, et
    `DatasetRecorder` ne décide que par les règles. Le jeu qu'il enregistre (colonne `day`) contient donc
    déjà tout ce que la boucle aurait vu. Le walk-forward se fait sur ce jeu, à une condition : filtrer
    les étiquettes sur `imputation.updated_at < Dₖ`. C'est le correctif de F3 appliqué à chaque Dₖ.
  - Les jours qui précèdent le premier modèle n'ont pas de score : soit on les exclut de la passe 2,
    soit on les traite comme un rang manquant.
- À défaut, grouper les plis par **débiteur**, et non par paiement. C'est moins juste : ni la dérive
  dans le temps ni l'horizon des étiquettes ne sont respectés.

---

## F5 — Imputation datée avant l'arrivée de son paiement

**Constat [certain].** Le journal place `IMPUTATION_APPLIED` à `updated_at`
(`src/load/events.py:122-124`) et `PAYMENT_RECEIVED` à `booking_date`, sinon `value_date`
(`events.py:84-86`). Rien n'impose que la première soit postérieure à la seconde. Le contrôle qualité
compte ces cas (`avant_connaissance_du_paiement`, `src/load/quality.py:87-89`) mais ne les corrige pas.

Pour un tel paiement, l'état figé au jour de son arrivée **contient déjà sa propre imputation** :
- le restant dû de la vraie facture est déjà réduit ou soldé (`open_balance_at`,
  `src/timeline/state.py:397-400`). Cela modifie `amount_diff`, `balance_is_partial`, `ratio_to_initial`,
  ou retire la facture des candidats ;
- les agrégats comportementaux du débiteur l'incluent : `debtor_payment_count`, `debtor_grouping_rate`,
  `debtor_partial_payment_rate`, `debtor_ref_citation_rate`, `debtor_mean_payment_delay`
  (`_add_window`, `state.py:237-265`) ;
- `payment_imputed_at` le sort du reliquat dès le lendemain (`src/timeline/loop.py:115`).

Le cas le plus probable : `booking_date` non mappé et `value_date` postérieure à la connaissance du
virement. Les dates de valeur futures existent en banque.

**Effet [déduit].** Ce n'est pas forcément une fuite « favorable » : la vraie facture peut disparaître des
candidats. Mais les features comportementales portent une information sur l'étiquette du paiement courant,
et l'effet n'est pas le même en production.

**Correctifs.**
- Rendre le contrôle **bloquant** au-delà d'un seuil. Sinon, décaler l'événement
  `IMPUTATION_APPLIED` à `max(updated_at, arrivée du paiement + ε)`, et le signaler.
- Mapper `booking_date` si elle existe (question déjà ouverte dans `docs/codebase/08`, § 5).

---

## F6 — L'indicateur principal choisit son seuil sur le test

**Constat [certain].** `taux_automatisation_à_précision_cible` est lu sur `precision_curve`, qui trie les
propositions **du test** par score et prend le point le plus à droite encore au-dessus de la cible
(`src/evaluation/metrics.py:168-174`, `:138-140`). C'est un seuil oracle choisi sur le test.
`RESULTATS.md` (§ 6.9, point 4 de la conclusion) le qualifie bien de « plafond ». Mais l'interface
l'affiche **en premier**, sous le titre « Taux d'automatisation à précision ≥ 99,5 % »
(`src/ui/views/evaluation.py:57-61`). Le taux réellement obtenu avec les seuils de validation,
`taux_automatisation`, n'arrive qu'en second.

**Effet.** Quiconque lit l'écran ou le rapport sans lire `RESULTATS.md` prend un plafond optimiste pour le
résultat du POC. Le brief demande le taux **à précision fixée** : le seul chiffre légitime est
`taux_automatisation`, accompagné de `précision`.

**Correctif.** Afficher en tête `taux_automatisation` et `précision`. Garder la courbe comme diagnostic,
intitulé « plafond a posteriori ».

---

## F7 — Vocabulaires de catégories construits sur toutes les données

**Constat [certain], déjà relevé** (`docs/codebase/03_gestion_des_donnees.md` § 6.2, D3 de
`docs/codebase/08`). Les listes de `channel`, `bankroll_code`, `market`, `product` et `recourse`
viennent de toutes les lignes, test compris (`src/reconcile_ml/features.py:100-107`), comme les devises
(`:97`).

**Effet [déduit] : faible.** Une valeur qui n'apparaît qu'au test reçoit un code que LightGBM n'a jamais vu
(branche « autre »). En production, elle serait inconnue (`NaN`). Les deux comportements sont proches.

**Correctif.** Construire ces listes sur les seules lignes du train. Cela ne coûte rien.

---

## 8. Vérifié, sans fuite

| Point | Pourquoi ce n'est pas une fuite |
|---|---|
| Restant dû des factures | reconstruit par les imputations ; `current_amount` retiré au chargement (`loader.py:360-364`) |
| Agrégats comportementaux | fenêtre `[as_of − 180 j, as_of)`, expirée à l'avance (`state.py:267-271`, `:348-374`), hors F5 |
| Client files | lignes visibles seulement après `received_at` (`state.py:333-340`) |
| Vocabulaire des références (`ReferenceIndex`) | contient les clés des factures futures, mais chaque usage filtre ensuite les factures ouvertes à D. `_pair_load` ne sert qu'au découpage en blocs. `n_label_numbers` a déjà été corrigé (`features.py:118-120`) |
| Étiquetage | seconde passe séparée, après le rejeu (`pipeline.py:392`), hors F3 |
| Features de compétition en validation et en inférence | calculées avec le modèle de passe 1 complet, comme en production (`model.py:127-128`, `:93-99`) |
| Seuils du test | calibrés sur la validation uniquement (`online_thresholds`) |
| Test vs entraînement | le rejeu d'apprentissage s'arrête à `valid_p.end` (`pipeline.py:387`) |

---

## 9. Mesurer l'impact

Chaque fuite se mesure par un écart. Les mesures se font sur les données réelles, hors tests unitaires
(`CLAUDE.md` § 10).

1. **F1** : refaire l'apprentissage avec la validation coupée en V1 et V2, puis comparer la précision du
   ML sur le test aux seuils actuels et aux nouveaux seuils.
2. **F2** : rejouer le test **sans** le signal IBAN pour les couples (débiteur, IBAN) dont le premier
   paiement imputé est postérieur à D, et comparer le taux d'automatisation et la précision. Mesurer
   aussi l'importance de `iban_route` et `alloc_sig_iban` (`feature_importance.csv`).
3. **F3** : part des paires positives dont l'imputation est postérieure à la fin de leur période.

   ```python
   imp = data.tables["imputation"][["payment_id", "invoice_id", "updated_at"]]
   pos = ds[ds["label"] == 1].merge(imp, on=["payment_id", "invoice_id"])
   end = {"train": train_p.end, "validation": valid_p.end}
   late = pos["updated_at"] > pd.to_datetime(pos["period"].map(end)) + pd.Timedelta(days=1)
   late.groupby(pos["period"]).mean()
   ```

4. **F4** : comparer la distribution de `score_margin` sur le train (scores hors échantillon) à celle de
   la validation. Un écart net confirme la fuite.
5. **F5** : lire le compte `avant_connaissance_du_paiement` du rapport de chargement. S'il est nul, F5 est
   sans objet.

**Test à ajouter.** Un test anti-fuite de l'**apprentissage**. On construit le jeu d'entraînement sur un
journal complet et sur un journal tronqué à la fin de la validation, avec les mêmes catégories. Les
features doivent être identiques ; les étiquettes doivent l'être aussi une fois l'horizon de F3 appliqué.
