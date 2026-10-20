# L'étape ML de bout en bout : ce que fait chaque brique, et pourquoi

> Pour chaque étape : **ce qu'elle fait**, **pourquoi on en a besoin**, **comment elle marche**, **ce qu'on
> mesure**, et **ses limites**. Les chiffres viennent du run de référence sur le synthétique à 2 M de paiements, sauf
> mention contraire ; ils restent à confirmer sur les données réelles. Les références de code sont de la forme
> `fichier:ligne`.
>
> Documents voisins : `docs/choix_etape_ml.md` (les décisions de conception), `docs/fuites_ml.md` (le leakage),
> `notebooks/review_ml.ipynb` (toutes les mesures citées ici, recalculables).

---

## 0. Vue d'ensemble

```
                          ┌──────────────────────── fabrication du modèle (une fois, puis à chaque retrain) ────────────────────────┐
 journal + LedgerState ─► │ 4. replay train + validation ─► dataset de paires ─► 5. pass 1 ─► 6. folds OOF ─► 7. pass 2              │
                          │                        labels (2de passe)                            8. isotonique   10. seuils online   │
                          └──────────────────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                                                                 │ modèle + seuils + vocabulaires (model.json)
                          ┌──────────────────────────────── chaque jour D (replay = production) ─▼──────────────────────────────────┐
 lot du jour D ─► allocation ─► règles ─► 1. résiduel ─► 2. candidats ─► 3. features ─► score (pass 1 → compétition → pass 2)      │
 (nouveaux + reliquat)                                     ─► 9. propositions ─► 11. décision ─► arbitrage du lot ─► auto / revue / manuel │
                          └────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                                                 │
                                                                    13. évaluation (backtest du test)
```

**L'idée directrice.** Le problème est un **retrieval + ranking** décidé **à précision fixée**. Pour un paiement
donné :
1. on récupère un petit nombre de factures plausibles ;
2. on les classe ;
3. on n'auto-valide que lorsque la meilleure proposition est assez sûre pour tenir 99,5 % de précision.

Tout le reste part en revue ou en manuel. Une auto-validation fausse coûte cher, une auto-validation manquée ne coûte
qu'un traitement manuel : le système est conçu pour **refuser de décider** dès qu'il doute.

Le même code sert au replay de l'historique et à la boucle de production (`src/timeline/loop.py:1-19`). Le backtest
mesure donc ce que ferait la production.

---

## 1. L'entrée : le résiduel des règles

**Ce que ça fait.** Chaque jour, le ML ne reçoit que les paiements du lot que les règles R1–R5 n'ont pas résolus
(`_ResidualMixin._residual`, `src/reconcile_ml/pipeline.py:59-71`). Il reçoit aussi leur **allocation** : les débiteurs
candidats, avec leur score et le signal qui les a désignés.

**Pourquoi.**
- Les règles sont **déterministes et auditables**. Ce qu'elles résolvent avec une solution unique, elles le résolvent
  avec une précision quasi parfaite sur les cas simples (R2, référence exacte unique : 99,99 % ; R3, montant exact
  unique : 99,89 %). Il n'y a aucune raison de confier ces cas à un modèle.
- Le ML apporte de la valeur **là où il faut pondérer des indices faibles et contradictoires** : référence bruitée,
  montant approché, paiement groupé, partiel.
- C'est aussi la baseline à battre, demandée par le brief.

**Ce qu'on mesure.** Sur le test, 339 193 paiements :
- les règles en automatisent **62,75 %** ;
- 115 695 paiements passent par le ML, qui en auto-valide 25 018, soit **+7,4 pts** à 99,64 % de précision ;
- au total : **70,13 %** automatisés, à 99,61 %.

**Limite.** Le ML hérite des décisions des règles. Une règle peu précise (R5 seule : 93,8 % ; R4 : 98,4 %) passe avant
lui. La question de désactiver ces règles se pose : le brief le prévoit pour une règle sous la cible, et un backtest
sans elles chiffrerait la décision.

---

## 2. Les candidats (retrieval)

**Ce que ça fait.** Pour chaque paiement du résiduel, on construit une liste d'au plus 50 factures plausibles
(`Featurizer.candidates`, `src/reconcile_ml/features.py:136` ; `_candidates_rows`, `:178` ; `_cap`, `:234`).

**Pourquoi.** Scorer toutes les factures ouvertes serait impossible (des millions par jour) et inutile : la vraie
facture est presque toujours chez un des débiteurs alloués, ou désignée par une clé précise. Le brief interdit
d'ailleurs toute boucle sur toutes les factures.

**Comment.**

Quatre sources, réunies. Chaque paire garde la trace de ses sources (`src_*`), qui sont aussi des features :

| Source | Règle | Part des paires | Part des vraies factures trouvées | Vraies factures trouvées par elle seule |
|---|---|---|---|---|
| débiteur alloué | factures ouvertes des 3 premiers débiteurs de l'allocation, échéance de D − 180 j à D + 30 j | 88,2 % | 98,3 % | 20,2 % |
| référence citée | factures dont une clé de référence apparaît dans le libellé, sans fenêtre | 12,8 % | 52,2 % | 1,1 % |
| montant exact | factures ouvertes de même montant, créées à ± 90 j | 2,4 % | 23,5 % | 0,3 % |
| client file | factures citées dans le client file rattaché | 2,3 % | 11,8 % | 0,2 % |

Les pourcentages portent sur la validation.

Trois **filtres durs** s'appliquent ensuite :
- même devise ;
- facture ouverte à D, **nette des réservations du moteur** : une facture déjà prise par une auto-validation n'est plus
  disponible ;
- contrat actif.

Le **plafond de 50** garde d'abord les sources précises (référence, montant, client file). Il complète ensuite avec les
factures du débiteur les plus proches en montant et en échéance parmi celles qui « tiennent » dans le paiement, ce qui
préserve les paiements groupés.

**Ce qu'on mesure : le recall des candidats.** C'est la part des paiements dont **toutes** les vraies factures sont
candidates : **79,8 %** en validation, 79,7 % en test.

C'est le **plafond de tout l'étage ML**. Un paiement dont la vraie facture manque ne peut pas être rapproché juste,
quel que soit le modèle. Le plus souvent il part en manuel ; s'il est auto-validé, c'est une erreur, déjà comptée dans
la précision mesurée.

**Limites.**
- C'est le **premier levier de gain**, avant tout réglage du modèle. Il faut diagnostiquer les manqués par cause :
  plafond atteint, fenêtre d'échéance, débiteur absent de l'allocation, factures d'un 1↔n hors fenêtre.
- Le recall chute sur les paiements groupés, dont une facture peut sortir de la fenêtre.

---

## 3. Les features

**Ce que ça fait.** Une ligne par paire (paiement, facture candidate), avec 54 features en huit familles plus la base
(`FAMILIES`, `src/reconcile_ml/features.py:29` ; `Featurizer.features`, `:294`).

**Pourquoi.** Le modèle doit répondre à « cette facture est-elle celle que ce paiement règle ? ». Il a donc besoin de
mesures **relatives** de la paire (écarts, ratios, correspondances) plutôt que de valeurs brutes, et du **contexte** du
débiteur.

**Comment.**

| Famille | Contenu | Pourquoi |
|---|---|---|
| base (6) | sources de la paire, nombre de candidats, rang de l'écart de montant dans le paiement | par quelle porte la facture est arrivée ; densité de la concurrence |
| montant (10) | écart, ratio, montant exact, couverture ; formes typiques d'écart (escompte de 0,5 à 3 %, frais de 5 à 40 €, retenue de 4 à 6 %) ; facture déjà partiellement payée | le signal le plus fort : les débiteurs paient rarement « au hasard » |
| temporel (4) | jours à l'échéance, depuis la création, paiement avant création ; retard rapporté au retard habituel du débiteur (z-score) | un paiement arrive autour de l'échéance, selon les habitudes du débiteur |
| textuel (6) | référence complète, clé ou référence interne dans le libellé ; forme du libellé | la référence est l'indice le plus précis quand elle existe |
| identité (4) | route IBAN, IBAN du débiteur de la facture, canal, portefeuille | qui paie, et par où |
| comportemental (8) | délai moyen et écart-type, taux de partiels, de regroupement, de citation de référence, nombre de paiements, encours ; fenêtre **[D − 180 j, D)** | les habitudes généralisent aux factures nouvelles d'un débiteur connu |
| contrat (3) | marché, produit, recours | segments aux pratiques différentes |
| allocation (9) | rang et score du débiteur dans l'allocation, allocation ferme, signaux qui l'ont désigné | le résultat de l'étape 3, dont la fiabilité varie selon le signal |
| client file (4) | client file présent, facture citée, écart au montant de la ligne | l'information du client, quand elle existe |

Trois principes :
- **Tout est calculé sur l'état figé à D** (`LedgerState`, `as_of`) : aucune feature ne lit le futur. Le test
  anti-fuite le vérifie (truncation test : mêmes features avec un journal tronqué).
- **Pas d'identifiant, pas de montant brut** : pas de mémorisation, et une indépendance d'échelle.
- **Catégories à vocabulaire figé**, sauvegardé avec le modèle. Une valeur inconnue devient `NaN`, et les valeurs
  manquantes restent natives, sans imputation arbitraire.

**Ce qu'on mesure.** Gain de la pass 1 : `alloc_rank` 22,3 %, `ref_full_in_label` 16,0 %, `amount_exact` 9,1 %,
`ref_key_in_label` 8,9 %, `alloc_sig_name` 6,8 %, `client_file_line_amount_diff` 6,5 %. Aucune feature seule ne
s'approche du modèle complet : la meilleure atteint 0,585 de precision@1, contre 0,951 (audit du notebook
`ml_pas_a_pas`).

**Limites.**
- `client_file_total_matches_payment` est une copie de `has_client_file` : il n'y a que 53 informations distinctes.
- Les attributs **statiques** (IBAN des parties, noms, champs de facture) ne sont pas datés : c'est le leakage F2
  (`docs/fuites_ml.md`).
- `debtor_client_file_rate`, demandée par le brief, manque.

---

## 4. Le dataset d'entraînement

**Ce que ça fait.** Le dataset est construit en **rejouant la boucle quotidienne** sur le train et la validation
(`DatasetRecorder`, `src/reconcile_ml/pipeline.py:92-122`, appelé par `fit_ml`, `:359`).
- Chaque paiement du résiduel est enregistré **le jour de son arrivée**, avec ses candidats et leurs features
  calculées sur l'état à D.
- On garde 50 % des paiements, par un tirage déterministe sur le hachage de l'identifiant (`_sampled`, `:43`).
- Les labels sont posés **dans une seconde passe séparée**, à partir de la table `imputation` (`label_pairs`, `:290`) :
  1 si le couple (paiement, facture) a été réellement imputé, 0 sinon.

**Pourquoi.**
- **Rejouer plutôt que reconstruire à part** : les features d'apprentissage sont calculées par le même code, sur le
  même état daté, que celles de la production. Deux implémentations séparées finissent toujours par diverger.
- **Labelliser à part** : pendant le replay, aucune information sur la vérité ne peut entrer dans les features.
- **Tous les négatifs du paiement**, sans échantillonnage : la tâche est de départager les candidats d'un même
  paiement, et les features de compétition (§ 6) en ont besoin.

**Ce qu'on mesure.**
- 24,9 M de paires sur le synthétique ; sur les données réelles : 2,2 M de paires de train et 0,78 M de validation.
- Quelques pourcents de positifs.
- Le dataset est partitionné par mois et versionné : une empreinte du journal, de la featurisation et des réglages
  permet de le reprendre sans replay (`dataset_key`, `:323`).

**Limites.**
- Les labels sont lus **sans horizon de date**. Un paiement de fin de période reçoit sa vérité finale, même prononcée
  plus tard (F3).
- **Écart entre training et production.** Le training ne voit que l'état **au jour d'arrivée**, alors qu'en production
  un paiement en reliquat est rescoré chaque jour, sur un état qui a changé. Les seuils online (§ 10) le compensent pour
  la décision, mais le modèle n'apprend jamais sur ces états.

---

## 5. La pass 1 : un score par paire

**Ce que ça fait.** Un LightGBM binaire (log-loss) sur les 54 features donne, pour chaque paire, un score : « cette
facture est-elle une vraie imputation de ce paiement ? » (`_train`, `src/reconcile_ml/model.py:69` ; `PairModel.fit`,
`:111`).

**Pourquoi ce choix.**
- **Gradient boosting** : des données tabulaires hétérogènes (ratios, catégories, valeurs manquantes), des
  interactions non linéaires (un petit écart de montant ne compte que si la référence manque…), rapide et déterministe.
- **Pointwise binaire plutôt que ranking (LambdaMART)** : la décision compare le score à un seuil **absolu**, commun à
  tous les paiements. Un objectif listwise optimise l'ordre dans le groupe, pas l'échelle : un 0,9 ne voudrait pas dire
  la même chose d'un paiement à l'autre.

**Comment.**
- Learning rate 0,05, 63 feuilles, 50 lignes minimum par feuille, `feature_fraction` 0,9.
- Au plus 400 arbres, avec **early stopping** à 30 itérations sur la log-loss de validation.
- Mode déterministe, seed fixe ; threads réglés sur les CPU réellement disponibles du pod.

**Ce qu'on mesure.**
- 257 arbres.
- En validation : AUC 0,9988, **precision@1 95,7 %**, MRR 0,971, calculés sur les paiements qui ont au moins un vrai
  candidat.
- La precision@1 et le MRR mesurent le **classement dans le paiement**, ce qui compte pour la décision ; l'AUC mélange
  tous les paiements.

**Limites.**
- Les hyperparamètres ne sont pas optimisés : ce n'est pas le levier principal.
- Le score d'une paire est **isolé** : il ignore la concurrence entre candidats. C'est ce que corrige la pass 2.

---

## 6. Les folds OOF et les features de compétition

**Ce que ça fait.** On calcule, pour chaque paire, trois features qui la situent face aux autres candidats du même
paiement (`competition_features`, `src/reconcile_ml/model.py:37`) :
- `rank_in_payment` : son rang par score de pass 1 ;
- `score_margin` : l'écart entre son score et celui du meilleur **autre** candidat ;
- `score_best_other` : le score de ce meilleur autre candidat.

**Pourquoi.** Un score de 0,8 n'a pas le même sens si le meilleur concurrent est à 0,79 (ambigu) ou à 0,1 (net).
Cette information est **au niveau du paiement** ; un modèle pointwise ne peut pas la déduire seul.

**Pourquoi des folds.** Sur le train, on ne peut pas calculer ces features avec la pass 1 entraînée sur ce même
train : elle a vu les labels, ses scores sont trop nets (precision@1 en in-sample 0,991 contre 0,949 en validation sur
le 40 k), et la pass 2 apprendrait des marges qu'elle ne reverra jamais.

On découpe donc le train en **3 folds par paiement** (`model.py:119-125`) :
- chaque paire est scorée par un modèle de fold qui n'a vu **aucun label de son paiement** (score OOF) ;
- les modèles de fold utilisent le nombre d'itérations de la pass 1 et sont **jetés** ensuite ;
- en validation, en test et en production, la compétition vient de la pass 1 complète, comme en production.

**Pourquoi par paiement.** L'unité du fold doit être l'unité d'agrégation des features de compétition. Sinon, le
« meilleur concurrent » d'une paire pourrait être scoré par un modèle qui a vu son label.

**Ce qu'on mesure** (échantillon du 2 M) : precision@1 de 99,4 % en in-sample, **92,9 % en OOF**, 92,9 % en
validation. Les scores OOF ressemblent à ceux de la validation, pas à ceux de l'in-sample : pas de stacking leakage.

**Limites.**
- Les folds ne sont **pas temporels** : un paiement de janvier est scoré par un modèle qui a vu février à août. C'est un
  look-ahead leakage (F4), confiné au training de la pass 2. Mesuré : une pass 2 nourrie en walk-forward fait au moins
  aussi bien, donc la correction ne coûte rien.
- Leur nombre d'itérations vient de l'early stopping **sur la validation**.

Les deux points sont détaillés dans `docs/fuites_ml.md` et le notebook de séance (§ 4).

---

## 7. La pass 2 : le score avec le contexte du paiement

**Ce que ça fait.** Un second LightGBM, avec les mêmes hyperparamètres, sur 57 features : les 54 de la pass 1 et les
3 de compétition (`model.py:126-129`). Son **score brut** est celui qu'utilise la décision.

**Pourquoi.** C'est le stacking classique : la pass 1 résume les 54 features en un score ; la pass 2 apprend comment
la **concurrence** modifie la confiance.

**Ce qu'on mesure.**
- 140 arbres.
- Gain sur la pass 1 : modeste en precision@1 (+0,2 à +0,3 pt), net en log-loss (−8 % sur le 40 k).
- Les features de compétition portent l'essentiel du gain de la pass 2, ce qui est attendu : elles réinjectent le
  score de la pass 1.

**Limites.** Son apport réel se joue dans la **marge**, donc dans le taux d'automatisation à précision fixée, plus que
dans la precision@1. Une ablation (`second_pass: false`) sur le test le chiffrerait.

---

## 8. La calibration isotonique

**Ce que ça fait.** Une régression isotonique, ajustée sur la validation (`model.py:131-133`), transforme le score
brut de pass 2 en **probabilité calibrée** : « parmi les paires de validation à ce niveau de score, x % étaient
vraies ».

**Comment.** On cherche la fonction **croissante** f qui minimise `Σ (yᵢ − f(sᵢ))²`, par l'algorithme PAVA (*pool
adjacent violators*) :
1. on trie les paires par score ;
2. tant que deux blocs voisins violent la monotonie, on les fusionne en prenant leur moyenne.

Le résultat est une fonction **en escalier**.

**Pourquoi le brief la demande.** LightGBM en log-loss sort déjà une quasi-probabilité, mais elle peut être décalée :
early stopping, classes déséquilibrées, et surtout le stacking. La pass 2 est entraînée sur une compétition issue des
scores OOF, puis appliquée à une compétition issue de la pass 1 complète.

**Pourquoi la décision ne l'utilise pas** (`src/reconcile_ml/decision.py:1-16`) :
1. **Elle ne change pas l'ordre.** f est croissante : un seuil sur p équivaut à un seuil sur le score brut.
2. **Elle crée des paliers là où se joue la décision.** Dans les 5 % de scores les plus hauts, il reste 46 valeurs
   distinctes au lieu de 9 186 (17 contre 2 944 sur le 40 k). Un seuil ne pourrait se placer qu'entre ces paliers.
3. **Hors échantillon, elle dégrade la log-loss.** Ajustée sur la première moitié de la validation et mesurée sur la
   seconde : 0,0116 contre 0,0112 en brut. LightGBM était déjà bien calibré.
4. **Une probabilité de paire ne répond pas à la question métier**, même parfaite :

| La probabilité calibrée mesure… | La cible métier porte sur… |
|---|---|
| une paire vraie | un **ensemble exact**. Une paire vraie dans un paiement 1↔n donne une proposition `single` fausse, car incomplète. |
| un niveau local de score | la précision **moyenne de tout ce qui dépasse τ** |
| toutes les paires | seulement les propositions de marge ≥ δ |
| un instant donné | le **premier jour** où le paiement franchit le seuil (§ 10) |

D'où des seuils calibrés **directement par la précision des auto-validations**.

**À quoi elle sert.**
- La **lecture** : en revue, « p = 0,93 » parle à un gestionnaire.
- Le **suivi** : le tableau `ml_calibration` du test détecte une dérive de calibration.

**Limites.** La courbe de fiabilité du rapport est mesurée sur les données mêmes de l'ajustement : elle est presque
parfaite par construction (F1). L'idéal serait de l'ajuster sur une sous-période dédiée, ou de passer au Platt
scaling, lisse et sans paliers.

---

## 9. Des paires aux propositions

**Ce que ça fait.** Pour chaque paiement, on construit des **propositions** concurrentes (`propose`,
`src/reconcile_ml/decision.py:30-77`). On garde la meilleure, avec sa **marge** au second.

**Pourquoi.** On n'impute pas des paires mais des **ensembles** : un paiement peut solder une facture, en payer une
partie, ou en régler plusieurs. La décision doit porter sur l'ensemble complet, puisque c'est lui qui est juste ou faux.

**Comment.** La tolérance vaut le maximum de 5 € et de 3 % du paiement.

| Type | Construction | Score |
|---|---|---|
| `single` | une des 2 meilleures factures qui absorbent le paiement et le soldent à la tolérance près | score de la paire |
| `partial` | une des 2 meilleures factures plus grandes que le paiement | score de la paire |
| `set` | recherche bornée (DFS, `near_subsets`, `src/reconcile_rules/subset.py:79`) parmi les 25 meilleurs candidats, au plus 5 factures dont la somme égale le paiement à la tolérance près ; cardinalité minimale préférée | **moyenne** des scores de ses paires |

- La meilleure proposition l'emporte, et sa marge est l'écart au score de la meilleure proposition **différente**.
- **Règle d'ambiguïté** : une facture seule qui n'explique le paiement qu'à un escompte près, face à un ensemble qui
  l'explique exactement, est forcée à une marge 0, donc envoyée en revue, quel que soit l'écart de score.
- **n↔n** (`_n_to_n`, `src/reconcile_ml/pipeline.py:205`) : les paiements non auto-validés d'un même débiteur
  (allocation ferme), arrivés à moins de 72 h d'écart, sont additionnés. Un ensemble unique de factures qui égale leur
  somme devient une proposition **en revue seulement**.

**Limites.**
- Le score d'un `set` est une moyenne de paires, pas une probabilité calibrée de l'ensemble. Son seuil propre compense
  en partie, mais il ne tient pas sur le test (§ 10.5). Un modèle au niveau de l'ensemble serait la vraie solution.
- `partial` n'est jamais auto-validé (§ 10).

---

## 10. Les seuils : offline puis online

### 10.1 Le principe commun

Pour chaque **type** de proposition, on cherche le seuil τ_high **le plus permissif** dont la précision des
auto-validations atteint **99,5 %**. Les seuils sont par type, parce que les fiabilités diffèrent : un `partial` peut
masquer un n↔n, et le score d'un `set` est une moyenne.

Il faut au moins 20 propositions (`MIN_KIND_VOLUME`, `decision.py:27`). Sans seuil qui tienne la cible, τ = 1,01 :
**jamais d'auto-validation** pour ce type.

### 10.2 Offline : une photo par paiement

`calibrate_thresholds`, `decision.py:97-133`, appelé par `pipeline.py:411-420`.
1. On prend le dataset de validation, c'est-à-dire les paires du **jour d'arrivée**.
2. On score, et on construit une proposition par paiement.
3. On trie par score décroissant et on calcule la précision cumulée depuis le haut.
4. τ est le plus petit score où elle vaut encore ≥ 99,5 %, parmi les propositions de marge ≥ δ.

C'est simple et rapide, mais **ce n'est pas ce que fait la production**.

### 10.3 Ce que l'offline ne voit pas : le rescoring quotidien

En production, un paiement non résolu reste en reliquat jusqu'à 60 jours et **il est rescoré chaque jour**. Son score
bouge :
- des factures sont créées ;
- d'autres sont soldées ;
- un client file arrive ;
- les agrégats du débiteur changent.

Il est auto-validé **le premier jour où son score franchit τ**, avec la proposition de ce jour-là. Chaque jour est une
nouvelle chance de franchir le seuil, y compris avec une mauvaise facture : c'est un problème de tests multiples.

```
paiement P, vraie facture A
jour 1 (arrivée)  meilleure proposition A, score 0,990   ← la seule que voit l'offline : sous τ, pas d'auto-validation
jour 4            A a été soldée par un autre paiement
jour 5            meilleure proposition B (fausse), score 0,998 ≥ τ, marge ≥ δ   ← la production auto-valide B : erreur
```

L'offline ne compte pas cette erreur : il **surestime la précision**, donc il choisit un seuil trop permissif.

### 10.4 Online : rejouer la validation comme en production

`online_thresholds`, `pipeline.py:451-467` ; `calibrate_online`, `decision.py:154-190`.
1. On **rejoue la période de validation** avec le vrai pipeline : règles, scoring quotidien, reliquat.
2. On y **désactive l'auto-validation ML**. Chaque paiement reste dans la boucle jusqu'au bout, et on enregistre la
   **trajectoire** de ses propositions : une ligne par jour (type, score, marge, factures, juste ou non). C'est ce qui
   permet d'évaluer après coup **n'importe quel τ**.
3. Pour un τ donné, on simule la production. On pose `s = score` si la marge est ≥ δ, sinon −∞. Le paiement est
   auto-validé le jour où `max(s des jours précédents) < τ ≤ s du jour` (premier franchissement), et l'auto-validation
   est juste si la proposition de ce jour l'est. Seuls les jours qui établissent un nouveau maximum peuvent être des
   jours de franchissement, d'où un `cummax` dans le code.
4. On calcule la précision(τ) par type sur environ 2 000 seuils, et on retient le plus permissif qui tient 99,5 % sur
   au moins 20 auto-validations.

**C'est la méthode utilisée** (`"method": "online"` dans `model.json`). L'offline n'est gardé que pour comparaison.

| | single | set | partial |
|---|---|---|---|
| online (utilisé) | 0,9976 | 0,9986 | jamais |
| offline (pour comparaison) | 0,9978 | 0,9986 | jamais |

Les deux sont proches sur le synthétique, dont les scores bougent peu d'un jour à l'autre ; l'écart devrait être plus
net sur le réel. Le coût est un replay complet de la validation (environ 25 min sur le 2 M).

### 10.5 Le seuil tient-il sur le test ? La calibration par la précision

On rejoue le **test** avec exactement le même protocole, pour mesurer seulement (les seuils restent ceux de la
validation). On compare la précision au même seuil (notebook de séance, § 3.7) :

| Type | Validation | Test | Borne basse 95 % sur le test |
|---|---|---|---|
| `single` | 99,52 % (27 684) | **99,56 %** (24 738) | 99,48 % |
| `set` | 99,52 % (833) | **98,89 %** (719) | 98,00 % |

- **`single` tient** : le test aurait même choisi un seuil voisin (0,9965).
- **`set` ne tient pas.** C'est le **winner's curse** : on prend le seuil le plus permissif qui passe, parmi environ
  2 000, sur peu de propositions, donc on retient souvent celui qui a eu de la chance.

Correctif : choisir τ sur la **borne basse de Clopper-Pearson** plutôt que sur la précision observée. Une borne
≥ 99,5 % à 95 % demande environ 600 propositions toutes correctes. Avec cette règle, `set` ne serait plus auto-validé
(borne basse 98,9 % en validation), pour un coût d'environ −0,2 pt d'automatisation.

### 10.6 Limites

- La même validation sert à l'early stopping, à la calibration et aux seuils (F1). La mesure honnête est sur le test.
- Pendant le replay de calibration, les factures que le ML aurait soldées restent ouvertes pour les jours suivants :
  c'est une interaction de second ordre, ignorée.

---

## 11. La décision

**Ce que ça fait.** Pour chaque proposition (`decide`, `decision.py:136-151`) :
- **auto-validation** si `score ≥ τ_high(type)` et `marge ≥ δ` (0,05) ;
- **revue** si `score ≥ τ_low` (0,3) ;
- **manuel** sinon.

Les auto-validations ML du lot sont ensuite **arbitrées** (`_arbitrate_ml`, `pipeline.py:187-203`) : par score
décroissant, chacune réserve ses factures. Si une facture n'a plus assez de restant dû, parce qu'une règle ou un
meilleur score l'a déjà prise, la proposition redescend en revue.

**Pourquoi.**
- **La marge** protège contre deux candidats presque à égalité. Le score du premier peut être haut alors que le choix
  est ambigu.
- **La revue** donne au gestionnaire une proposition pré-remplie (factures et probabilité calibrée) au lieu d'une page
  blanche.
- **L'arbitrage** garantit qu'une facture n'est jamais soldée deux fois dans le même lot.

**Ce qu'on mesure** (test) :
- `ML_SINGLE` : 24 315 auto-validations, à 99,67 % ;
- `ML_SET` : 703, à 98,86 % ;
- au total : 25 018 auto-validations ML à 99,64 %, avec une borne basse à 95 % de 99,58 %.

**Limites.**
- Les seuils par segment (marché, portefeuille, tranche de montant, présence de client file) existent mais sont
  désactivés (`segmented_thresholds: false`) : le volume par segment doit le permettre.

---

## 12. En production : la boucle quotidienne

**Ce que ça fait.** La boucle de production est celle du replay (`DailyIterator`, `src/timeline/loop.py:83` ;
`PipelineMatcher.process`, `src/reconcile_ml/pipeline.py:145-170`). Chaque jour D :
1. l'état avance à la veille de D : tous les événements antérieurs à D (`LedgerState.advance_to`,
   `src/timeline/state.py:178`) ;
2. le **lot** se compose des paiements arrivés à D et du **reliquat** non résolu des jours précédents. Un paiement quitte
   le reliquat s'il a été imputé, s'il a été auto-validé par le moteur, ou s'il dépasse 60 jours ;
3. on alloue, on applique les règles, puis le ML sur le résiduel : candidats nets des réservations, features, score des
   deux passes, propositions, décision, arbitrage, n↔n ;
4. on obtient des **auto-validations** à écrire comme imputations, une **file de revue** (proposition et probabilité
   calibrée), et le reste en manuel.

Un paiement en reliquat est **rescoré chaque jour**. Il peut donc être auto-validé plus tard, par exemple le jour où son
client file arrive ou celui où un concurrent est soldé.

**Pourquoi un seul code.** Le backtest mesure ce que ferait la production. Deux implémentations, une pour le replay et
une pour la production, finiraient par diverger, et le backtest ne prouverait plus rien.

**Ce qui distingue le replay de la production.**

| | Replay (backtest) | Production |
|---|---|---|
| État | suit le **journal réel** : les décisions du moteur ne modifient pas les soldes ; elles sont comparées aux imputations réelles | les auto-validations deviennent des **imputations** du journal, et l'état du lendemain en tient compte |
| Factures prises par le moteur | réservées en mémoire (`_claimed`) jusqu'à ce que l'imputation réelle apparaisse dans le journal (`_release_realised`, `src/reconcile_rules/matcher.py:80-87`) | à persister d'une exécution à l'autre, ou à reconstruire à partir des décisions pas encore visibles comme imputations |
| Garde-fou | `run_replay` refuse toute décision qui cite une facture inconnue à D (`LeakError`) | identique |

**Ce qui reste à construire pour la production** (`docs/codebase/08_zones_dombre_et_dette.md`, N9) :
- un **point d'entrée quotidien** : aujourd'hui, seul le replay existe ;
- l'**ingestion incrémentale** des événements de la veille (journal en ajout, empreinte) ;
- l'**écriture** des auto-validations et de la file de revue dans le système cible ;
- la **persistance des réservations** entre deux exécutions ;
- le **retrain** à cadence fixe, mensuel par exemple, compatible avec le walk-forward de F4. À chaque retrain :
  - nouveau dataset, nouveau modèle et nouveaux seuils online ;
  - versionnage par `model.json` : empreinte du journal, version de la featurisation, des règles et des réglages,
    périodes, seuils, métriques et vocabulaires ;
  - un `recalibrate_ml` (`pipeline.py:470`) recalcule les seuils sans réentraîner, si seule la cible ou la décision
    change ;
- le **monitoring** :
  - précision des auto-validations par échantillonnage contrôlé par les gestionnaires, avec intervalle de confiance ;
  - taux d'automatisation et volume de revue par mois ;
  - dérive de calibration (`ml_calibration`) et recall des candidats ;
  - un **repli** simple : relever un seuil ou désactiver un type par configuration.

**Ordres de grandeur** (synthétique 2 M) : environ 21 s par jour de replay en mode complet (règles et ML), mémoire
bornée par blocs.

Côté apprentissage :

| Étape | Durée |
|---|---|
| construction du dataset | 29 min |
| entraînement | 6 min |
| calibration online des seuils | 25 min |

---

## 13. L'évaluation

**Ce que ça fait.** Le backtest rejoue la période de **test** (les 2 derniers mois, jamais vus) avec le pipeline
complet et des seuils figés, puis compare chaque décision aux imputations réellement prononcées (`evaluate`,
`src/evaluation/metrics.py:143`).

**Comment.**
- Une auto-validation est **juste si l'ensemble de factures est exactement l'ensemble imputé**, sans crédit partiel.
- Indicateurs :
  - taux d'automatisation et précision, en cascade (actuel, règles, règles + ML) ;
  - détail par règle, par type de groupe réel et par mois ;
  - courbe automatisation / précision ;
  - diagnostics ML (recall des candidats, precision@1, calibration).
- Le notebook de séance ajoute la **borne basse de Clopper-Pearson**, la **calibration par la précision** (§ 10.5) et
  la comparaison validation / test à définition égale.

**Pourquoi.**
- La cible porte sur la précision au **paiement** : c'est l'unité de travail du gestionnaire.
- Le test isolé par le split temporel est la seule mesure que ni le training ni la sélection n'ont vue.

**Ce qu'on mesure.**
- 70,13 % d'automatisation à 99,61 %, borne basse 99,59 %.
- Stable par mois : 70,26 % en novembre, 70,02 % en décembre.
- precision@1 sur le test **95,65 %**, contre 95,66 % en validation. Les deux sont calculées sur les paiements qui ont
  un vrai candidat. Le chiffre de 80,3 % du rapport compte comme un échec les paiements sans vrai candidat.

**Limites.**
- La courbe automatisation / précision choisit son point sur le test : c'est un **plafond a posteriori** (74,2 %),
  pas un résultat (F6).
- Un test regardé à chaque itération de développement devient peu à peu un jeu de validation. Avant une conclusion
  définitive, figer la configuration et évaluer sur une période jamais vue.

---

## 14. Récapitulatif

| Étape | Ce qu'elle fait | Pourquoi | Chiffre clé (synthétique 2 M) | Limite principale |
|---|---|---|---|---|
| 1. Résiduel | le ML ne traite que ce que les règles n'ont pas résolu | règles auditables et quasi parfaites sur les cas simples | règles 62,75 %, ML +7,4 pts | R4, R5 sous la cible |
| 2. Candidats | ≤ 50 factures par paiement, 4 sources | impossible et inutile de tout scorer | recall 79,8 % | le plafond du ML, premier levier |
| 3. Features | 54 features relatives, datées | mesurer la paire et le contexte du débiteur | aucune feature seule ne domine | attributs statiques non datés (F2) |
| 4. Dataset | replay train + validation, labels à part | mêmes features qu'en production, pas de fuite de vérité | 24,9 M de paires | labels sans horizon (F3), jour d'arrivée seulement |
| 5. Pass 1 | LightGBM binaire par paire | tabulaire, non linéaire, score absolu | precision@1 95,7 % | ignore la concurrence |
| 6. Folds OOF | compétition sans label du paiement | stacking sans fuite | OOF ≈ validation | folds non temporels (F4) |
| 7. Pass 2 | score avec le contexte du paiement | la confiance dépend des concurrents | log-loss −8 % (40 k) | apport à chiffrer en automatisation |
| 8. Isotonique | score → probabilité lisible | lecture, suivi | hors échantillon : pas de gain | ne sert pas à décider |
| 9. Propositions | single / partial / set, marge | on impute des ensembles | `set` à 98,9 % sur le test | score d'ensemble = moyenne |
| 10. Seuils online | τ par type, au premier franchissement | la production rescore chaque jour | `single` tient sur le test | winner's curse sur `set` |
| 11. Décision | auto / revue / manuel, arbitrage | refuser de décider dès qu'il y a doute | ML 99,64 % (borne 99,58 %) | segments désactivés |
| 12. Production | même boucle, chaque jour | le backtest prouve la production | ≈ 21 s par jour | point d'entrée, persistance, retrain, monitoring à construire |
| 13. Évaluation | test jamais vu, ensemble exact | seule mesure non sélectionnée | 70,13 % à 99,61 % | plafond oracle à ne pas citer |
