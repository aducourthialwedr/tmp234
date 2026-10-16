**Objet : Étape ML : ce que les folds OOF ne leakent pas, le leakage réel, et comment le corriger**

Bonjour [Prénom],

Suite à notre échange sur les folds du stacking, voici notre analyse. Pour chaque point, nous donnons d'abord le
**raisonnement théorique** (ce que la construction garantit ou non), puis la **vérification empirique**. Celle-ci
vient du notebook `notebooks/ml_pas_a_pas.ipynb`, qui refait `PairModel.fit` à la main, vérifie que le résultat est
identique à celui du code (même `best_iteration`, mêmes scores, mêmes probabilités), puis ajoute des tests de
leakage causaux.

En bref :

1. **Les folds OOF ne leakent pas le label.** Par construction, aucune paire n'est scorée par un modèle qui a vu un
   label de son paiement, et aucun modèle de fold ne voit le validation set. Le notebook le confirme.
2. **Il y a pourtant du leakage, mais ailleurs**, et la plupart du temps autour des folds plutôt que dans les
   folds :
   - le CV n'est pas temporel : le futur du train set entre dans les features de compétition (F4) ;
   - un canal étroit va du validation set vers le train set, par le nombre d'itérations des modèles de fold ;
   - le validation set sert à cinq choix successifs, puis à mesurer leur qualité (F1) ;
   - les labels sont lus sans horizon (F3) ;
   - des attributs statiques ne sont pas datés (F2). C'est probablement le leakage le plus coûteux sur les données
     réelles.
3. Tout cela a un correctif simple, sauf F2, qui dépend de l'historique disponible côté données. Vous les trouverez
   au § 4.

**Précaution sur les chiffres.** Les mesures viennent du dataset synthétique (40 000 paiements, seed fixe),
conformément à notre règle de ne pas développer sur les données réelles. Le validation set y compte 1 880 paiements
avec au moins un positif, d'où une standard error d'environ **±0,5 pt** sur la precision@1. Le générateur n'a pas
de drift et ses délais d'imputation sont courts : le leakage temporel y est **sous-estimé**. Nous rejouons le même
notebook sur les données réelles (`DATASET = "real"`) ; c'est ce run qui tranchera.

---

## 1. Rappel du dispositif

### 1.1 Split temporel

Le split se fait par jours, jamais au hasard, avec une purge de 5 jours entre les blocs (`config/settings.yaml`,
section `split`). Sur le synthétique, bornes incluses :

| Période | Du | Au | Jours | Rôle |
|---|---|---|---|---|
| train | 2023-11-09 | 2024-08-23 | 289 | training des pass 1 et 2, folds OOF |
| validation | 2024-08-29 | 2024-10-28 | 61 | early stopping, calibration, seuils |
| test | 2024-11-03 | 2025-01-02 | 61 | backtest complet, seuils figés, jamais vu |

Le test set n'intervient ni dans le training ni dans le choix des seuils.

### 1.2 Construction du dataset de paires

Le training set est produit en **rejouant la boucle quotidienne de production** sur le train et la validation
(`DatasetRecorder`, `src/reconcile_ml/pipeline.py`) :

```
jour D ─► état figé à la veille (LedgerState, as_of = D)
        ─► allocation → règles déterministes → résiduel
        ─► pour chaque paiement du résiduel arrivé le jour D :
              candidats = factures ouvertes à D (débiteur, référence, montant, client file)
              54 features par paire, calculées sur l'état à D
```

- **Un paiement n'est enregistré qu'une fois**, le jour de son arrivée (filtre `is_new`, `pipeline.py:107`), même
  s'il reste plusieurs jours en reliquat. Le groupe LightGBM est donc le paiement (`ds[GROUP] = ds["payment_id"]`,
  `pipeline.py:393`).
- **Le labelling se fait dans une seconde passe séparée** (`label_pairs`, `pipeline.py:289-294`) : label 1 si le
  couple (paiement, facture) figure dans `imputation`, 0 sinon.
- Ordres de grandeur sur le synthétique :
  - 173 706 paires en train et 59 447 en validation ;
  - 19 à 24 candidats par paiement (médiane) ;
  - 4,5 % de positifs ; 19 % de paiements 1↔n ;
  - recall des candidats de 91 à 92 %.

C'est un problème de **ranking par groupe**, entraîné en pointwise (objectif `binary`). On suit donc precision@1 et
MRR par paiement, en plus de l'AUC.

### 1.3 Le stacking en deux passes

Le schéma suit `src/reconcile_ml/model.py:111-134` :

```
pass 1      : LightGBM, 54 features, entraîné sur tout le train set
              early stopping sur le validation set (patience 30, model.py:76-77) → best_iteration (195 ici)

folds OOF   : fold = hash(payment_id) mod 3, soit un GroupKFold par paiement, non temporel  (model.py:119)
              3 modèles de fold, chacun entraîné sur 2/3 du train set,
              195 itérations FIXES, sans validation set                                   (model.py:121-125)

compétition : rank_in_payment, score_margin (écart au meilleur concurrent), score_best_other
              - train      : à partir des scores OOF
              - validation : à partir de la pass 1 complète, comme en production          (model.py:127-128)

pass 2      : LightGBM, 57 features (54 + 3), entraîné sur le train set
              early stopping sur le validation set                                        (model.py:129)
calibration : régression isotonique, fit sur les scores bruts de pass 2 du validation set  (model.py:131-133)
seuils      : τ_high par type (single, partial, set), choisis sur le validation set par un replay en ligne
test        : backtest complet, seuils figés
```

**Notations.**

| Symbole | Sens |
|---|---|
| `p` | un paiement |
| `C(p)` | ses candidats |
| `x_pj`, `y_pj` | les features et le label de la paire (p, j) |
| `k(p)` | le fold de p |
| `A` | l'algorithme de training (LightGBM à nombre d'itérations fixé) |
| `D` | le train set |
| `D₋ₖ` | le train set privé du fold k |
| `f₋ₖ = A(D₋ₖ)` | le modèle de fold k |
| `g` | la fonction qui calcule rang, marge et meilleur concurrent à partir des scores d'un paiement |

---

## 2. Pourquoi les folds ne leakent pas le label

### 2.1 Pas de stacking leakage

**Théorie.**
- Le score OOF d'une paire du train vaut `s(p, j) = f₋ₖ₍ₚ₎(x_pj)`, et ses features de compétition valent
  `c_pj = g( s(p, ·) )_j`.
- Le fold est tiré par paiement : toutes les paires de p sont dans le fold k(p). Le training set `D₋ₖ₍ₚ₎` de
  `f₋ₖ₍ₚ₎` ne contient donc **aucun** label `y_p·`.
- `c_pj` est ainsi une fonction de `x_p·` et de `D₋ₖ₍ₚ₎` seulement. Elle n'a aucun accès, direct ou indirect, aux
  labels du paiement qu'elle décrit. C'est exactement la condition d'un stacking sans target leakage.
- À l'inverse, en in-sample, `f(x_pj)` a été fit sur `y_pj` lui-même, et les marges seraient artificiellement
  nettes.

Deux conditions d'implémentation sont nécessaires, et elles sont remplies :
1. **L'unité du fold doit coïncider avec l'unité d'agrégation de `g`**, c'est-à-dire le paiement. Sinon, le
   « meilleur concurrent » d'une paire pourrait être scoré par un modèle qui a vu le label de ce concurrent, qui est
   aussi un label du même paiement.
2. **Aucun paiement ne doit apparaître sous deux groupes.** C'est garanti par l'enregistrement unique (`is_new`) :
   un paiement en reliquat n'est pas réenregistré les jours suivants.

Une troisième condition porte sur le nombre d'itérations, fixé de l'extérieur : il ne doit pas dépendre de `y_p·`.
C'est le cas, puisqu'il dépend du validation set. Mais cela ouvre un autre canal, traité au § 3.2.

**Ce que la théorie ne garantit pas.** Elle garantit l'absence de leakage du label **de la paire**. Elle ne garantit
pas que `s(p, ·)` suive la même distribution que le score de production. Pour cela, il faudrait que les paiements
soient échangeables dans le temps, ce qui est faux. C'est l'objet du § 3.1.

**Vérification empirique** (notebook, § 3 et § 4). Si les scores OOF contenaient le label, ils ressembleraient aux
scores in-sample. Ils ressemblent en fait à ceux de la validation :

| Scoring (pass 1) | AUC | precision@1 | MRR | log-loss |
|---|---|---|---|---|
| train, in-sample | 0,9997 | 0,991 | 0,995 | 0,0128 |
| train, OOF | 0,9964 | 0,945 | 0,966 | 0,0261 |
| validation (pass 1 complète) | 0,9970 | 0,949 | 0,968 | 0,0224 |

Pris un par un, chaque modèle de fold obtient 0,997 en in-sample, contre 0,942 à 0,949 sur le fold qu'il score, et
0,947 à 0,948 sur le validation set. Les 4,6 points d'optimisme in-sample (0,991 contre 0,945) disparaissent.

Les features de compétition ont des distributions très proches entre le train et la validation :

| Feature | Moyenne train (OOF) | Moyenne validation |
|---|---|---|
| `rank_in_payment` | 19,2 | 19,7 |
| `score_margin` | −0,747 | −0,728 |
| `score_best_other` | 0,791 | 0,768 |

Les médianes sont quasi identiques.

**Attention.** Cette proximité ne suffit pas à exclure **tout** leakage, car deux effets jouent en sens opposés :
- les modèles de fold ne voient que 2/3 du train set, ce qui rend les scores OOF plus pessimistes ;
- ils voient le futur (§ 3.1), ce qui les rend plus optimistes.

Elle confirme seulement l'absence de leakage du label de la paire.

### 2.2 Pas d'overlap entre le train et la validation

**Théorie.**
- Les périodes sont disjointes par jour, séparées par une purge.
- Les modèles de fold sont fit sur des sous-ensembles de `D`, sans validation set. Ils n'ont pas d'early stopping.
- Côté validation, les features de compétition viennent de la pass 1 entraînée sur `D` seul.

Aucune ligne de validation n'entre donc dans un training des folds.

**Vérification empirique** (notebook, § 4.3, tests T1 et T3) :

| Test | Mesure |
|---|---|
| Paiements à la fois dans le train set et le validation set | 0 |
| Gap entre le dernier jour du train et le premier jour de la validation | 6 jours (purge configurée : 5) |
| Lignes vues à l'entraînement par les 3 modèles de fold | 347 412 = 2 × 173 706, toutes du train set |
| Paires positives du train labellisées par une imputation datée de la validation ou après | 0 |

### 2.3 Pas d'information future dans les features (hors attributs non datés)

**Théorie.**
- Toute lecture d'état passe par `LedgerState` avec un `as_of`. L'état n'avance que vers l'avant, en n'appliquant
  que les événements antérieurs à `as_of`, et une lecture à une autre date que celle de l'état lève une
  `TemporalError`.
- `invoice.current_amount`, qui donnerait le solde final, est retiré au chargement. Le restant dû est reconstruit
  par les imputations antérieures à D.
- Les agrégats comportementaux sont calculés sur la fenêtre `[D − 180 j, D)`, strictement antérieure.
- Les client files ne sont visibles qu'après leur `received_at`.

Une feature calculée à D est donc une fonction du journal avant D, et elle est invariante si l'on retire les
événements postérieurs.

**Vérification empirique** (notebook, § 8).
- **Truncation test.** On supprime du journal tout ce qui date du 2024-10-29 ou après, c'est-à-dire tout ce qui
  suit la fin de la validation, puis on reconstruit l'intégralité du training set. On obtient les mêmes 238 846
  paires, avec **les mêmes valeurs sur toutes les features**. Seuls les labels diffèrent (§ 3.4).
- **Audit feature par feature.** On a entraîné un modèle par feature, seule. La meilleure est
  `amount_rank_in_payment`, avec une precision@1 de 0,585, contre 0,951 pour le modèle complet. Aucune feature n'est
  un proxy du label.

Le truncation test ne couvre pas les tables sans date (§ 3.5).

**Un test non concluant.** Nous avons aussi lancé une adversarial validation sur les features de compétition. Le
contrôle positif in-sample, qui leake franchement, n'est pas détecté (AUC 0,510). Le test n'est donc pas assez
sensible, et nous ne nous appuyons pas dessus.

---

## 3. Le leakage réel

### 3.1 F4 : le CV des folds n'est pas temporel

**Théorie.** Un GroupKFold suppose les groupes **échangeables**, c'est-à-dire indépendants et identiquement
distribués. Ce n'est pas le cas ici, pour deux raisons :
- **Dépendance par débiteur.** Les paiements d'un même débiteur partagent des régularités. Les features `debtor_*`
  (délai moyen, taux de partiels, taux de regroupement…), `bankroll_code`, `channel`, `market` et les signaux
  d'allocation forment une quasi-empreinte du débiteur.
- **Non-stationnarité.** La relation entre features et label dérive dans le temps : nouveaux débiteurs, nouveaux
  libellés, changements de pratique.

Avec `hash(payment_id) mod 3`, un paiement de janvier est scoré par `f₋ₖ₍ₚ₎`, entraîné sur des paiements de février
à août, dont ceux du même débiteur. L'**ensemble d'information** du score OOF est donc strictement plus large que
celui dont dispose la pass 1 en production au moment du paiement, à savoir le passé seul.

Ce n'est pas un target leakage sur `y_p·`, mais un **look-ahead leakage** : il porte sur la relation features →
label du futur.

**Conséquences attendues.**
- Les scores OOF du train sont plus optimistes que les scores de production, à volume égal.
- La pass 2 apprend `P(y | marge, rang)` sous la mauvaise distribution. Elle surpondère `score_margin` et
  `rank_in_payment`.
- **Portée :** le leakage est confiné aux features de **training** de la pass 2. En validation, en test et en
  production, les features de compétition viennent de la pass 1 entraînée sur le train set et appliquée à des
  paiements postérieurs. F4 **ne gonfle donc pas** les métriques de test. Il dégrade le modèle, et donc le taux
  d'automatisation atteignable.

**Vérification empirique** (notebook, § 4.2 et § 6.1). On reproduit le scoring « comme en production » par un
**walk-forward mensuel** en expanding window. Pour scorer le mois k, un modèle est entraîné uniquement sur :
- les paires des paiements arrivés avant `Dₖ − purge` ;
- avec les labels connus à `Dₖ` ;
- en écartant les paiements dont l'issue est encore inconnue à `Dₖ`, plutôt que de les labelliser négatifs ;
- au même nombre d'itérations que les modèles de fold, pour comparer à armes égales.

Le premier mois n'a pas d'historique : il est exclu de la comparaison.

| Scoring des paires du train (mêmes paires, hors premier mois) | AUC | precision@1 | log-loss |
|---|---|---|---|
| in-sample | 0,9997 | 0,990 | 0,0127 |
| GroupKFold aléatoire, OOF (code actuel) | 0,9964 | 0,945 | 0,0258 |
| walk-forward (sans futur) | 0,9920 | 0,931 | 0,0353 |
| *validation, pour référence* | *0,9970* | *0,949* | *0,0224* |

L'écart de 1,4 pt entre les deux mélange le futur et le **volume** : les premiers modèles walk-forward sont
entraînés sur très peu de paires (1 292 pour le modèle qui score janvier, 9 167 pour février). On ne peut pas
l'attribuer au seul futur.

**Le test décisif est en aval** : on entraîne la pass 2 sur chacune des deux sources de scores, à paires
identiques, puis on l'évalue sur le validation set.

| Pass 2, évaluée sur le validation set | precision@1 | MRR | log-loss |
|---|---|---|---|
| pass 1 seule, pour référence | 0,949 | 0,968 | 0,0224 |
| GroupKFold aléatoire, tout le train set (code actuel) | 0,951 | 0,969 | 0,0207 |
| GroupKFold aléatoire, mêmes paires que le walk-forward | 0,948 | 0,968 | 0,0207 |
| walk-forward | 0,954 | 0,971 | 0,0201 |

Le walk-forward fait **au moins aussi bien** : +0,6 pt de precision@1, soit environ 1,2 standard error, avec une
meilleure log-loss. Le mécanisme prédit par la théorie est réel, mais son coût n'est pas détectable sur le
synthétique, qui n'a ni drift ni dépendance forte par débiteur. La correction ne coûte rien en performance. Sur les
données réelles, nous attendons un écart plus net.

**Remarque hors leakage.** Les modèles de fold voient 2/3 du train set, au même nombre d'itérations que la pass 1
complète qui score la validation. Les échelles de score diffèrent donc légèrement entre le training et l'inférence de
la pass 2. C'est un distribution shift, que le walk-forward réduit aussi : son dernier modèle est proche du modèle
de production.

### 3.2 Un canal validation → train : le nombre d'itérations des modèles de fold

**Théorie.**
- Les modèles de fold ne voient aucune ligne de validation (§ 2.2). Mais `A` est paramétré par
  `n_iter = max(pass1.best_iteration, 10)` (`model.py:121`).
- `best_iteration` est l'argmin de la loss de validation sur le chemin d'early stopping. C'est donc une fonction des
  labels de validation.
- Par composition, `s(p, j) = A_{n_iter(V)}(D₋ₖ₍ₚ₎)(x_pj)` dépend du validation set.

Le canal ne fait passer qu'un entier, mais il existe. Et ce même validation set sert ensuite à l'early stopping de
la pass 2 et à la calibration.

**Vérification empirique** (notebook, § 4.3, test T4) : deux tests causaux.

| Variante | Itérations des modèles de fold | Écart moyen des scores OOF | precision@1 OOF |
|---|---|---|---|
| référence | 195 | — | 0,945 |
| contrôle : mêmes itérations, folds refaits | 195 | 0,0000 (écart max 0) | 0,945 |
| labels du validation set shufflés | 10 | 0,051 (écart max 0,91) | 0,915 |

- **Le contrôle** montre que `n_iter` est le **seul** canal : le training est déterministe, et à itérations égales
  les scores OOF sont identiques au bit près.
- **Le shuffle des labels** montre que ce canal est actif : quand le validation set ne porte plus aucune
  information, l'early stopping s'arrête à 10 itérations et les features de compétition du train changent.

### 3.3 F1 : le validation set sert à choisir, puis à mesurer

**Théorie.** Le même validation set sert, dans l'ordre :
1. à l'early stopping de la pass 1 ;
2. au nombre d'itérations des modèles de fold (§ 3.2) ;
3. à l'early stopping de la pass 2 ;
4. au fit de la régression isotonique ;
5. au choix des seuils τ_high par type de proposition.

On reporte ensuite, sur ces mêmes données, l'AUC, la precision@1, la reliability curve et la « précision des
auto-validations ». Une métrique mesurée sur les données qui ont servi à la sélection est biaisée vers le haut : la
dernière vaut même au moins 99,5 % **par construction**.

**Le point le plus sensible est τ_high**, car c'est un **winner's curse**.
- `calibrate_online` retient le seuil **le plus permissif** dont la precision observée atteint la cible, sur au
  moins `MIN_KIND_VOLUME = 20` propositions (`src/reconcile_ml/decision.py:27`).
- On prend ainsi le maximum, sur environ 2 000 seuils, d'une quantité bruitée : `E[precision réelle | seuil choisi]`
  est inférieure à la precision observée.
- À 99,5 %, une tranche de 20 à 199 propositions doit être correcte à 100 %. Une tranche de 20 propositions
  dont la vraie precision n'est que de 98 % passe ce test avec une probabilité de 0,98²⁰ ≈ 67 %.
- La precision attendue sur le test set est donc inférieure à la cible, surtout pour les types à faible volume
  (`partial`, `set`).

**Vérification empirique.**
- **Calibration** (notebook, § 6.2). Nous avons coupé le validation set en deux : fit sur V1, évaluation sur V2.

  | Scores | log-loss sur V2 |
  |---|---|
  | bruts | 0,02081 |
  | isotoniques, fit sur V1 | 0,02115 |

  - **Out-of-sample, l'isotonique dégrade la log-loss** : LightGBM, entraîné en log-loss, est déjà bien calibré.
    La reliability curve in-sample, presque parfaite, ne prouvait donc rien.
  - L'isotonique écrase aussi les 5 % de scores les plus hauts en **17 valeurs distinctes**, contre 2 944 en brut.
    C'est pourquoi la décision travaille déjà sur le score brut.
- **τ_high.** Le notebook ne mesure pas ce biais : il s'arrête avant la décision. La mesure se fera sur le test, en
  comparant la precision obtenue par type aux seuils actuels et aux seuils corrigés (§ 4, point 4).

### 3.4 F3 : des labels lus sans horizon

**Théorie.**
- `label_pairs` calcule `y_pj = 1{(p, j) ∈ imputation}` sur **toute** la table, sans condition sur `updated_at`.
- Le label d'un paiement de fin de période utilise donc de l'information postérieure à la fin de la période :
  pendant la purge, pendant le test, ou après.
- La purge ne joue pas son rôle : elle dure 5 jours, alors qu'un paiement reste en reliquat jusqu'à 60 jours et
  que l'imputation manuelle prend souvent plusieurs semaines.
- **Biais de sélection.** À la date T d'un retrain en production, l'issue des paiements récents est inconnue. Ici,
  les paiements difficiles, qui sont imputés tard, sont labellisés **grâce** au futur. Le training set est donc plus
  informé que celui d'un retrain réel, précisément sur les cas les plus difficiles.

**Vérification empirique** (notebook, § 8.1, même truncation test qu'au § 2.3).
- Les paires et les features sont identiques, mais **56 labels basculent**. Tous concernent des paiements des
  derniers jours de la validation, imputés après sa fin.
- Le modèle entraîné avec l'après-validation atteint 0,951 de precision@1, contre 0,948 sans : l'écart est dans le
  bruit.
- Sur le synthétique, les délais d'imputation ne dépassent pas 3 jours (P95 : 3 jours). Ces labels viennent donc de
  la purge et non du test, et c'est aussi pourquoi aucun label du train set ne vient de la validation (§ 2.2).

Sur les données réelles, le chiffre à regarder est le **P95 du délai entre l'arrivée du paiement et son
imputation** : s'il dépasse la purge, F3 devient significatif.

### 3.5 F2 : des attributs statiques non datés, probablement le leakage le plus coûteux en réel

**Théorie.**
- Plusieurs informations n'ont pas de date et sont visibles dès le premier jour du replay, dans l'état où elles ont
  été **extraites**, c'est-à-dire en fin d'historique :
  - tous les IBAN d'une partie (`party_iban`) ;
  - les variantes de nom ;
  - le référentiel des comptes techniques ;
  - les champs de facture (`client_reference`, `debtor_id`, `due_date`, `agreement_id`) ;
  - les attributs de contrat.
- En affacturage, le référentiel se complète souvent **à partir des paiements eux-mêmes**. Un gestionnaire rapproche
  à la main un virement venu d'un IBAN inconnu, puis rattache cet IBAN au débiteur. Dans le replay, l'IBAN est alors
  connu **avant** le paiement qui l'a fait découvrir.
- Pour ce paiement, `alloc_sig_iban` ou `iban_route = DEBTOR_DIRECT` est en partie un **label déguisé** : un
  target leakage au sens strict. En production, à l'inverse, les IBAN encore inconnus sont précisément ceux des
  paiements difficiles.

**Vérification empirique : pas encore possible.**
- Le truncation test ne peut pas voir ce leakage : sans date, ces tables ne sont pas tronquées (`party_iban` garde
  ses 938 lignes).
- Le synthétique ne le simule pas : son référentiel est complet dès le départ.

Il faut une mesure dédiée sur les données réelles (§ 4, point 5).

### 3.6 Leakage secondaire

- **F5 : imputation datée avant l'arrivée de son paiement.**
  - Le journal place l'imputation à `updated_at` et le paiement à sa date de valeur (ou de comptabilisation).
  - Si la première précède la seconde, l'état figé au jour d'arrivée contient déjà l'imputation : restant dû
    réduit, agrégats du débiteur qui l'incluent.
  - Le rapport de chargement compte ces cas (`avant_connaissance_du_paiement`) mais ne les corrige pas.
- **F6 : le KPI affiché en tête choisit son seuil sur le test set.**
  - `taux_automatisation_à_précision_cible` est le point de la precision curve du test le plus à droite au-dessus
    de la cible : c'est un seuil oracle, donc un plafond a posteriori.
  - Le résultat légitime est `taux_automatisation`, obtenu avec les seuils de validation.
  - C'est une question de reporting ; le modèle n'est pas en cause.
- **F7 : vocabulaires de catégories construits sur toutes les données.** Cela concerne `channel`, `bankroll_code`,
  `market`, `product` et `recourse`. L'effet théorique est faible, et le truncation test ne montre aucun
  changement d'encodage.

---

## 4. Correctifs proposés, par ordre de priorité

1. **Labelliser avec un horizon (F3).** C'est le prérequis des autres correctifs.
   - Labels à la date de cutoff C : seules les imputations avec `updated_at < C` comptent.
   - Un paiement sans issue connue à C et plus récent que `retention_days` est **écarté**, et non labellisé négatif.
     Le labelliser négatif remplacerait un leakage par du label noise.
   - Purge ≥ P95 du délai arrivée → imputation, mesuré sur le train set.

2. **Remplacer le GroupKFold aléatoire par un walk-forward (F4).**
   - Retrain aux dates D₀ < D₁ < …, à la cadence prévue en production (mensuelle), en expanding window.
   - Le modèle entraîné à Dₖ voit les paiements arrivés avant `Dₖ − purge`, avec les labels connus à Dₖ (point 1),
     et score `[Dₖ, Dₖ₊₁)`. Le score OOF a alors le même ensemble d'information qu'en production.
   - **Aucun nouveau replay n'est nécessaire.** Les features de pass 1 ne dépendent pas du modèle, et
     `DatasetRecorder` ne décide que par les règles. Le dataset enregistré, avec sa colonne `day`, contient donc déjà
     tout ce que la boucle aurait vu.
   - Coût, d'après le notebook : la somme des tailles de training est d'environ 451 000 paires, contre 347 000 pour
     les 3 modèles de fold actuels, soit environ 1,3 fois le coût actuel du CV.
   - Le premier mois, qui n'a pas de score, est exclu de la pass 2, ou traité comme un rang manquant.
   - Fallback, moins juste : un GroupKFold par **débiteur**. Il coupe la dépendance par débiteur, mais ne traite ni le
     drift ni l'horizon des labels.

3. **Fixer le nombre d'itérations sans le validation set (§ 3.2).**
   - Early stopping interne au train set, sur son dernier mois après purge (inner holdout temporel) ;
   - ou nombre d'itérations choisi à l'intérieur du walk-forward.
   - Le validation set ne modifie alors plus aucune feature du train set.

4. **Séparer les rôles du validation set (F1).**
   - **V1** pour l'early stopping des deux passes, **V2** pour les seuils.
   - Ne reporter comme « out-of-sample » que des métriques calculées sur une période qui n'a servi à aucun choix.
   - Choisir τ_high sur une **lower bound** de la precision plutôt que sur la precision observée. Avec Clopper-Pearson
     à 95 %, une lower bound ≥ 99,5 % impose environ 600 propositions toutes correctes
     (`n ≥ ln 0,05 / ln 0,995 ≈ 598`). Relever `MIN_KIND_VOLUME` en conséquence.
   - Retirer l'isotonique du chemin de décision, ce qui est déjà le cas, ou la remplacer par du Platt scaling pour
     l'affichage.
   - Si V1 et V2 font un mois chacun, vérifier le volume par type de proposition ; au besoin, étendre la validation à
     trois mois.

5. **Dater les attributs statiques (F2).**
   - Demander à l'équipe données si `party_iban`, les noms et les champs de facture ont un historique (date d'ajout ou
     de modification). Si oui, en faire des événements du journal (`PARTY_IBAN_ADDED`…).
   - Sinon, estimer la date d'apparition d'un IBAN par le premier paiement imputé au débiteur depuis cet IBAN, et ne
     le rendre visible qu'à partir de ce jour.
   - **Mesure d'impact** : rejouer le test sans le signal IBAN pour les couples (débiteur, IBAN) dont le premier
     paiement imputé est postérieur à D, puis comparer le taux d'automatisation et la precision.

6. **Leakage secondaire.**
   - F5 : rendre le contrôle bloquant au-delà d'un seuil, ou décaler l'imputation à
     `max(updated_at, arrivée du paiement + ε)`.
   - F6 : afficher en tête le taux obtenu avec les seuils de validation, et intituler la courbe « plafond a
     posteriori (seuil oracle) ».
   - F7 : construire les vocabulaires sur le seul train set.

7. **Automatiser le contrôle.** Ajouter un test anti-leakage du **training**, et pas seulement de l'inférence :
   - construire le training set sur le journal complet et sur le journal tronqué à la fin de la validation, en
     tronquant aussi les attributs une fois datés ;
   - les features doivent être identiques, et les labels aussi une fois l'horizon appliqué.

   Le test anti-leakage actuel rejoue un modèle déjà entraîné sur un journal tronqué : il ne couvre ni le training,
   ni les tables non datées.

Les points 1 à 4 sont locaux à `model.py`, `pipeline.py` et `decision.py`. Nous proposons de les implémenter en
premier, puis de comparer, **sur le test set**, le taux d'automatisation à precision fixée avant et après
correction. Le point 5 dépend de la réponse de l'équipe données.

## 5. Mesures en cours sur les données réelles

Nous rejouons le même notebook sur les données réelles, en suivant en priorité :
1. le P95 du délai entre l'arrivée du paiement et son imputation, et le nombre de labels qui basculent au truncation
   test (F3) ;
2. l'écart entre la pass 2 « GroupKFold aléatoire » et la pass 2 « walk-forward » (F4) ;
3. le test sans signal IBAN pour les IBAN découverts après coup (F2) ;
4. le compte `avant_connaissance_du_paiement` (F5).

Je vous envoie les résultats dès qu'ils sont disponibles. Si vous voulez examiner un point en détail, le notebook
déroule chaque étape avec ses sorties intermédiaires, et `docs/fuites_ml.md` détaille chaque leakage avec les
références de code.

Bien à vous,

[Prénom Nom]
