**Objet : Leakage dans l'étape ML : pas de leakage grossier, un impact limité, et des correctifs pour le résiduel**

Bonjour [Prénom],

Merci pour votre retour sur l'étape ML et pour la question du leakage. L'intuition était bonne : un stacking en deux
passes avec des folds OOF, sur des données temporelles, est exactement le genre de montage où le leakage se glisse.
Nous avons donc instrumenté le training étape par étape. Le notebook `notebooks/ml_pas_a_pas.ipynb` refait
`PairModel.fit` à la main, vérifie que le résultat est identique à celui du code (même `best_iteration`, mêmes scores,
mêmes probabilités), puis ajoute des tests de leakage causaux.

Notre conclusion tient en trois points :

1. **Il n'y a pas de leakage grossier.** En particulier, les folds OOF ne leakent pas le label des paires, aucun
   modèle ne voit le validation set pendant son training, et aucune feature ne contient d'information future. Le
   raisonnement et les mesures sont au § 1.4.
2. **L'impact du leakage restant est faible**, pour une raison structurelle : toutes nos évaluations finales se font
   sur le **test set**, que le split temporel principal isole complètement. Il ne sert ni au training, ni à l'early
   stopping, ni à la calibration, ni au choix des seuils. Un leakage d'apprentissage ou de sélection peut dégrader le
   modèle ou rendre les métriques de validation optimistes, mais il ne peut pas gonfler la mesure sur le test (§ 1.5).
3. **Il existe en revanche des leakages résiduels**, décrits au § 2.
   - La plupart sont des leakages d'apprentissage ou de sélection, que la barrière du test neutralise.
   - Deux d'entre eux, F2 et F5, sont des leakages **de données** : ils touchent aussi le replay du test, que le split
     ne protège donc pas. Ce sont les premiers à mesurer sur les données réelles.
   - Nos propositions d'évolution sont au § 3.

**Précaution sur les chiffres.** Sauf mention contraire, les mesures viennent du dataset synthétique (40 000
paiements, seed fixe), conformément à notre règle de ne pas développer sur les données réelles. Le validation set y
compte 1 880 paiements avec au moins un positif, d'où une standard error d'environ **±0,5 pt** sur la precision@1. Le
générateur n'a pas de drift et ses délais d'imputation sont courts : le leakage temporel y est **sous-estimé**. Nous
rejouons le même notebook sur les données réelles (`DATASET = "real"`) pour confirmer.

---

## 1. La pipeline actuelle, et pourquoi le leakage y pèse peu

### 1.1 Split temporel

Le split se fait par jours, jamais au hasard, avec une purge de 5 jours entre les blocs (`config/settings.yaml`,
section `split`). Sur le synthétique, bornes incluses :

| Période | Du | Au | Jours | Rôle |
|---|---|---|---|---|
| train | 2023-11-09 | 2024-08-23 | 289 | training des pass 1 et 2, folds OOF |
| validation | 2024-08-29 | 2024-10-28 | 61 | early stopping, calibration, seuils |
| test | 2024-11-03 | 2025-01-02 | 61 | backtest complet, seuils figés, jamais vu |

Le replay d'apprentissage s'arrête à la fin de la validation. Le test set n'est lu qu'une fois, par le backtest final,
avec un modèle et des seuils déjà figés.

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

### 1.4 Pas de leakage grossier

**a) Les folds OOF ne leakent pas le label.**

*Théorie.*
- Le score OOF d'une paire du train vaut `s(p, j) = f₋ₖ₍ₚ₎(x_pj)`, et ses features de compétition valent
  `c_pj = g( s(p, ·) )_j`.
- Le fold est tiré par paiement : le training set `D₋ₖ₍ₚ₎` ne contient **aucun** label `y_p·`.
- `c_pj` est donc une fonction de `x_p·` et de `D₋ₖ₍ₚ₎` seulement, sans accès, direct ou indirect, aux labels du
  paiement qu'elle décrit. C'est exactement la condition d'un stacking sans target leakage.

Deux conditions d'implémentation sont nécessaires, et elles sont remplies :
- l'unité du fold coïncide avec l'unité d'agrégation de `g`, le paiement. Sinon, le « meilleur concurrent » d'une paire
  pourrait être scoré par un modèle qui a vu son label ;
- aucun paiement n'apparaît sous deux groupes, grâce à l'enregistrement unique (`is_new`).

*Vérification.* Si les scores OOF contenaient le label, ils ressembleraient aux scores in-sample. Ils ressemblent en
fait à ceux de la validation (notebook, § 3 et § 4) :

| Scoring (pass 1) | AUC | precision@1 | MRR | log-loss |
|---|---|---|---|---|
| train, in-sample | 0,9997 | 0,991 | 0,995 | 0,0128 |
| train, OOF | 0,9964 | 0,945 | 0,966 | 0,0261 |
| validation (pass 1 complète) | 0,9970 | 0,949 | 0,968 | 0,0224 |

- Chaque modèle de fold obtient 0,997 en in-sample, contre 0,942 à 0,949 sur le fold qu'il score : les 4,6 points
  d'optimisme in-sample disparaissent.
- Les features de compétition ont des distributions proches entre le train et la validation : `rank_in_payment`
  19,2 contre 19,7 en moyenne, `score_margin` −0,747 contre −0,728, médianes quasi identiques.

**b) Aucun modèle ne voit le validation set pendant son training.**

*Théorie.* Les périodes sont disjointes par jour. Les modèles de fold sont fit sur des sous-ensembles du train set,
sans early stopping. Côté validation, les features de compétition viennent de la pass 1 entraînée sur le train set
seul.

*Vérification* (notebook, § 4.3, tests T1 et T3) :
- 0 paiement à la fois dans le train set et le validation set ;
- 6 jours entre le dernier jour du train et le premier de la validation ;
- 347 412 lignes vues par les 3 modèles de fold, soit 2 × 173 706, toutes du train set ;
- 0 paire positive du train labellisée par une imputation datée de la validation ou après.

**c) Aucune information future dans les features.**

*Théorie.*
- Toute lecture d'état passe par `LedgerState` avec un `as_of`. L'état n'avance que vers l'avant, en n'appliquant que
  les événements antérieurs à `as_of`, et une lecture à une autre date lève une `TemporalError`.
- `invoice.current_amount` est retiré au chargement : le restant dû est reconstruit par les imputations antérieures.
- Les agrégats comportementaux portent sur une fenêtre strictement antérieure, `[D − 180 j, D)`.
- Les client files ne sont visibles qu'après leur `received_at`.

*Vérification* (notebook, § 8) :
- **Truncation test.** On supprime du journal tout ce qui suit la fin de la validation, puis on reconstruit
  l'intégralité du training set. On obtient les mêmes 238 846 paires, avec **les mêmes valeurs sur toutes les
  features**. Seuls des labels changent (§ 2.4).
- **Audit feature par feature.** La meilleure feature seule, `amount_rank_in_payment`, plafonne à 0,585 de
  precision@1, contre 0,951 pour le modèle complet : aucune feature n'est un proxy du label.

Cette garantie a une limite : elle ne couvre pas les tables sans date (§ 2.5).

### 1.5 Pourquoi le leakage restant pèse peu

**La barrière du test.** Le test set est isolé par le split principal, et les seuils sont figés avant de l'ouvrir. Un
leakage qui intervient pendant le training ou la sélection (F1, F3, F4, nombre d'itérations des folds) a donc deux
effets possibles, et seulement ceux-là :
- un **modèle moins bon** que possible, ce qui se voit sur le test comme un taux d'automatisation plus bas, jamais plus
  haut ;
- des **métriques de validation optimistes**. C'est pourquoi nous ne communiquons que les chiffres du test.

Les résultats de référence (synthétique, 2 M paiements) sont mesurés de cette façon, avec les seuils choisis sur la
validation : 62,75 % d'automatisation avec les règles seules, 70,13 % avec règles + ML, à 99,61 % de précision.

**Les ordres de grandeur mesurés sont dans le bruit :**

| Leakage | Ce qui passe | Impact mesuré | Détail |
|---|---|---|---|
| F4 · CV non temporel | le futur du train, dans les features de compétition | la pass 2 entraînée sans futur fait au moins aussi bien (0,954 contre 0,948) | § 2.1 |
| Itérations des folds | un entier, `best_iteration` | à itérations égales, les scores OOF sont identiques au bit près | § 2.2 |
| F1 · validation réutilisée | des choix sur la validation | métriques de validation optimistes ; le test n'est pas touché | § 2.3 |
| F3 · labels sans horizon | 56 labels de fin de validation | precision@1 0,951 contre 0,948 sans eux | § 2.4 |

**La limite de cette barrière.** Elle ne protège pas contre un leakage **de données**, présent dans le journal
lui-même : il touche le replay du test comme celui du train. C'est le cas de F2, les attributs non datés (§ 2.5), et de
F5, les imputations datées avant leur paiement (§ 2.6). Le synthétique ne permet pas de les mesurer. Ce sont les
priorités du run sur les données réelles.

---

## 2. Description du leakage résiduel

Pour chaque leakage, nous donnons le mécanisme théorique, puis la vérification dans le notebook.

### 2.1 F4 : le CV des folds n'est pas temporel

**Théorie.** Un GroupKFold suppose les groupes **échangeables**, ce qui n'est pas le cas ici, pour deux raisons :
- **Dépendance par débiteur.** Les features `debtor_*`, `bankroll_code`, `channel`, `market` et les signaux
  d'allocation forment une quasi-empreinte du débiteur.
- **Non-stationnarité.** La relation entre features et label dérive dans le temps.

Avec `hash(payment_id) mod 3`, un paiement de janvier est scoré par un modèle de fold entraîné sur février à août,
y compris sur les paiements du même débiteur. L'ensemble d'information du score OOF est donc plus large que celui dont
dispose la pass 1 en production. Ce n'est pas un target leakage sur `y_p·`, mais un **look-ahead leakage**.

Ce leakage est confiné aux features de **training** de la pass 2. En validation, en test et en production, les
features de compétition viennent de la pass 1 appliquée à des paiements postérieurs. Il ne peut que dégrader le
modèle : la pass 2 apprend une relation marge → label plus favorable que celle qu'elle rencontrera.

**Vérification** (notebook, § 4.2 et § 6.1). On reproduit le scoring « comme en production » par un **walk-forward
mensuel** en expanding window. Pour scorer le mois k, un modèle est entraîné sur :
- les paiements arrivés avant `Dₖ − purge` ;
- avec les labels connus à `Dₖ` ;
- au même nombre d'itérations que les folds.

On entraîne ensuite la pass 2 sur chacune des deux sources de scores, à paires identiques, et on l'évalue sur la
validation :

| Pass 2, évaluée sur le validation set | precision@1 | MRR | log-loss |
|---|---|---|---|
| GroupKFold aléatoire, mêmes paires que le walk-forward | 0,948 | 0,968 | 0,0207 |
| walk-forward | 0,954 | 0,971 | 0,0201 |

Le walk-forward fait au moins aussi bien : +0,6 pt, soit environ 1,2 standard error. Le mécanisme est réel, son coût
n'est pas détectable sur le synthétique, et la correction ne coûte rien. Avec du drift, nous attendons un écart plus
net.

Au niveau des scores OOF eux-mêmes (precision@1 de 0,945 en aléatoire contre 0,931 en walk-forward), l'écart mélange
le futur et le volume : les premiers modèles walk-forward voient très peu de paires.

### 2.2 Un canal validation → train : le nombre d'itérations des folds

**Théorie.** Les modèles de fold ne voient aucune ligne de validation, mais leur nombre d'itérations vaut
`max(pass1.best_iteration, 10)` (`model.py:121`). `best_iteration` dépend des labels de validation par l'early
stopping. Les scores OOF, donc les features de compétition du train, dépendent ainsi du validation set, qui sert ensuite
à l'early stopping de la pass 2 et à la calibration.

**Vérification** (notebook, § 4.3, test T4) : deux tests causaux.
- **Contrôle.** Avec le même nombre d'itérations, les folds refaits donnent des scores OOF identiques au bit près :
  c'est le **seul** canal.
- **Shuffle des labels de validation.** L'early stopping s'arrête à 10 itérations au lieu de 195, les scores OOF
  changent de 0,051 en moyenne, et leur precision@1 passe de 0,945 à 0,915 : le canal est actif.

C'est un leakage faible, puisqu'un seul entier passe.

### 2.3 F1 : le validation set sert à choisir, puis à mesurer

**Théorie.** Le même validation set sert à cinq choix :
1. l'early stopping de la pass 1 ;
2. le nombre d'itérations des folds ;
3. l'early stopping de la pass 2 ;
4. le fit de l'isotonique ;
5. les seuils τ_high.

Les métriques de validation qu'on en tire sont biaisées vers le haut. La « précision des auto-validations » vaut même au
moins 99,5 % par construction.

Le point le plus sensible est τ_high, car il relève du **winner's curse** :
- `calibrate_online` retient le seuil le plus permissif dont la precision observée atteint la cible, sur au moins
  `MIN_KIND_VOLUME = 20` propositions (`src/reconcile_ml/decision.py:27`) ;
- une tranche de 20 propositions dont la vraie precision n'est que de 98 % passe ce test avec une probabilité de
  0,98²⁰ ≈ 67 % ;
- la precision sur le test peut donc tomber sous la cible, surtout pour les types à faible volume.

Le test le mesure honnêtement, mais c'est un risque sur la tenue de l'engagement de 99,5 %.

**Vérification** (notebook, § 6.2). Nous avons coupé la validation en deux : fit de l'isotonique sur V1, évaluation sur
V2. La log-loss sur V2 vaut 0,02115 avec l'isotonique, contre 0,02081 avec les scores bruts : la calibration
**dégrade** la log-loss out-of-sample. Elle écrase aussi les 5 % de scores les plus hauts en 17 valeurs distinctes. La
décision travaille déjà sur le score brut. Le biais sur τ_high n'est pas mesuré par le notebook : il le sera sur le test,
seuils actuels contre seuils corrigés.

### 2.4 F3 : des labels lus sans horizon

**Théorie.**
- `label_pairs` labellise à partir de toute la table `imputation`, sans condition sur `updated_at`.
- Un paiement de fin de période reçoit donc son label final, même s'il a été prononcé plus tard. La purge de 5 jours
  est courte face à une imputation manuelle qui prend souvent plusieurs semaines.
- Il y a aussi un biais de sélection. À la date d'un retrain réel, l'issue des paiements récents est inconnue ; ici,
  les paiements difficiles, imputés tard, sont labellisés grâce au futur.

**Vérification** (notebook, § 8.1, truncation test du § 1.4).
- Les paires et les features sont identiques, mais **56 labels basculent**, tous sur des paiements des derniers jours
  de la validation.
- Le modèle entraîné avec ces labels atteint 0,951 de precision@1, contre 0,948 sans : l'écart est dans le bruit.

Sur le synthétique, le P95 du délai d'imputation est de 3 jours. Sur les données réelles, c'est ce délai qui dira si F3
compte : s'il dépasse la purge, l'effet grandit.

### 2.5 F2 : des attributs statiques non datés (leakage de données)

**Théorie.**
- Plusieurs informations sont visibles dès le premier jour du replay, dans l'état où elles ont été extraites en fin
  d'historique : les IBAN des parties (`party_iban`), les variantes de nom, les comptes techniques, les champs de facture
  et les attributs de contrat.
- En affacturage, le référentiel se complète souvent à partir des paiements eux-mêmes. Un IBAN découvert lors d'un
  rapprochement manuel est alors connu **avant** le paiement qui l'a révélé.
- Pour ce paiement, `alloc_sig_iban` ou `iban_route = DEBTOR_DIRECT` est en partie un label déguisé.
- **Ce leakage touche aussi le replay du test** : la barrière du § 1.5 ne le neutralise pas.

**Vérification : pas encore possible.** Le truncation test ne peut pas le voir, puisque ces tables n'ont pas de date
(`party_iban` garde ses 938 lignes). Le synthétique ne le simule pas non plus. Une mesure dédiée est prévue sur les
données réelles (§ 3, point 5).

### 2.6 Leakage secondaire

- **F5 : imputation datée avant l'arrivée de son paiement** (leakage de données).
  - Si `updated_at` précède la date de valeur du paiement, l'état figé à son arrivée contient déjà l'imputation.
  - Cela touche aussi le test.
  - Le rapport de chargement compte ces cas (`avant_connaissance_du_paiement`), sans les corriger.
- **F6 : un KPI de l'interface choisit son seuil sur le test.**
  - `taux_automatisation_à_précision_cible` est un seuil oracle.
  - Les chiffres que nous communiquons sont ceux obtenus avec les seuils de validation (`taux_automatisation`), mais
    l'interface affiche l'oracle en premier.
- **F7 : vocabulaires de catégories construits sur toutes les données.** L'effet est faible, et le truncation test ne
  montre aucun changement d'encodage.

---

## 3. Propositions d'évolution

Les correctifs sont classés par priorité.

1. **Labelliser avec un horizon (F3).** C'est le prérequis des autres.
   - Labels à la date de cutoff C : seules les imputations avec `updated_at < C` comptent.
   - Un paiement sans issue connue à C et plus récent que `retention_days` est écarté, et non labellisé négatif, pour ne
     pas remplacer un leakage par du label noise.
   - Purge ≥ P95 du délai entre l'arrivée du paiement et son imputation.

2. **Remplacer le GroupKFold aléatoire par un walk-forward (F4).**
   - Retrain mensuel en expanding window : le modèle entraîné à Dₖ voit les paiements arrivés avant `Dₖ − purge`, avec
     les labels connus à Dₖ, et score `[Dₖ, Dₖ₊₁)`.
   - Aucun nouveau replay n'est nécessaire : les features de pass 1 ne dépendent pas du modèle, et le dataset enregistré
     contient déjà la colonne `day`.
   - Coût : environ 451 000 paires cumulées, contre 347 000 pour les 3 folds actuels, soit à peu près 1,3 fois.
   - Fallback : un GroupKFold par débiteur, moins juste.

3. **Fixer le nombre d'itérations sans la validation (§ 2.2).** Early stopping interne au train, sur son dernier mois
   après purge, ou choisi dans le walk-forward.

4. **Séparer les rôles de la validation (F1).**
   - V1 pour l'early stopping, V2 pour les seuils.
   - τ_high choisi sur une lower bound de Clopper-Pearson plutôt que sur la precision observée. Une borne ≥ 99,5 % à
     95 % impose environ 600 propositions toutes correctes (`n ≥ ln 0,05 / ln 0,995 ≈ 598`) ; relever
     `MIN_KIND_VOLUME` en conséquence.
   - Isotonique réservée à l'affichage, ou remplacée par du Platt scaling.
   - Au besoin, étendre la validation à 3 mois.

5. **Dater les attributs statiques (F2).**
   - Récupérer l'historique de `party_iban`, des noms et des champs de facture s'il existe, et en faire des événements du
     journal.
   - Sinon, estimer la date d'apparition d'un IBAN par le premier paiement imputé depuis cet IBAN.
   - **Mesure d'impact** : rejouer le test sans le signal IBAN pour les couples (débiteur, IBAN) découverts après D.

6. **Leakage secondaire.**
   - F5 : contrôle bloquant au-delà d'un seuil, ou décalage de l'imputation à `max(updated_at, arrivée + ε)`.
   - F6 : afficher en tête le taux obtenu avec les seuils de validation.
   - F7 : vocabulaires construits sur le seul train set.

7. **Automatiser le contrôle.** Ajouter un test anti-leakage du **training** :
   - même training set sur le journal complet et sur le journal tronqué, attributs datés compris ;
   - mêmes features, et mêmes labels une fois l'horizon appliqué.

Les points 1 à 4 sont locaux à `model.py`, `pipeline.py` et `decision.py`. Nous proposons de les implémenter en
premier, puis de comparer sur le test le taux d'automatisation à precision fixée, avant et après correction. Le point 5
dépend de l'historique disponible côté données.

Sur les données réelles, nous suivrons en priorité :
- le P95 du délai d'imputation et le nombre de labels qui basculent au truncation test (F3) ;
- l'écart walk-forward / GroupKFold sur la pass 2 (F4) ;
- l'ablation IBAN (F2) ;
- le compte `avant_connaissance_du_paiement` (F5).

Je vous envoie les résultats dès qu'ils sont disponibles. Le notebook déroule chaque étape avec ses sorties
intermédiaires, et `docs/fuites_ml.md` détaille chaque leakage avec ses références de code.

Bien à vous,

[Prénom Nom]
