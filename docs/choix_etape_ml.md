# Les choix de conception de l'étape ML

> **Résumé.** L'étape 5 traite les paiements que les règles n'ont pas résolus. Elle est découpée en
> quatre décisions indépendantes :
> - **quelles factures** regarder (génération de candidats) ;
> - **comment les noter** (un classifieur de paires LightGBM en deux passes, calibré) ;
> - **comment en faire une imputation** (résolution des ensembles, non apprise) ;
> - **quand décider seul** (seuils par type, calibrés en rejouant la validation jour par jour).
>
> Chaque choix ci-dessous est justifié, comparé aux alternatives, et relié au code. Les points
> faibles connus et les pistes d'amélioration terminent le document. Pour explorer ou tester ces
> choix sur les données : `notebooks/exploration_ml.ipynb`.

Le détail technique (features, hyperparamètres, fichiers) est dans
[docs/codebase/04_modeles_ml.md](codebase/04_modeles_ml.md). Ce document explique **pourquoi**.

---

## 1. Le cadre imposé par le problème

Contraintes qui orientent tous les choix (`CLAUDE.md` §1, §7) :

| Contrainte | Conséquence |
|---|---|
| **Précision ≥ 99,5 %** sur les décisions automatiques | on optimise le volume automatisé **à précision fixée**, pas une métrique moyenne. Il faut des scores comparables à un seuil et un mécanisme de refus (revue) |
| **Nombre de candidats variable** : une facture parmi des milliers d'ouvertes | impossible de traiter le problème comme une classification à classes fixes |
| **Solutions à plusieurs factures** (1↔n) et partielles | la sortie n'est pas une facture mais un **ensemble** avec des montants |
| **Pas de fuite temporelle** et un seul code pour le rejeu et la production | tout est calculé dans la boucle quotidienne, sur l'état de la veille |
| **Volume réel** : environ 2 M de paiements par an, des débiteurs à milliers de factures | coût et mémoire bornés, modèle rapide à l'inférence |
| **Auditabilité** (contexte bancaire) | décisions explicables, traçables (règle ou type, score, version), seuils lisibles |
| **Résiduel des règles seulement** | le ML ne voit que les cas difficiles ; sa valeur se mesure en points d'automatisation **ajoutés** aux règles |

---

## 2. Décision 1 — générer des candidats plutôt que tout scorer

**Choix.** Pour chaque paiement, une union de quatre sources (`Featurizer.candidates`,
`src/reconcile_ml/features.py:136`) :
1. les factures ouvertes des 3 premiers débiteurs de l'allocation, avec une échéance entre −180 et
   +30 jours ;
2. les factures dont une clé de référence est dans le libellé ;
3. les factures ouvertes de montant exact, créées à ±90 jours ;
4. les factures citées par le client file.

Viennent ensuite des filtres durs (devise, facture ouverte, contrat actif) et un plafond par paiement
(`max_per_payment`).

**Pourquoi.**
- Scorer toutes les factures ouvertes (des millions) pour chaque paiement est impossible.
- Chaque source correspond à une façon dont un humain retrouve la facture.
- Les sources précises (référence, montant, client file) passent avant les factures « du débiteur »
  dans le plafond (`_cap`), pour ne pas perdre les vraies factures des paiements groupés. C'était un
  problème constaté puis corrigé pendant le développement.

**Prix à payer.** Une facture absente des candidats ne peut jamais être trouvée. Sur le synthétique,
le **rappel des candidats est d'environ 80 %** (`rappel_candidats_validation` = 0,798) : c'est le
**plafond** de l'étape ML, et le premier levier d'amélioration (§9).

**Alternatives écartées.**
- **Recherche par similarité** (plus proches voisins sur un plongement des libellés) : plus coûteux,
  moins auditable, et redondant avec les index exacts de référence et de montant.
- **Un seul candidat par paiement** (le meilleur selon l'allocation) : perd tous les paiements
  ambigus, qui sont justement le résiduel.

---

## 3. Décision 2 — un classifieur de paires (paiement, facture)

**Choix.** Un modèle **binaire** prédit, pour chaque paire, la probabilité que la facture fasse partie
de l'imputation du paiement. Les négatifs sont **tous les autres candidats du même paiement, sans
échantillonnage** (`CLAUDE.md:166`).

**Pourquoi.**
- **Ensembles de taille variable** : le modèle ne dépend pas du nombre de candidats.
- **Paiements groupés** : plusieurs factures peuvent valoir 1, ce qu'une formulation « une seule
  bonne réponse » interdirait.
- **Négatifs complets** : sans échantillonnage, le modèle apprend sur la même distribution que celle
  qu'il verra en production. C'est indispensable pour des probabilités calibrées, donc pour des
  seuils à 99,5 %.
- **Mesure naturelle** : AUC, precision@1 et MRR par paiement (`pair_metrics`).

**Alternatives considérées.**

| Alternative | Pour | Contre | Statut |
|---|---|---|---|
| Classement « listwise » (LightGBM `lambdarank`) | optimise directement l'ordre au sein du paiement | scores non calibrés, moins adaptés à un seuil de précision absolu ; gère mal plusieurs bonnes réponses | non retenu ; une partie de l'information de classement est apportée par la passe 2 (§5). Piste à tester (§9) |
| Modèle au niveau du **paiement** (« ce paiement est-il automatisable ? ») | correspond directement à la décision | ne dit pas **quelles** factures ; jeu bien plus petit | non retenu ; la décision de paiement est construite à partir des paires (§6) |
| Prédiction directe d'un ensemble (séquentiel, attention) | modélise les interactions | complexe, peu auditable, coûteux sur 2 M de paiements | écarté pour un POC |
| Programmation linéaire (affectation globale paiements ↔ factures) | cohérence globale du lot | combinatoire lourde ; la cohérence est déjà assurée par l'arbitrage et les réservations | écarté |

---

## 4. Décision 3 — LightGBM

**Choix.** LightGBM binaire, hyperparamètres sobres : 63 feuilles, taux d'apprentissage 0,05, au plus
400 arbres avec arrêt anticipé, `min_data_in_leaf` 50, `feature_fraction` 0,9
(`_params`, `src/reconcile_ml/model.py:57`).

**Pourquoi.**
- **Données tabulaires hétérogènes** : écarts de montant, dates, booléens, catégories (marché, route
  IBAN). Le gradient boosting d'arbres reste la référence sur ce type de données.
- **Valeurs manquantes natives** : agrégats sans historique, débiteur absent de l'allocation. Pas
  d'imputation arbitraire.
- **Catégories natives** (`CATEGORICAL`), vocabulaires figés dans le modèle pour l'inférence.
- **Rapide** à l'apprentissage (quelques minutes pour 2 M de paires) et à l'inférence (le rejeu
  re-score le reliquat chaque jour).
- **Déterministe** (`deterministic=True`, seed fixe) et explicable : importance par gain, valeurs SHAP
  natives (`pred_contrib`), utilisées dans le notebook d'exploration.

**Alternatives.**
- **Régression logistique** : trop linéaire pour des seuils comme « escompte entre 0,5 et 3 % ».
- **Réseaux de neurones** : pas de gain attendu sur ce tabulaire, plus coûteux et moins explicables.
- **XGBoost / CatBoost** : équivalents ; LightGBM a été retenu pour sa vitesse et sa gestion des
  catégories.

---

## 5. Décision 4 — deux passes, avec des features de compétition

**Choix.**
- **Passe 1** : modèle sur les features de la paire.
- **Passe 2** : même modèle, plus trois features calculées au sein du paiement à partir des scores de
  passe 1 : rang, écart au meilleur concurrent, score du meilleur concurrent (`competition_features`,
  `src/reconcile_ml/model.py:37`).
- Sur le train, ces features viennent de **scores hors échantillon** (3 plis par paiement).

**Pourquoi.**
- Un score de paire isolé ignore les concurrents : 0,9 n'a pas le même sens si un autre candidat a
  0,88. La passe 2 apporte ce contexte, ce que le brief demande (`CLAUDE.md:170`).
- **Hors échantillon** : sinon, sur le train, les scores de passe 1 seraient trop optimistes (le
  modèle a vu ces paires), et la passe 2 apprendrait à faire trop confiance au rang. C'est une fuite
  classique des modèles empilés.

**Prix à payer.** Trois apprentissages de plus (les plis). La passe 2 se désactive par
`reconcile_ml.second_pass: false`.

---

## 6. Décision 5 — calibrer, mais décider sur le score brut

**Choix.** Une régression isotonique transforme le score brut de passe 2 en probabilité, ajustée sur
la validation. **Les décisions utilisent pourtant le score brut** ; le score calibré sert aux
diagnostics (`ml_calibration`).

**Pourquoi.**
- **La calibration rend les scores lisibles** : « 0,9 » ≈ 90 % de chances. Elle permet de vérifier
  que le modèle n'est pas trop confiant dans les tranches hautes.
- **Mais l'isotonique est une fonction en escalier.** Beaucoup de paires partagent la même
  probabilité calibrée près de 1, ce qui empêche de placer un seuil fin. Le score brut a le même ordre
  (la calibration est monotone) sans ces paliers. Ce problème a été constaté pendant le
  développement : les seuils ne trouvaient pas de point de coupure.

**Alternative.** Une calibration de Platt (sigmoïde) n'a pas de paliers mais suppose une forme
paramétrique. Elle n'a pas été nécessaire, puisque les décisions passent par le score brut.

---

## 7. Décision 6 — résoudre les ensembles sans apprentissage

**Choix.** `propose` (`src/reconcile_ml/decision.py:30`) construit, pour chaque paiement :
- des propositions **facture seule** : les 2 meilleurs candidats qui absorbent le paiement, de type
  `single` (soldée) ou `partial` ;
- des propositions **ensemble** (`set`) : les sous-ensembles des 25 meilleurs candidats dont la somme
  approche le paiement à 5 € ou 3 % près, cardinalité minimale préférée.

Le score d'un ensemble est la moyenne de ses paires. La meilleure proposition gagne. La marge est
l'écart à la meilleure proposition **différente**.

**Pourquoi.**
- **La contrainte de somme est arithmétique, pas statistique** : un DFS borné la vérifie exactement,
  et c'est auditable.
- **La moyenne des scores reste interprétable** et comparable aux scores de factures seules.
- **Règle d'ambiguïté.** Une facture seule qui n'explique le paiement qu'à un escompte près, face à un
  ensemble qui l'explique exactement, reçoit une marge de 0, donc part en revue. Constat du
  développement : les factures seules « escomptées » battaient à tort des ensembles exacts.

**Limite.** Si le budget de nœuds est atteint, l'information est perdue : un ensemble concurrent peut
manquer et la marge être surestimée (`08`, F18).

---

## 8. Décision 7 — seuils par type, calibrés en rejouant la validation

**Choix.**
- **Décision** : auto-validation si score ≥ τ_high **du type** (`single`, `set`, `partial`) **et**
  marge ≥ δ (0,05) ; revue entre τ_low (0,3) et τ_high ; sinon le paiement reste en attente.
- **Calibration de τ_high** : on rejoue toute la validation avec le ML en mode « propositions
  seulement ». On simule, pour chaque τ, **le premier jour où le score d'un paiement franchit τ**, et
  l'on retient le plus petit τ qui tient 99,5 % sur au moins 20 cas (`calibrate_online`,
  `src/reconcile_ml/decision.py:154`).

**Pourquoi.**
- **Par type** : les fiabilités diffèrent nettement. Sur le synthétique, la meilleure proposition de
  chaque paiement est exacte environ 61 % du temps pour `single`, 56 % pour `partial`, 15 % pour
  `set`. Un seuil unique serait dicté par le type le plus faible. Les partiels n'atteignent pas la
  précision requise : τ = 1,01, donc jamais d'auto-validation (synthétique : `partial` à 1,01,
  `set` ≈ 0,9986, `single` ≈ 0,9976).
- **En rejouant la boucle** : un paiement en attente est re-scoré chaque jour. Il a donc plusieurs
  chances de franchir le seuil, et c'est le **premier** franchissement qui décide. Un seuil choisi
  sur une seule photo par paiement (calibration « hors boucle ») est optimiste. Ce biais a été mesuré
  pendant le développement, d'où la calibration en ligne. Les seuils hors boucle sont conservés pour
  comparaison.
- **La marge δ** évite d'auto-valider quand deux propositions différentes sont presque à égalité.
- **La bande de revue** donne aux gestionnaires des propositions classées, sans risque.

**Alternative non activée.** Les seuils **par segment** (marché, bankroll, tranche de montant, client
file) existent dans le code hors boucle, mais n'atteignent pas les décisions (`08`, N2). Il faudrait
les porter dans la calibration en ligne.

---

## 9. Décision 8 — construire le jeu d'entraînement par rejeu

**Choix.** On rejoue train + validation avec les **règles actives**. Pour chaque paiement **à son jour
d'arrivée**, non résolu par les règles, on enregistre tous ses candidats et leurs features, calculées
sur l'état de la veille. L'étiquetage se fait ensuite, dans une passe séparée, à partir des
imputations (`fit_ml`, `src/reconcile_ml/pipeline.py:359`). Une part des paiements suffit
(`payment_sample` = 0,5).

**Pourquoi.**
- **Mêmes features en apprentissage et en production**, sans fuite : c'est le même code, sur l'état
  daté.
- **Le ML apprend sur ce qu'il verra** : le résiduel des règles, pas les cas faciles.
- **Étiquetage séparé** : aucune information de vérité ne peut contaminer les features.
- **Jour d'arrivée seulement** : une étiquette par paiement, sans dupliquer les paiements qui restent
  des semaines dans le reliquat. Le coût d'apprentissage reste maîtrisé.

**Prix à payer.**
- **Écart entre apprentissage et production.** En production, le reliquat est re-scoré chaque jour
  avec un état qui a évolué (réservations du ML, nouvelles factures). La calibration en ligne
  corrige les seuils, pas le modèle.
- **La validation sert à tout** : arrêt anticipé, calibration, seuils. Ses métriques sont donc
  optimistes, et seul le test est propre.

---

## 10. Ce que ces choix donnent (jeu synthétique, à confirmer sur données réelles)

| Mesure | Valeur | Source |
|---|---|---|
| AUC des paires (validation) | 0,9988 | `model.json` |
| precision@1 / MRR (validation, paiements avec une vraie candidate) | 95,7 % / 0,971 | `model.json` |
| Rappel des candidats (validation) | 79,8 % | `model.json` |
| Apport du ML sur le test | **+7,4 points** d'automatisation, précision 99,64 % | `reports/synthetic/evaluation_pipeline_test.md` |
| Cascade (test) | règles 62,75 % → règles + ML 70,13 %, à 99,61 % | idem |
| Plafond optimiste (courbe) | 74,2 % à 99,5 % | idem |

Ablation rapide par famille, sur un échantillon de 3 % (notebook d'exploration, §8). Sans la famille
**montant**, la precision@1 perd environ 4 points ; sans **temporel**, environ 3 points ; sans
**allocation**, environ 2 points. Les autres familles comptent peu sur le synthétique, dont la
réalité est simplifiée.

---

## 11. Pistes d'amélioration, par ordre de levier attendu

1. **Rappel des candidats (plafond d'environ 80 %).** Mesurer pourquoi la vraie facture manque
   (débiteur hors des 3 premiers, échéance hors fenêtre, plafond, référence non reconnue), puis
   élargir la source concernée. Le notebook (§3) montre quelles sources produisent les positifs.
2. **Apprendre aussi sur les jours de re-scorage** (pas seulement le jour d'arrivée), ou ajouter
   « jours en attente » comme feature, pour réduire l'écart entre apprentissage et production.
3. **Tester `lambdarank`** en passe 2 (le score de passe 1 restant calibrable) et comparer la
   precision@1 et le volume automatisé à précision fixée.
4. **Seuils par segment dans la calibration en ligne** : marché, présence de client file, tranche de
   montant.
5. **Nouvelles features** issues de l'analyse d'erreurs (notebook §9–10) : historique de citation du
   cédant, régularité des montants du débiteur, montant rond, `debtor_client_file_rate`
   (demandé par le brief, absent).
6. **Nettoyer** : supprimer `client_file_total_matches_payment` (copie de `has_client_file`) et les
   features redondantes repérées par le notebook (§6).
7. **Paiements groupés n↔n** : aujourd'hui une heuristique toujours en revue. Un modèle au niveau du
   débiteur et de la fenêtre serait l'étape suivante si leur volume le justifie.
8. **LLM sur les libellés** (`CLAUDE.md` §7.6) : seulement si l'analyse des erreurs montre des
   références bruitées que la normalisation ne capte pas.

**Méthode pour chaque piste** :
1. prototyper dans `notebooks/exploration_ml.ipynb` sur la validation ;
2. implémenter dans `Featurizer` ou `decision.py`, en incrémentant `FEATURIZATION_VERSION` si les
   features changent ;
3. relancer `project.train()` ;
4. juger **uniquement** sur le backtest de la période de test, au taux d'automatisation à précision
   fixée.
