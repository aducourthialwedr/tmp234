"""Export des imputations proposées par un rejeu, une ligne par couple (paiement, facture).

Destiné aux statistiques hors du moteur : le fichier se joint à la table d'imputation d'origine (et à
ses métadonnées) sur (payment_id, invoice_id). Il contient :
- chaque couple proposé par le moteur (auto-validé ou mis en revue), une ligne par couple et par type
  de décision, avec le premier et le dernier jour de proposition ;
- chaque couple **réellement imputé** que le moteur n'a pas proposé, pour mesurer ce qui manque.

Même périmètre et même vérité que l'évaluation (`src/evaluation/metrics.py`) : paiements arrivés dans
la période, imputations réelles toutes dates confondues.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.metrics import _proposals

PROPOSED_REAL, PROPOSED_NOT_REAL, REAL_NOT_PROPOSED = "proposée et réelle", "proposée non réelle", "réelle non proposée"

EXPORT_COLUMNS = [
    "payment_id", "invoice_id", "statut_paire", "action", "step", "rule_id", "rule_version",
    "montant_propose", "score_premier_jour", "score_max", "premier_jour", "dernier_jour", "jours_proposee",
    "decision_retenue", "action_retenue_paiement", "decision_retenue_exacte", "arrivee_paiement", "paiement_avec_imputation_reelle",
    "type_groupe_reel", "nb_factures_reelles",
]


def decision_pairs(decisions: pd.DataFrame, truth: pd.DataFrame, scope: pd.DataFrame) -> pd.DataFrame:
    """`decisions` : sortie d'un rejeu (colonnes DECISION_COLUMNS + day) ; `truth` : `ground_truth` ;
    `scope` : paiements du périmètre (payment_id, arrival_day)."""
    in_scope = set(scope["payment_id"])
    d = decisions[decisions["payment_id"].isin(in_scope) & decisions["invoice_id"].notna()
                  & decisions["action"].isin(["auto", "review"])]
    d = d.sort_values(["payment_id", "invoice_id", "action", "day"], kind="mergesort")
    keys = ["payment_id", "invoice_id", "action", "step", "rule_id"]
    g = d.groupby(keys, sort=False, dropna=False)
    proposed = g.agg(rule_version=("rule_version", "first"), montant_propose=("amount", "first"),
                     score_premier_jour=("score", "first"), score_max=("score", "max"),
                     premier_jour=("day", "min"), dernier_jour=("day", "max"),
                     jours_proposee=("day", "nunique")).reset_index()

    # Couples réellement imputés aux paiements du périmètre.
    real = truth[truth["payment_id"].isin(in_scope)]
    real_pairs = pd.DataFrame({"payment_id": np.repeat(real["payment_id"].to_numpy(), real["n_invoices"].to_numpy()),
                               "invoice_id": [i for t in real["truth_invoices"] for i in t]})
    real_set = set(zip(real_pairs["payment_id"], real_pairs["invoice_id"]))
    proposed["statut_paire"] = np.where(
        [(p, i) in real_set for p, i in zip(proposed["payment_id"], proposed["invoice_id"])],
        PROPOSED_REAL, PROPOSED_NOT_REAL)
    seen = set(zip(proposed["payment_id"], proposed["invoice_id"]))
    missed = real_pairs[[(p, i) not in seen for p, i in zip(real_pairs["payment_id"], real_pairs["invoice_id"])]]
    missed = missed.assign(statut_paire=REAL_NOT_PROPOSED)
    out = pd.concat([proposed, missed], ignore_index=True)

    # Décision retenue par l'évaluation pour le paiement (première auto-validation, sinon première revue),
    # et son exactitude au niveau du paiement.
    retained = _proposals(decisions[decisions["payment_id"].isin(in_scope)])
    t_map = truth.set_index("payment_id")["truth_invoices"]
    retained["decision_retenue_exacte"] = [t_map.get(p) == inv for p, inv in
                                           zip(retained["payment_id"], retained["invoices"])]
    retained_pairs = {(p, i, a) for p, inv, a in zip(retained["payment_id"], retained["invoices"], retained["action"])
                      for i in inv}
    out["decision_retenue"] = [(p, i, a) in retained_pairs for p, i, a in
                               zip(out["payment_id"], out["invoice_id"], out["action"])]
    out = out.merge(retained[["payment_id", "action", "decision_retenue_exacte"]].rename(
        columns={"action": "action_retenue_paiement"}), on="payment_id", how="left")
    out["action_retenue_paiement"] = out["action_retenue_paiement"].fillna("aucune")
    out = out.merge(scope[["payment_id", "arrival_day"]].rename(columns={"arrival_day": "arrivee_paiement"}),
                    on="payment_id", how="left")
    out = out.merge(truth[["payment_id", "group_type", "n_invoices"]].rename(
        columns={"group_type": "type_groupe_reel", "n_invoices": "nb_factures_reelles"}), on="payment_id", how="left")
    out["paiement_avec_imputation_reelle"] = out["type_groupe_reel"].notna()
    out["type_groupe_reel"] = out["type_groupe_reel"].fillna("sans imputation")
    out["nb_factures_reelles"] = out["nb_factures_reelles"].fillna(0).astype(int)
    out["jours_proposee"] = out["jours_proposee"].astype("Int64")
    out["decision_retenue_exacte"] = out["decision_retenue_exacte"].astype("boolean")   # vide : aucune décision
    order = {PROPOSED_REAL: 0, PROPOSED_NOT_REAL: 1, REAL_NOT_PROPOSED: 2}
    out = out.assign(_o=out["statut_paire"].map(order)).sort_values(["payment_id", "_o", "invoice_id"], kind="mergesort")
    return out[EXPORT_COLUMNS].reset_index(drop=True)
