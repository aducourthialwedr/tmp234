"""Export des couples (paiement, facture) proposés, comparés aux imputations réelles."""

import pandas as pd

from src.evaluation.export import PROPOSED_NOT_REAL, PROPOSED_REAL, REAL_NOT_PROPOSED, decision_pairs
from src.evaluation.metrics import ground_truth


def dec(pid, inv, action, day, step="rules", rule="R2_REFERENCE_UNIQUE", score=1.0):
    return {"payment_id": pid, "invoice_id": inv, "amount": 100, "action": action, "step": step, "rule_id": rule,
            "rule_version": 1, "score": score, "day": pd.Timestamp(day)}


def test_decision_pairs():
    decisions = pd.DataFrame([
        dec("P1", "I1", "auto", "2024-03-01"),                                            # exacte
        dec("P2", "I2", "review", "2024-03-01", "ml", "ML_SINGLE", 0.4),                  # revue répétée
        dec("P2", "I2", "review", "2024-03-02", "ml", "ML_SINGLE", 0.6),
        dec("P2", "I3", "auto", "2024-03-03", "ml", "ML_SET", 0.99),                      # auto incomplète
        dec("P9", "I9", "auto", "2024-03-01"),                                            # hors périmètre
    ])
    imputation = pd.DataFrame({"payment_id": ["P1", "P2", "P2", "P3"], "invoice_id": ["I1", "I3", "I4", "I5"]})
    scope = pd.DataFrame({"payment_id": ["P1", "P2", "P3"], "arrival_day": pd.Timestamp("2024-03-01")})
    out = decision_pairs(decisions, ground_truth(imputation), scope).set_index(["payment_id", "invoice_id", "action"]).sort_index()
    assert "P9" not in out.index.get_level_values(0)
    p1 = out.loc[("P1", "I1", "auto")]
    assert p1["statut_paire"] == PROPOSED_REAL and p1["decision_retenue"] and p1["decision_retenue_exacte"]
    review = out.loc[("P2", "I2", "review")]
    assert review["statut_paire"] == PROPOSED_NOT_REAL and review["jours_proposee"] == 2
    assert review["score_premier_jour"] == 0.4 and review["score_max"] == 0.6 and not review["decision_retenue"]
    auto = out.loc[("P2", "I3", "auto")]
    assert auto["decision_retenue"] and auto["action_retenue_paiement"] == "auto"
    assert not auto["decision_retenue_exacte"]                      # I4 manque
    missed = out[out["statut_paire"] == REAL_NOT_PROPOSED].index.droplevel(2).tolist()
    assert sorted(missed) == [("P2", "I4"), ("P3", "I5")]
    assert out.reset_index().set_index("payment_id").loc["P3", "action_retenue_paiement"] == "aucune"
