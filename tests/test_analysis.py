"""Évaluation approfondie (src/evaluation/analysis.py) : issue, type d'erreur, raison du résiduel, segments."""

import pandas as pd
import pytest

from src.api import Project
from src.evaluation import analysis as A
from src.evaluation.metrics import evaluate, ground_truth
from src.settings import LoadSettings, PathsSettings, Settings, save_settings

DAY = pd.Timestamp("2024-03-01")


def _decision(pid, invoices, action, step, rule, score):
    return [{"payment_id": pid, "invoice_id": i, "amount": 100, "action": action, "step": step, "rule_id": rule,
             "rule_version": 1, "score": score, "day": DAY} for i in invoices]


@pytest.fixture
def world():
    payment = pd.DataFrame({"payment_id": [f"P{i}" for i in range(1, 8)],
                            "amount": [5_000, 50_000, 250_000, 12_000, 900, 3_000_000, 7_000],
                            "label": ["VIR A", "VIR B", "VIR C", "VIR D", "VIR E", "VIR F", "VIR G"],
                            "channel": ["SEPA"] * 6 + ["LCR"], "payment_type": "VIREMENT", "bankroll_code": None,
                            "iban_debtor": ["FR1", None, "FR3", "FR4", "FR5", "FR6", "FR7"]})
    invoice = pd.DataFrame({"invoice_id": [f"I{i}" for i in range(1, 10)],
                            "debtor_id": ["D1", "D1", "D2", "D2", "D1", "D3", "D9", "D2", "D3"],
                            "agreement_id": ["A1", "A1", "A2", "A2", "A1", "A3", "A9", "A2", "A3"],
                            "client_reference": [f"F{i}" for i in range(1, 10)]})
    agreement = pd.DataFrame({"agreement_id": ["A1", "A2", "A3", "A9"], "market": ["BTP", "SERVICES", "BTP", "BTP"],
                              "product": "CLASSIQUE"})
    # Vérité : P1→I1, P2→I2, P3→I3+I4, P4→I5, P5 jamais imputé, P6→I6, P7→I9.
    imputation = pd.DataFrame({"payment_id": ["P1", "P2", "P3", "P3", "P4", "P6", "P7"],
                               "invoice_id": ["I1", "I2", "I3", "I4", "I5", "I6", "I9"],
                               "updated_at": DAY + pd.to_timedelta([0, 1, 2, 2, 10, 3, 1], unit="D")})
    decisions = pd.DataFrame(
        _decision("P1", ["I1"], "auto", "rules", "R2_REFERENCE_UNIQUE", 1.0)        # juste
        + _decision("P2", ["I7"], "auto", "ml", "ML_SINGLE", 0.99)                 # mauvais débiteur
        + _decision("P3", ["I3"], "auto", "rules", "R4_EXACT_SUM", 1.0)            # ensemble incomplet
        + _decision("P4", ["I5"], "review", "ml", "ML_SINGLE", 0.90)               # juste, sous τ_high
        + _decision("P5", ["I8"], "auto", "rules", "R3_AMOUNT_UNIQUE", 1.0))       # jamais imputé
    candidates = pd.DataFrame({"payment_id": ["P2", "P2", "P4", "P6", "P6", "P7"],
                               "invoice_id": ["I7", "I2", "I5", "I8", "I6", "I1"],
                               "p": [0.99, 0.5, 0.9, 0.7, 0.4, 0.3]})
    info = pd.DataFrame({"payment_id": [f"P{i}" for i in range(1, 8)], "client_file_id": [None] * 6 + ["CF1"],
                         "allocation_status": ["ferme"] * 5 + ["multiple", "ferme"],
                         "iban_route": "DEBTOR_DIRECT",
                         "allocation_debtors": [["D1"], ["D9"], ["D2"], ["D1"], ["D2"], ["D3", "D2"], ["D1"]]})
    scope = pd.DataFrame({"payment_id": payment["payment_id"], "arrival_day": DAY})
    truth = ground_truth(imputation)
    return dict(decisions=decisions, truth=truth, scope=scope, payment=payment, invoice=invoice, agreement=agreement,
                imputation=imputation, candidates=candidates, info=info,
                thresholds={"tau_high": 1.01, "kinds": {"single": 0.95, "set": 0.97, "partial": 1.01}})


def test_issues_errors_and_reasons(world):
    p = A.analyze_payments(**world).set_index("payment_id")
    assert p["issue"].to_dict() == {"P1": A.AUTO_OK, "P2": A.AUTO_KO, "P3": A.AUTO_KO, "P4": A.IN_REVIEW,
                                    "P5": A.AUTO_KO, "P6": A.NO_DECISION, "P7": A.NO_DECISION}
    assert p.loc["P2", "type_erreur"] == "mauvais débiteur"
    assert p.loc["P3", "type_erreur"] == "ensemble incomplet"
    assert p.loc["P5", "type_erreur"] == "paiement jamais imputé"
    assert p.loc["P4", "raison"] == "proposition juste, score sous τ_high"
    assert p.loc["P6", "raison"] == "vraie facture candidate, pas en tête"
    assert p.loc["P6", "rang_vraie"] == 2
    assert p.loc["P7", "raison"] == "vraie facture absente des candidats (débiteur non alloué)"
    assert p.loc["P1", "raison"] == "" and p.loc["P1", "factures_réelles"] == "F1"
    assert p.loc["P2", "débiteur_proposé"] == "D9" and p.loc["P2", "débiteur_réel"] == "D1"
    assert p.loc["P6", "tranche_montant"] == "10 000 – 100 000 €" and p.loc["P7", "canal"] == "LCR"
    assert p.loc["P7", "client_file"] == "avec client file" and p.loc["P2", "iban"] == "IBAN absent"
    assert p.loc["P3", "marché"] == "SERVICES" and p.loc["P5", "marché"] == A.MISSING


def test_consistent_with_evaluate(world):
    p = A.analyze_payments(**world)
    ev = evaluate(world["decisions"], world["truth"], world["scope"], 0.995).summary
    s = A.summary(p)
    assert s["auto juste"] == ev["auto_corrects"] and s["auto juste"] + s["auto fausse"] == ev["auto"]
    assert s["revue"] == ev["revue"] and s["taux_automatisation"] == pytest.approx(ev["taux_automatisation"])


def test_tables(world):
    p = A.analyze_payments(**world)
    by_channel = A.rate_by(p, "canal").set_index("canal")
    assert by_channel.loc["SEPA", "paiements"] == 6 and by_channel.loc["SEPA", "auto"] == 4
    assert by_channel["paiements"].sum() == len(p)
    reasons = A.reason_table(p)
    assert reasons["paiements"].sum() == 3 and reasons["gain maximal (points)"].sum() == pytest.approx(300 / 7)
    errors = A.error_table(p).set_index("type d'erreur")
    assert errors["total"].sum() == 3 and errors.loc["mauvais débiteur", "ML_SINGLE"] == 1
    before = p.assign(issue=A.NO_DECISION)
    table = A.compare_outcomes(before, p)
    assert table.loc[A.NO_DECISION].sum() == len(p) and table.loc[A.NO_DECISION, A.AUTO_OK] == 1


def test_rules_only_reasons_use_allocation(world):
    rules_only = {**world, "candidates": None, "thresholds": None,
                  "decisions": world["decisions"][world["decisions"]["step"] == "rules"]}
    p = A.analyze_payments(**rules_only).set_index("payment_id")
    assert p.loc["P2", "raison"] == "vrai débiteur absent de l'allocation"
    assert p.loc["P4", "raison"] == "aucune règle applicable sans ambiguïté"


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    root = tmp_path_factory.mktemp("analysis")
    settings = Settings(paths=PathsSettings(interim_dir=str(root / "interim"), reports_dir=str(root / "reports"),
                                            synthetic_dir=str(root / "synthetic")), load=LoadSettings(workers=1))
    save_settings(settings, root / "settings.yaml")
    p = Project(settings_path=root / "settings.yaml", log=lambda m: None)
    p.load(n_payments=3000)
    p.backtest("validation", "rules")
    return p


def test_backtest_writes_analysis_with_allocation(project):
    info = pd.read_parquet(project.interim_dir / "replay" / "replay_rules_validation_payment_info.parquet")
    assert {"allocation_status", "iban_route", "allocation_debtors", "client_file_id"} <= set(info.columns)
    p = pd.read_parquet(project.reports_dir / "analysis_rules_validation.parquet")
    ev = project.report("rules", "validation")["summary"]
    assert len(p) == ev["paiements_périmètre"]
    assert (p["issue"] == A.AUTO_OK).sum() == ev["auto_corrects"]
    assert p["vrai_débiteur_alloué"].notna().any()
    assert set(p.loc[p["raison"] != "", "raison"]) <= set(A.REASONS)
