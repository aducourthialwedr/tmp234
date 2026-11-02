"""Diagnostics partagés par l'interface et les outils (src/diagnostics.py), sur un petit jeu synthétique."""

import numpy as np
import pandas as pd
import pytest

from src import diagnostics as D
from src.api import Project
from src.reconcile_ml.decision import calibrate_online
from src.settings import DecisionSettings, LoadSettings, PathsSettings, Settings, save_settings


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    root = tmp_path_factory.mktemp("diag")
    settings = Settings(paths=PathsSettings(interim_dir=str(root / "interim"), reports_dir=str(root / "reports"),
                                            synthetic_dir=str(root / "synthetic")), load=LoadSettings(workers=1))
    save_settings(settings, root / "settings.yaml")
    p = Project(settings_path=root / "settings.yaml", log=lambda m: None)
    p.load(n_payments=3000)
    return p


def test_split_matches_project(project):
    assert D.data_split(project.interim_dir, project.settings.split) == project.split()


def test_open_invoice_stock_is_consistent(project):
    split = project.split()
    stock = D.open_invoice_stock(project.interim_dir, split)
    curve = stock["curve"]
    assert (curve["ouvertes"] >= 0).all() and curve["créées"].is_monotonic_increasing
    total = stock["total"]
    assert total["factures"] == len(pd.read_parquet(project.interim_dir / "invoice.parquet", columns=["invoice_id"]))
    assert stock["age"]["factures"].sum() == total["ouvertes au début du test"]
    assert stock["age"]["jamais imputées"].le(stock["age"]["factures"]).all()


def test_imputation_delays_and_batches(project):
    delays = D.imputation_delays(project.interim_dir, purge_days=5, retention_days=60)
    assert 0 <= delays["au-delà de la rétention"] <= delays["au-delà de la purge"] <= 1
    assert delays["describe"]["min"] >= 0
    batches = D.pending_batches(project.interim_dir, project.split(), retention_days=60)
    assert set(batches["période"]) == {"validation", "test"}
    assert (batches["lot (majorant)"] >= 0).all()


def test_client_file_conditions(project):
    cf = D.client_file_conditions(project.interim_dir, date_tolerance_days=5)
    assert cf is not None and cf["fichiers"] > 0
    assert cf["vrai paiement retrouvé"] <= cf["fichiers"]
    assert cf["conditions"].between(0, 1).all()
    assert abs(cf["références des lignes"].sum() - 1) < 1e-9


def test_iban_worst_case_by_period(project):
    f2 = D.iban_worst_case(project.interim_dir, project.split())
    table = f2["par période"]
    assert {"train", "validation", "test"} <= set(table.index)
    assert (table.iloc[:, 1] <= table.iloc[:, 0]).all()           # révélé ⊂ connu
    assert "auto-validations" not in f2                           # sans export de décisions


def test_clopper_pearson_lower():
    assert D.clopper_pearson_lower(100, 100) == pytest.approx(0.05 ** (1 / 100))
    assert D.clopper_pearson_lower(0, 0) != D.clopper_pearson_lower(0, 0)        # nan
    assert D.clopper_pearson_lower(10, 0) == 0.0
    assert D.clopper_pearson_lower(10_000, 9_970) < 0.997 < D.clopper_pearson_lower(10_000, 9_990)
    table = D.with_precision_bounds(pd.DataFrame({"auto": [100, 0], "corrects": [100, 0]}))
    assert table["précision min. 95 %"].iloc[0] == pytest.approx(0.9705, abs=1e-4)


def test_threshold_curve_agrees_with_online_calibration(tmp_path):
    rng = np.random.default_rng(0)
    n = 3000
    records = pd.DataFrame({"day": rng.integers(0, 5, n), "payment_id": [f"P{i}" for i in rng.integers(0, 800, n)],
                            "kind": rng.choice(["single", "set"], n), "score": rng.uniform(0.5, 1, n),
                            "margin": rng.uniform(0, 0.3, n)})
    records["correct"] = rng.uniform(0, 1, n) < records["score"] ** 4
    records = records.sort_values("day", kind="mergesort").reset_index(drop=True)
    path = tmp_path / "proposals.parquet"
    records.to_parquet(path)
    cfg = DecisionSettings()
    online = calibrate_online(records, records["correct"].to_numpy(), 0.9, cfg)
    curve = D.threshold_curve(path, cfg.min_margin, applied=online["kinds"])
    assert online["kinds"]["single"] <= 1
    for kind, tau in online["kinds"].items():
        sub = curve[curve["type"] == kind]
        assert sub["auto-validations"].is_monotonic_decreasing
        if tau <= 1:
            at = sub[sub["appliqué"]].iloc[0]
            assert at["τ_high"] == tau and at["précision"] >= 0.9
            assert (sub.loc[sub["τ_high"] < tau, "précision"] < 0.9).all()       # seuil le plus bas qui tient


def test_importance_by_family():
    imp = pd.DataFrame({"feature": ["alloc_rank", "amount_exact", "amount_diff", "inconnue"], "gain": [3, 2, 1, 4]})
    out = D.importance_by_family(imp).set_index("famille")
    assert out.loc["amount", "gain"] == 3 and out.loc["amount", "features"] == 2
    assert out.loc["autre", "gain"] == 4 and out["part du gain"].sum() == pytest.approx(1)
