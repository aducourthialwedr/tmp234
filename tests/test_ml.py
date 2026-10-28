"""Étape 5 — réconciliation ML."""

import hashlib
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.reconcile_ml.decision import calibrate_online, decide, propose
from src.reconcile_ml.model import GROUP, PairModel, competition_features
from src.reconcile_ml.pipeline import PipelineMatcher, fit_ml
from src.settings import DecisionSettings, RulesConfig, SetSettings, Settings
from src.timeline.loop import run_replay
from tests.test_timeline import ledger, truncate


def scored(rows):
    return pd.DataFrame(rows, columns=["row", "inv", "balance", "p"])


def test_single_exact_and_partial():
    cfg = SetSettings()
    props = propose(scored([(0, 1, 100_000, 0.9), (0, 2, 80_000, 0.2), (1, 3, 500_000, 0.8)]),
                    np.array([100_000, 200_000]), cfg)
    p = props.set_index("row")
    assert p.loc[0, "invoices"] == (1,) and p.loc[0, "kind"] == "single" and p.loc[0, "amounts"] == (100_000,)
    assert p.loc[1, "kind"] == "partial" and p.loc[1, "amounts"] == (200_000,)
    assert p.loc[0, "margin"] == pytest.approx(0.9)          # I2 n'absorbe pas le paiement : pas de concurrent


def test_set_beats_incomplete_single():
    cfg = SetSettings()
    # Paiement de 1500 = 1000 + 500 ; une facture seule n'absorbe pas le paiement.
    props = propose(scored([(0, 1, 100_000, 0.95), (0, 2, 50_000, 0.9), (0, 3, 30_000, 0.1)]),
                    np.array([150_000]), cfg)
    assert props.iloc[0]["invoices"] == (1, 2) and props.iloc[0]["kind"] == "set"


def test_discount_single_competes_with_exact_set():
    cfg = SetSettings()
    # 1000 soldée à 2 % d'escompte près, ou 980 + 20 exactement : les deux sont proposés, marge faible.
    props = propose(scored([(0, 1, 100_000, 0.9), (0, 2, 90_000, 0.88), (0, 3, 8_000, 0.87)]),
                    np.array([98_000]), cfg)
    assert props.iloc[0]["margin"] < 0.05


def test_decide_by_kind():
    props = pd.DataFrame({"score": [0.99, 0.99, 0.5], "margin": [0.5, 0.01, 0.5], "kind": ["single", "single", "set"]})
    th = {"tau_high": 1.01, "tau_low": 0.3, "min_margin": 0.05, "kinds": {"single": 0.9, "set": 0.95}}
    assert decide(props, th).tolist() == ["auto", "review", "review"]


def test_online_calibration_counts_first_crossing():
    # Paiement A : score 0,6 (faux) puis 0,95 (juste) ; B : 0,9 (faux). Avec τ = 0,95 seul A-jour-2 passe.
    n = 40
    records = pd.DataFrame({"day": [1, 2, 1] * n, "payment_id": [f"A{i // 3}" if i % 3 < 2 else f"B{i // 3}"
                                                            for i in range(3 * n)],
                            "kind": "single", "score": [0.6, 0.95, 0.9] * n, "margin": 0.5})
    correct = np.array([False, True, False] * n)
    th = calibrate_online(records, correct, 0.99, DecisionSettings())
    assert th["kinds"]["single"] == pytest.approx(0.95)


def test_competition_features():
    comp = competition_features(np.array([0.2, 0.9, 0.5, 0.7]), np.array(["a", "a", "a", "b"]))
    assert comp["rank_in_payment"].tolist() == [3, 1, 2, 1]
    assert comp["score_margin"].tolist() == pytest.approx([-0.7, 0.4, -0.4, 0.7], abs=1e-6)


# --- Entraînement et boucle sur jeu synthétique ---------------------------------------------------------

@pytest.fixture(scope="module")
def trained(tmp_path_factory, synthetic_dir):
    from src.config import REPO_ROOT, read_yaml
    from src.load.events import build_journal, journal_hash
    from src.load.loader import load_all
    data = load_all(read_yaml(REPO_ROOT / "config" / "schema.synthetic.yaml"), synthetic_dir)
    journal, _ = build_journal(data)
    root = tmp_path_factory.mktemp("ml")
    interim = root / "interim"
    interim.mkdir()
    for name, df in data.tables.items():
        df.to_parquet(interim / f"{name}.parquet", index=False)
    journal.to_parquet(interim / "journal.parquet", index=False)
    import json
    (interim / "journal_meta.json").write_text(json.dumps({"journal_sha256": journal_hash(journal)}))
    settings = Settings.model_validate({"reconcile_ml": {"training": {"payment_sample": 1.0, "num_boost_round": 60}}})
    meta = fit_ml(interim, root / "model", settings, RulesConfig(), log=lambda m: None)
    trained_dirs[:] = [interim, root]
    return data, settings, PairModel.load(root / "model"), meta


trained_dirs: list = []                 # répertoires de l'apprentissage du module (reprise du jeu)


def test_fit_produces_versioned_model(trained):
    _, _, model, meta = trained
    assert meta["journal_sha256"] and meta["featurization_version"]
    assert set(meta["thresholds"]["kinds"]) <= {"single", "partial", "set"}
    assert meta["metrics"]["validation"]["auc"] > 0.9


def test_model_roundtrip(trained, tmp_path):
    data, settings, model, _ = trained
    ds = pd.DataFrame(np.random.default_rng(0).random((20, len(model.features))), columns=model.features)
    ds[GROUP] = np.repeat(["a", "b"], 10)
    model.save(tmp_path / "m")
    again = PairModel.load(tmp_path / "m")
    assert np.allclose(model.predict(ds, ds[GROUP].to_numpy()), again.predict(ds, ds[GROUP].to_numpy()))


def _pipeline_hashes(data, settings, model, start, end):
    state = ledger(data)
    matcher = PipelineMatcher(state, settings, RulesConfig(), model)
    hashes = {}

    def hook(ctx, row):
        pass

    class Probe:
        name = "probe"

        def process(self, ctx):
            dec = matcher.process(ctx)
            hashes[ctx.day] = hashlib.sha256(dec.sort_values(["payment_id", "invoice_id"]).round(9)
                                             .to_csv(index=False).encode()).hexdigest()
            return dec

    res = run_replay(state, Probe(), start, end, retention_days=30)
    return hashes, res


def test_pipeline_runs_without_leak(trained):
    data, settings, model, _ = trained
    start, end, cutoff_day = date(2024, 11, 5), date(2024, 11, 30), date(2024, 11, 18)
    full, res = _pipeline_hashes(data, settings, model, start, end)
    assert (res.decisions["step"] == "ml").any()
    cut, _ = _pipeline_hashes(truncate(data, pd.Timestamp(cutoff_day) + pd.Timedelta(days=1)), settings, model,
                              start, end)
    days = [d for d in full if d.date() <= cutoff_day]
    assert [full[d] for d in days] == [cut[d] for d in days]


def test_candidates_in_blocks_match_single_pass(trained, monkeypatch):
    """Découpage du lot par charge de paires : décisions identiques à un passage unique."""
    import src.reconcile_ml.features as features
    data, settings, model, _ = trained
    start, end = date(2024, 11, 5), date(2024, 11, 8)
    whole, _ = _pipeline_hashes(data, settings, model, start, end)
    monkeypatch.setattr(features, "CANDIDATE_PAIR_BUDGET", 500)
    blocks, _ = _pipeline_hashes(data, settings, model, start, end)
    assert whole == blocks


def test_features_in_blocks_match_single_pass(trained, monkeypatch):
    import src.reconcile_ml.pipeline as pipeline
    data, settings, model, _ = trained
    start, end = date(2024, 11, 5), date(2024, 11, 7)
    whole, _ = _pipeline_hashes(data, settings, model, start, end)
    monkeypatch.setattr(pipeline, "FEATURE_BLOCK_PAIRS", 300)
    blocks, _ = _pipeline_hashes(data, settings, model, start, end)
    assert whole == blocks


def test_training_resumes_from_saved_dataset(trained):
    """Reprise sur le jeu enregistré : pas de rejeu, même modèle ; refus si les réglages changent."""
    _, settings, model, meta = trained
    interim, root = trained_dirs
    logs = []
    again = fit_ml(interim, root / "model_resumed", settings, RulesConfig(), log=logs.append, reuse_dataset=True)
    assert again["timings_s"].get("jeu repris") is True
    assert not any("pour construire le jeu" in line for line in logs)
    assert again["metrics"]["validation"] == meta["metrics"]["validation"]
    changed = settings.model_copy(deep=True)
    changed.reconcile_ml.training.payment_sample = 0.5 if settings.reconcile_ml.training.payment_sample != 0.5 else 0.9
    with pytest.raises(RuntimeError, match="ne correspond plus"):
        fit_ml(interim, root / "model_refused", changed, RulesConfig(), log=logs.append, reuse_dataset=True)


# --- Scores OOF de la pass 2 : walk-forward (par défaut) et folds aléatoires (version précédente) ----------------

def _wf_world():
    """4 mois de paires : 1 positif + 3 négatifs par paiement, 40 paiements par mois."""
    rows, imps = [], []
    for m, month in enumerate(["2024-01", "2024-02", "2024-03", "2024-04"]):
        for p in range(40):
            pid, day = f"P{m}{p:02d}", pd.Timestamp(f"{month}-10")
            for j in range(4):
                rows.append({"payment_id": pid, "invoice_id": f"I{m}{p:02d}{j}", "day": day, "label": int(j == 0),
                             "f": float(j == 0) + 0.01 * p})
            # Mois 2 : imputations prononcées très tard (après le 1er avril) → issue inconnue au 1er mars.
            when = pd.Timestamp("2024-04-15") if m == 1 else day + pd.Timedelta(days=2)
            imps.append({"payment_id": pid, "invoice_id": f"I{m}{p:02d}0", "updated_at": when})
    train = pd.DataFrame(rows)
    train[GROUP] = train["payment_id"]
    return train, pd.DataFrame(imps)


def test_walk_forward_uses_only_the_past_and_known_labels():
    from src.reconcile_ml.model import walk_forward_scores
    from src.settings import TrainingSettings
    train, imps = _wf_world()
    cfg = TrainingSettings(num_boost_round=10, min_data_in_leaf=5, walk_forward_min_positives=5)
    scores, log = walk_forward_scores(train, train[["f"]], 10, cfg, imps, purge_days=5, retention_days=60)
    by_date = {e["date"]: e for e in log}
    # Premier mois : pas de passé, pas de score.
    assert np.isnan(scores[(train["day"] < "2024-02-01").to_numpy()]).all()
    # Au 1er mars : janvier seul (160 paires) ; février (issue inconnue au 1er mars, < rétention) est écarté.
    assert by_date["2024-03-01"]["paires_apprises"] == 160 and by_date["2024-03-01"]["positifs"] == 40
    # Au 1er avril : février toujours inconnu (imputé le 15 avril) ; janvier et mars connus.
    assert by_date["2024-04-01"]["paires_apprises"] == 320
    assert not np.isnan(scores[(train["day"] >= "2024-02-01").to_numpy()]).any()


def test_walk_forward_step_scores_following_months_with_one_model():
    from src.reconcile_ml.model import walk_forward_scores
    from src.settings import TrainingSettings
    train, imps = _wf_world()
    cfg = TrainingSettings(num_boost_round=10, min_data_in_leaf=5, walk_forward_min_positives=5,
                           walk_forward_step_months=2)
    scores, log = walk_forward_scores(train, train[["f"]], 10, cfg, imps, purge_days=5, retention_days=60)
    assert [e["date"] for e in log] == ["2024-02-01", "2024-04-01"]
    assert log[0]["paires_scorees"] == 320                # février et mars, par le modèle du 1er février


def test_fit_keeps_random_folds_available():
    from src.settings import TrainingSettings
    train, imps = _wf_world()
    valid = train[train["day"] >= "2024-04-01"]
    cfg = TrainingSettings(num_boost_round=10, min_data_in_leaf=5, walk_forward_min_positives=5)
    old = PairModel.fit(train, valid, ["f"], cfg, oof="random_folds")
    new = PairModel.fit(train, valid, ["f"], cfg, oof="walk_forward", imputations=imps)
    assert old.meta["oof"]["schema"] == "random_folds" and old.pass2 is not None
    assert new.meta["oof"]["schema"] == "walk_forward" and new.meta["oof"]["paires_sans_score"] == 160
    with pytest.raises(ValueError):
        PairModel.fit(train, valid, ["f"], cfg, oof="walk_forward")          # dates des labels manquantes


def test_fit_ml_records_the_oof_scheme(trained):
    _, settings, _, meta = trained
    assert meta["oof"]["schema"] == settings.reconcile_ml.training.oof_scheme == "walk_forward"
