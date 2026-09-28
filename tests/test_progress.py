"""Suivi de progression : rejeu jour par jour et itérations LightGBM."""

from datetime import date

import numpy as np
import pandas as pd

from src import progress
from src.reconcile_ml.model import _train
from src.settings import TrainingSettings
from src.timeline.loop import NullMatcher, run_replay
from tests.test_timeline import base, enriched, invoice, ledger


def test_replay_and_training_report_progress():
    lines = []
    progress.use_log(lines.append, every=0)
    try:
        state = ledger(enriched(**base(invoice=[invoice("I1")], payment=[], imputation=[])))
        run_replay(state, NullMatcher(), date(2024, 3, 1), date(2024, 3, 3), label="rejeu test")
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"a": rng.random(200), "b": rng.random(200)})
        _train(X, (X["a"] > 0.5).to_numpy().astype(int), None, None,
               TrainingSettings(num_boost_round=10, min_data_in_leaf=5), "modèle test")
    finally:
        progress.disable()
    assert lines[0] == "▶ rejeu test (3 jours)"
    assert any(l.startswith("  rejeu test : 2/3 jours") for l in lines)
    assert any(l.startswith("✓ rejeu test : 3/3 jours (100 %)") for l in lines)
    assert any("modèle test (200 paires) : 10/10 itérations" in l and l.startswith("✓") for l in lines)


def test_tasks_are_silent_without_reporter():
    progress.disable()
    with progress.task("x", 2, "jours") as t:
        t.advance()
    assert t.finished and t.done == 1
