"""Tests de l'UI (streamlit AppTest), dans un environnement isolé : config et données temporaires."""

import shutil
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.config import REPO_ROOT
from src.api import Project
from src.settings import PathsSettings, RulesConfig, Settings, LoadSettings, load_rules, load_settings, save_rules, \
    save_settings

VIEWS = REPO_ROOT / "src" / "ui" / "views"


@pytest.fixture(scope="module")
def ui_env(tmp_path_factory):
    root = tmp_path_factory.mktemp("ui")
    settings = Settings(
        paths=PathsSettings(interim_dir=str(root / "interim"), reports_dir=str(root / "reports"),
                            synthetic_dir=str(root / "synthetic")),
        load=LoadSettings(workers=1),
    )
    save_settings(settings, root / "settings.yaml")
    save_rules(RulesConfig(), root / "rules.yaml")
    shutil.copy(REPO_ROOT / "config" / "schema.yaml", root / "schema.yaml")
    Project(settings_path=root / "settings.yaml", log=lambda m: None).load(n_payments=3000)
    return root


@pytest.fixture
def env(ui_env, monkeypatch):
    monkeypatch.setenv("RECON_SETTINGS", str(ui_env / "settings.yaml"))
    monkeypatch.setenv("RECON_SCHEMA", str(ui_env / "schema.yaml"))
    monkeypatch.setenv("RECON_RULES", str(ui_env / "rules.yaml"))
    return ui_env


def app(name: str) -> AppTest:
    at = AppTest.from_file(str(VIEWS / f"{name}.py"), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


@pytest.mark.parametrize("name", ["home", "config", "ingestion", "load", "split", "allocation", "rules", "ml",
                                  "evaluation", "analysis", "runtime"])
def test_every_page_renders(env, name):
    app(name)


def test_navigation_app_renders(env):
    at = AppTest.from_file(str(REPO_ROOT / "src" / "ui" / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception


def test_ingestion_lists_missing_mappings(env):
    at = app("ingestion")
    # Les expanders avec icône sont exposés comme éléments « status » par AppTest.
    assert any("point(s) bloquant" in e.label for e in at.status)


def test_ingestion_maps_validates_and_saves(env, tmp_path, monkeypatch):
    from src.config import read_yaml
    from src.load.schema_io import load_schema, save_schema

    cfg = read_yaml(REPO_ROOT / "config" / "schema.synthetic.yaml")
    cfg["base_dir"] = str(env / "synthetic" / "3000_seed42")
    save_schema(cfg, tmp_path / "schema.yaml")
    monkeypatch.setenv("RECON_SCHEMA", str(tmp_path / "schema.yaml"))
    at = app("ingestion")
    assert any("Mapping valide" in m.value for m in at.success)
    assert any("Configuration complète" in m.value for m in at.success)

    at.selectbox(key="ingest.payment.col.label").set_value(None).run()          # champ requis retiré
    assert any("Mapping incomplet" in m.value for m in at.error)
    at.selectbox(key="ingestion.table").set_value("debtor").run()               # la saisie de payment est gardée
    assert at.session_state["ingest.payment.col.label"] is None
    at.button(key="ingestion.save").click().run()
    assert load_schema(tmp_path / "schema.yaml")["tables"]["payment"]["columns"]["label"] is None

    at.selectbox(key="ingestion.table").set_value("payment").run()
    at.selectbox(key="ingest.payment.col.label").set_value("label").run()
    at.button(key="ingestion.save").click().run()
    at.button(key="ingestion.validate").click().run()
    assert not at.exception
    assert any("Prêt à charger" in m.value for m in at.success)


def test_split_preview_and_save(env):
    at = app("split")
    assert len(at.dataframe) == 1          # tableau des périodes
    at.number_input(key="split.purge_days").set_value(3)
    at.button(key="split.save").click().run()
    assert not at.exception
    assert load_settings(env / "settings.yaml").split.purge_days == 3


def test_split_reports_impossible_split(env):
    at = app("split")
    at.number_input(key="split.test_months").set_value(24).run()
    assert any("Découpage impossible" in e.value for e in at.error)


def test_allocation_warns_when_no_signal(env):
    at = app("allocation")
    for name in ("client_file", "reference", "iban", "name", "amount"):
        at.toggle(key=f"allocation.signals.{name}.enabled").set_value(False)
    at.run()
    assert any("Aucun signal actif" in e.value for e in at.error)


def test_rules_save_bumps_version(env):
    before = load_rules(env / "rules.yaml")
    at = app("rules")
    at.toggle(key="rules.R3_AMOUNT_UNIQUE.enabled").set_value(False).run()
    at.button(key="rules.save").click().run()
    assert not at.exception
    after = load_rules(env / "rules.yaml")
    assert after.version == before.version + 1
    assert not next(r for r in after.rules if r.id == "R3_AMOUNT_UNIQUE").enabled


def test_ml_toggle_saved(env):
    at = app("ml")
    at.toggle(key="reconcile_ml.second_pass").set_value(False)
    at.button(key="reconcile_ml.save").click().run()
    assert load_settings(env / "settings.yaml").reconcile_ml.second_pass is False


def test_load_page_runs_step_1(env):
    at = app("load")
    at.number_input(key="load.n_payments").set_value(2_000)
    at.button(key="run_load").click().run()
    assert not at.exception
    assert (Path(env) / "synthetic" / "2000_seed42" / "payment.csv").exists()


def test_split_page_runs_replay(env):
    at = app("split")
    at.button(key="replay.run").click().run()
    assert not at.exception
    assert (Path(env) / "reports" / "synthetic" / "replay_null_test.json").exists()
    assert any(m.label == "Jours" for m in at.metric)


def test_evaluation_page_runs_replay_and_evaluation(env):
    at = app("evaluation")
    at.selectbox(key="evaluation.matcher").set_value("null").run()
    at.button(key="evaluation.run").click().run()
    assert not at.exception
    assert (Path(env) / "reports" / "synthetic" / "evaluation_null_test.json").exists()
    assert any(m.label == "Taux d'automatisation effectif" for m in at.metric)


def test_evaluation_page_with_decisions_shows_curve(env):
    import json

    import pandas as pd

    project = Project(settings_path=env / "settings.yaml", rules_path=env / "rules.yaml", log=lambda m: None)
    project.replay("validation", "null")
    imp = pd.read_parquet(env / "interim" / "synthetic" / "imputation.parquet")
    meta = json.loads((env / "reports" / "synthetic" / "replay_null_validation.json").read_text(encoding="utf-8"))
    journal = pd.read_parquet(env / "interim" / "synthetic" / "journal.parquet")
    arrived = journal[(journal["event_type"] == "PAYMENT_RECEIVED") & (journal["ts"] >= pd.Timestamp(meta["start"]))
                      & (journal["ts"] <= pd.Timestamp(meta["end"]))]["entity_id"]
    sample = imp[imp["payment_id"].isin(arrived)].drop_duplicates("payment_id").head(20)
    decisions = pd.DataFrame({
        "payment_id": sample["payment_id"], "invoice_id": sample["invoice_id"], "amount": pd.array([1] * 20, "Int64"),
        "action": ["auto"] * 19 + ["review"], "step": ["rules"] * 10 + ["ml"] * 10, "rule_id": ["R2"] * 10 + [None] * 10,
        "rule_version": pd.array([1] * 20, "Int64"), "score": [1 - i / 100 for i in range(20)],
        "day": pd.Timestamp(meta["start"]),
    })
    decisions.to_parquet(env / "interim" / "synthetic" / "replay" / "replay_null_validation_decisions.parquet")
    project.evaluate("validation", "null")

    at = AppTest.from_file(str(VIEWS / "evaluation.py"), default_timeout=120)
    at.run()
    at.selectbox(key="evaluation.matcher").set_value("null")
    at.segmented_control(key="evaluation.period").set_value("validation").run()
    assert not at.exception
    assert not any("Aucune proposition" in i.value for i in at.info)


def test_allocation_page_runs_measure(env):
    at = app("allocation")
    at.button(key="allocation.run").click().run()
    assert not at.exception
    assert (Path(env) / "reports" / "synthetic" / "allocation_validation.json").exists()
    assert any(m.label == "Rappel (1er passage)" for m in at.metric)


def test_rules_page_runs_baseline(env):
    at = app("rules")
    at.button(key="rules.run").click().run()
    assert not at.exception
    assert (Path(env) / "reports" / "synthetic" / "evaluation_rules_validation.json").exists()
    assert any(m.label == "Taux d'automatisation" for m in at.metric)


def test_ml_oof_scheme_is_a_choice(env):
    at = app("ml")
    box = at.selectbox(key="reconcile_ml.training.oof_scheme")
    assert box.options == ["walk_forward", "random_folds"]
    box.set_value("random_folds")
    at.button(key="reconcile_ml.save").click().run()
    assert load_settings(env / "settings.yaml").reconcile_ml.training.oof_scheme == "random_folds"


def test_config_shows_changed_settings_and_environment(env):
    at = app("config")
    assert any("Réglages modifiés" in m.value for m in at.markdown)
    assert any(m.label == "CPU utilisables" for m in at.metric)


def test_diagnostic_toggles_render(env):
    for page, key in (("load", "load.volumetry"), ("split", "split.diagnostics"),
                      ("allocation", "allocation.client_files")):
        at = app(page)
        at.toggle(key=key).set_value(True).run()
        assert not at.exception, (page, at.exception)


def test_tracked_task_feeds_runtime_page(env):
    from src import profiling
    from src.api import run_task

    run_task("replay", {"settings_path": str(env / "settings.yaml"), "rules_path": str(env / "rules.yaml"),
                        "period": "test", "matcher": "null", "_monitor": True, "_profile": True})
    assert (env / "reports" / "memory.csv").exists()
    runs = profiling.saved_runs(env / "reports" / "profiles")
    assert runs["nom"].str.startswith("replay · synthetic · test").any()
    assert app("runtime").selectbox(key="runtime.profile") is not None

    # Session de suivi aux relevés explicites (un rejeu de cette taille peut durer moins d'un relevé).
    import time

    from src import memory
    monitor = memory.MemoryMonitor(env / "reports" / "memory.csv", interval=60, echo=lambda _: None).start()
    try:
        memory.mark_step("replay")
        memory.mark("règles · R2_REFERENCE_UNIQUE")
        monitor.sample()
        time.sleep(0.05)
        memory.mark_step(memory.IDLE)
        monitor.sample()
    finally:
        monitor.stop()
    at = app("runtime")
    assert any("Par sous-étape" in m.value for m in at.markdown)


def test_home_lists_every_step(env):
    at = app("home")
    labels = " ".join(m.value for m in at.markdown)
    assert all(step.title in labels for step in __import__("src.ui.common", fromlist=["STEPS"]).STEPS.values())


def test_portable_ui_is_generated_from_src():
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_single_ui", REPO_ROOT / "tools" / "build_single_ui.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    expected = (REPO_ROOT / "single" / "reconciliation_ui.py").read_text(encoding="utf-8")
    assert builder.build() == expected, "single/reconciliation_ui.py périmé : lancer python tools/build_single_ui.py"


def test_analysis_page_shows_errors_residual_segments(env):
    project = Project(settings_path=env / "settings.yaml", rules_path=env / "rules.yaml", log=lambda m: None)
    project.backtest("validation", "rules")
    at = AppTest.from_file(str(VIEWS / "analysis.py"), default_timeout=120)
    at.run()
    at.selectbox(key="analysis.matcher").set_value("rules")
    at.segmented_control(key="analysis.period").set_value("validation").run()
    assert not at.exception
    assert any(m.label == "Paiements du périmètre" for m in at.metric)
    at.selectbox(key="analysis.cross").set_value("Canal").run()
    assert not at.exception
