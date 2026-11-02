"""Profil de temps (relevés du suivi mémoire) et profil par fonction (échantillonnage)."""

import time

from src import memory, profiling
from src.memory import MemoryMonitor


def test_time_profile_counts_phases_and_excludes_idle(tmp_path):
    path = tmp_path / "mem.csv"
    m = MemoryMonitor(path, interval=10, echo=lambda _: None).start()
    try:
        memory.mark_step("rejeu test (rules)")
        for block in ("1/2", "2/2"):
            memory.mark(f"allocation · nom · bloc {block}")
            m.sample()
            time.sleep(0.05)
        memory.mark("règles · R4_EXACT_SUM")
        m.sample()
        time.sleep(0.05)
        memory.mark_step(memory.IDLE)
        m.sample()
        time.sleep(0.2)
        m.sample()
    finally:
        m.stop()
    prof = memory.time_profile(path)
    phases = dict(zip(prof["sous-étape"], prof["minutes"]))
    assert set(prof["étape"]) == {"rejeu test (rules)"}                       # inactivité exclue
    assert "allocation · nom · bloc n/n" in phases and "règles · R4_EXACT_SUM" in phases
    assert phases["allocation · nom · bloc n/n"] * 60 >= 0.09                  # deux blocs cumulés
    assert abs(prof["part du total"].sum() - 1) < 1e-9


def _busy(seconds: float) -> int:
    end, n = time.perf_counter() + seconds, 0
    while time.perf_counter() < end:
        n += sum(range(200))
    return n


def test_thread_sampler_attributes_time_to_project_functions(tmp_path):
    prof = profiling.Profiler(tmp_path, engine="thread", rate=50, log=lambda _: None)
    with prof("essai"):
        _busy(0.6)
    run = prof.runs["essai"]
    assert run["moteur"] == "fil interne" and run["fichier"].exists()
    table = prof.summary("essai")
    assert {"fonction", "où", "propre (s)", "inclus (s)", "part propre"} <= set(table.columns)
    assert 0.4 < table["propre (s)"].sum() < 1.2          # durées calées sur le temps écoulé (≈ 0,6 s)


def test_stack_summary_separates_own_and_inclusive_time(tmp_path):
    path = tmp_path / "piles.txt"
    path.write_text("main (/repo/src/api.py:10);run (/repo/src/timeline/loop.py:5);merge (/x/site-packages/pandas/merge.py:1) 30\n"
                    "main (/repo/src/api.py:10);run (/repo/src/timeline/loop.py:5) 10\n"
                    "main (/repo/src/api.py:10) 10\n", encoding="utf-8")
    table = profiling.stack_summary(path, rate=10).set_index("fonction")
    assert table.loc["run", "propre (s)"] == 4.0 and table.loc["run", "inclus (s)"] == 4.0      # pandas compris
    assert table.loc["main", "propre (s)"] == 1.0 and table.loc["main", "inclus (s)"] == 5.0
    assert table.loc["run", "où"] == "timeline/loop.py:5"


def test_disabled_profiler_has_no_effect(tmp_path):
    prof = profiling.Profiler(tmp_path / "p", enabled=False)
    with prof("rien"):
        pass
    assert prof.runs == {} and not (tmp_path / "p").exists()


def test_saved_runs_are_read_back(tmp_path):
    prof = profiling.Profiler(tmp_path / "profiles", engine="thread", rate=50, log=lambda _: None)
    with prof("règles · test"):
        _busy(0.3)
    runs = profiling.saved_runs(tmp_path / "profiles")
    assert runs["nom"].tolist() == ["règles · test"] and runs.loc[0, "moteur"] == "fil interne"
    table = profiling.saved_summary(runs.iloc[0].to_dict())
    assert 0.15 < table["propre (s)"].sum() < 0.8


def test_sessions_list_each_monitor_start(tmp_path):
    path = tmp_path / "mem.csv"
    for step in ("load", "replay"):
        m = MemoryMonitor(path, interval=10, echo=lambda _: None).start()
        try:
            memory.mark_step(step)
            m.sample()
            memory.mark_step(memory.IDLE)
            m.sample()
        finally:
            m.stop()
    sessions = memory.sessions(path)
    assert sessions["étapes"].tolist() == ["load", "replay"]
    assert sessions["début"].str.len().gt(0).all()
