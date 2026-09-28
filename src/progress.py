"""Suivi de progression des tâches longues : rejeux jour par jour, apprentissage LightGBM.

    from src import progress
    progress.use_notebook()        # barres mises à jour en place dans le notebook
    progress.use_log(print)        # ou : une ligne de journal toutes les 30 s (console, interface)

Chaque tâche affiche : avancement (jours, itérations), temps écoulé, temps restant estimé et un détail
(jour et taille du lot, perte de validation LightGBM...). Sans afficheur choisi, `Project` journalise
la progression par son `log`. Les tâches s'annoncent avec `progress.task(...)` ; sans afficheur actif,
elles ne coûtent rien.
"""

from __future__ import annotations

import html
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

_REPORTER: Reporter | None = None


def _duration(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds} s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes} min {seconds:02d} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min"


class Task:
    """Une tâche mesurable : `total` unités (jours, itérations) ; `advance` à chaque unité faite."""

    def __init__(self, label: str, total: int, unit: str):
        self.label, self.total, self.unit = label, max(int(total), 0), unit
        self.done = 0
        self.detail = ""
        self.started = time.time()
        self.finished = False

    @property
    def elapsed(self) -> float:
        return time.time() - self.started

    @property
    def remaining(self) -> float | None:
        if self.done == 0 or self.total == 0 or self.finished:
            return None
        return self.elapsed / self.done * max(self.total - self.done, 0)

    @property
    def fraction(self) -> float:
        return 1.0 if self.finished else (min(self.done / self.total, 1.0) if self.total else 0.0)

    def advance(self, n: int = 1, detail: str | None = None) -> None:
        self.done += n
        if detail is not None:
            self.detail = detail
        if _REPORTER is not None:
            _REPORTER.update(self)

    def text(self) -> str:
        head = f"{self.label} : {self.done}/{self.total} {self.unit} ({self.fraction * 100:.0f} %)"
        timing = f"{_duration(self.elapsed)} écoulées"
        if self.finished:
            timing = f"terminé en {_duration(self.elapsed)}"
        elif self.remaining is not None:
            timing += f", reste ~{_duration(self.remaining)}"
        return " · ".join(p for p in (head, timing, self.detail) if p)


@contextmanager
def task(label: str, total: int, unit: str = "") -> Iterator[Task]:
    t = Task(label, total, unit)
    if _REPORTER is not None:
        _REPORTER.start(t)
    try:
        yield t
    finally:
        t.finished = True
        if _REPORTER is not None:
            _REPORTER.finish(t)


# --- Afficheurs -----------------------------------------------------------------------------------------

class Reporter:
    def start(self, t: Task) -> None: ...
    def update(self, t: Task) -> None: ...
    def finish(self, t: Task) -> None: ...


class LogReporter(Reporter):
    """Une ligne au début, puis toutes les `every` secondes, puis à la fin."""

    def __init__(self, log: Callable[[str], None], every: float = 30.0):
        self.log, self.every = log, every
        self._last: dict[int, float] = {}

    def start(self, t: Task) -> None:
        self._last[id(t)] = time.time()
        self.log(f"▶ {t.label} ({t.total} {t.unit})")

    def update(self, t: Task) -> None:
        now = time.time()
        if now - self._last.get(id(t), 0) >= self.every:
            self._last[id(t)] = now
            self.log(f"  {t.text()}")

    def finish(self, t: Task) -> None:
        self._last.pop(id(t), None)
        self.log(f"✓ {t.text()}")


class NotebookReporter(Reporter):
    """Une barre par tâche, mise à jour en place dans la sortie de la cellule (au plus 2 fois par seconde)."""

    def __init__(self, every: float = 0.5):
        self.every = every
        self._handles: dict[int, tuple[object, float]] = {}

    @staticmethod
    def _html(t: Task):
        from IPython.display import HTML
        color = "#2e7d32" if t.finished else "#1565c0"
        return HTML(
            f'<div style="font-family:monospace;font-size:12px;margin:2px 0">'
            f'<div style="width:360px;height:8px;background:#e0e0e0;border-radius:4px;display:inline-block;'
            f'vertical-align:middle;margin-right:8px"><div style="width:{t.fraction * 100:.1f}%;height:8px;'
            f'background:{color};border-radius:4px"></div></div>{html.escape(t.text())}</div>')

    def start(self, t: Task) -> None:
        from IPython.display import display
        self._handles[id(t)] = (display(self._html(t), display_id=True), time.time())

    def update(self, t: Task) -> None:
        handle, last = self._handles.get(id(t), (None, 0.0))
        if handle is not None and time.time() - last >= self.every:
            handle.update(self._html(t))
            self._handles[id(t)] = (handle, time.time())

    def finish(self, t: Task) -> None:
        handle, _ = self._handles.pop(id(t), (None, 0.0))
        if handle is not None:
            handle.update(self._html(t))


def use_notebook() -> None:
    """Barres de progression dans le notebook (Jupyter)."""
    global _REPORTER
    _REPORTER = NotebookReporter()


def use_log(log: Callable[[str], None] = print, every: float = 30.0) -> None:
    """Progression écrite dans un journal (console, interface), une ligne toutes les `every` secondes."""
    global _REPORTER
    _REPORTER = LogReporter(log, every)


def disable() -> None:
    global _REPORTER
    _REPORTER = None


def active() -> bool:
    return _REPORTER is not None


# --- LightGBM -------------------------------------------------------------------------------------------

def lightgbm_callback(t: Task):
    """Callback LightGBM : une unité par itération, perte de validation en détail."""
    def callback(env) -> None:
        results = env.evaluation_result_list or []
        detail = " · ".join(f"{name} {value:.5f}" for _, name, value, *_ in results)
        t.advance(1, f"itération {env.iteration + 1}" + (f" · {detail}" if detail else ""))
    callback.order = 5
    return callback
