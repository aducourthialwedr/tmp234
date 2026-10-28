"""Profil par fonction, par échantillonnage : d'où vient le temps d'un run, sans le ralentir.

    from src import profiling
    prof = profiling.Profiler("reports/profiles", enabled=True)
    with prof("règles · test"):
        project.backtest("test", matcher="rules")
    prof.summary("règles · test")          # temps par fonction du projet (bibliothèques comprises)

Pourquoi l'échantillonnage : un profileur déterministe (cProfile) instrumente chaque appel et ralentit un code
Python d'un facteur 1,5 à 3, ce qui fausse aussi la répartition. Un échantillonneur relève la pile d'appels à
intervalle régulier ; le temps d'une fonction est estimé par la part des relevés où elle figure.

Deux moteurs, même sortie : des piles « repliées » (`f1;f2;f3 n`), lisibles par speedscope.app ou flamegraph.pl.
- **py-spy** (`pip install py-spy`), choisi s'il est disponible : il lit la mémoire du processus depuis
  l'extérieur, sans le suspendre (`--nonblocking`) ; impact négligeable. Dans un conteneur, il faut la capacité
  `SYS_PTRACE` (celle qu'utilise `py-spy dump`).
- **fil interne** (repli, rien à installer) : un fil Python relève la pile du fil principal `rate` fois par seconde
  (10 par défaut), soit environ 0,1 ms de travail par relevé ; impact inférieur à 1 %.

Une fonction est rangée par sa ligne de définition, pas par la ligne en cours : ses relevés se cumulent.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

_FRAME = re.compile(r"^(?P<func>.*) \((?P<file>.*):(?P<line>\d+)\)$")
_FRAME_NO_LINE = re.compile(r"^(?P<func>.*) \((?P<file>[^()]*)\)$")       # py-spy --function, selon la version


def _match(frame: str):
    return _FRAME.match(frame) or _FRAME_NO_LINE.match(frame)


def _slug(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", name, flags=re.UNICODE).strip("_") or "profil"


class _ThreadSampler:
    """Relève la pile du fil `thread_id`, `rate` fois par seconde, dans un fil démon."""

    def __init__(self, rate: float, thread_id: int):
        self.interval, self.thread_id = 1.0 / rate, thread_id
        self.counts: Counter[str] = Counter()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="profil", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            frame = sys._current_frames().get(self.thread_id)
            stack = []
            while frame is not None:
                code = frame.f_code
                stack.append(f"{code.co_name} ({code.co_filename}:{code.co_firstlineno})")
                frame = frame.f_back
            if stack:
                self.counts[";".join(reversed(stack))] += 1

    def stop(self, output: Path) -> None:
        self._stop.set()
        self._thread.join()
        output.write_text("".join(f"{stack} {n}\n" for stack, n in self.counts.items()), encoding="utf-8")


class _PySpy:
    """`py-spy record` sur le processus courant, au format replié, arrêté par un signal d'interruption."""

    def __init__(self, executable: str, rate: float, output: Path, subprocesses: bool):
        cmd = [executable, "record", "--pid", str(os.getpid()), "--rate", str(int(rate)), "--format", "raw",
               "--function", "--nonblocking", "--output", str(output)]
        if subprocesses:
            cmd.append("--subprocesses")
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                                     creationflags=flags)
        time.sleep(0.5)
        if self.proc.poll() is not None:                       # échec d'attachement (droits ptrace…)
            raise RuntimeError((self.proc.stderr.read() or "py-spy s'est arrêté").strip())

    def stop(self, output: Path) -> None:
        self.proc.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
        try:
            self.proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def find_pyspy() -> str | None:
    """Exécutable py-spy : PATH, puis dossiers de scripts de cet interpréteur (pip install --user compris)."""
    import sysconfig
    found = shutil.which("py-spy")
    if found:
        return found
    names = ("py-spy.exe", "py-spy") if os.name == "nt" else ("py-spy",)
    folders = {Path(sys.executable).parent, Path(sysconfig.get_path("scripts"))}
    scheme = "nt_user" if os.name == "nt" else "posix_user"
    try:
        folders.add(Path(sysconfig.get_path("scripts", scheme)))
    except KeyError:
        pass
    for folder in folders:
        for name in names:
            if (folder / name).exists():
                return str(folder / name)
    return None


class Profiler:
    """Profils nommés, un fichier de piles repliées par mesure ; `enabled=False` : aucun effet, aucun coût."""

    def __init__(self, folder: str | Path = "reports/profiles", enabled: bool = True, rate: float = 10,
                 engine: str = "auto", subprocesses: bool = False, log=print):
        self.folder, self.enabled, self.rate = Path(folder), enabled, rate
        self.engine, self.subprocesses, self.log = engine, subprocesses, log
        self.runs: dict[str, dict] = {}

    def _start(self, output: Path):
        exe = find_pyspy() if self.engine in ("auto", "py-spy") else None
        if exe:
            try:
                return "py-spy", _PySpy(exe, self.rate, output, self.subprocesses)
            except RuntimeError as exc:
                if self.engine == "py-spy":
                    raise
                self.log(f"  profil : py-spy indisponible ({exc}) ; repli sur le fil interne")
        sampler = _ThreadSampler(self.rate, threading.main_thread().ident)
        sampler.start()
        return "fil interne", sampler

    @contextmanager
    def __call__(self, name: str):
        if not self.enabled:
            yield
            return
        self.folder.mkdir(parents=True, exist_ok=True)
        output = self.folder / f"{_slug(name)}.txt"
        engine, sampler = self._start(output)
        start = time.perf_counter()
        try:
            yield
        finally:
            sampler.stop(output)
            self.runs[name] = {"fichier": output, "secondes": time.perf_counter() - start, "moteur": engine,
                               "rate": self.rate}
            self.log(f"  profil « {name} » : {self.runs[name]['secondes']:.0f} s, {engine} → {output}")

    def summary(self, name: str | None = None, top: int = 25) -> pd.DataFrame:
        """Tableau du profil `name` (le dernier si absent) : voir `stack_summary`."""
        if not self.runs:
            raise ValueError("aucun profil : activer le profileur (enabled=True) puis relancer l'étape")
        run = self.runs[name] if name else list(self.runs.values())[-1]
        # Le fil interne peut relever moins souvent que prévu (GIL, horloge) : ses durées sont calées sur le temps
        # écoulé. py-spy respecte sa fréquence et n'enregistre pas l'inactivité : ses relevés valent 1 / rate.
        seconds = run["secondes"] if run["moteur"] == "fil interne" else None
        return stack_summary(run["fichier"], rate=run["rate"], top=top, seconds=seconds)


def _parse(path: Path) -> list[tuple[list[str], int]]:
    stacks = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        stack, _, count = line.rpartition(" ")
        if stack and count.isdigit():
            stacks.append((stack.split(";"), int(count)))
    return stacks


def _is_project(frame: str) -> bool:
    m = _match(frame)
    file = (m["file"] if m else frame).replace("\\", "/")
    return "/src/" in f"/{file}" and "site-packages" not in file or file.endswith("reconciliation.py")


def _label(frame: str) -> tuple[str, str]:
    m = _match(frame)
    if not m:
        return frame, ""
    file = m["file"].replace("\\", "/")
    short = file.split("/src/")[-1] if "/src/" in file else file.split("/")[-1]
    line = m.groupdict().get("line")
    return m["func"], f"{short}:{line}" if line else short


def stack_summary(path: str | Path, rate: float = 10, top: int = 25, seconds: float | None = None) -> pd.DataFrame:
    """Temps par fonction du projet, depuis un fichier de piles repliées.

    - `inclus` : temps où la fonction est dans la pile (elle et tout ce qu'elle appelle) ;
    - `propre` : temps où c'est la fonction du projet la plus profonde de la pile, bibliothèques comprises
      (pandas, numpy, LightGBM…) : c'est là qu'il faut optimiser.
    Les relevés hors code du projet (attente, import) sont comptés dans la ligne « hors projet ».
    `seconds` : durée réelle de la mesure ; si elle est donnée, les relevés sont répartis sur cette durée au lieu
    de valoir chacun 1 / `rate`."""
    stacks = _parse(Path(path))
    total = sum(n for _, n in stacks) or 1
    if seconds is not None:
        rate = total / seconds if seconds > 0 else rate
    inclusive: Counter[str] = Counter()
    own: Counter[str] = Counter()
    for frames, n in stacks:
        project = [f for f in frames if _is_project(f)]
        for f in set(project):
            inclusive[f] += n
        own[project[-1] if project else "hors projet"] += n
    rows = []
    for frame in set(inclusive) | set(own):
        func, where = _label(frame) if frame != "hors projet" else ("hors projet", "")
        rows.append({"fonction": func, "où": where, "propre (s)": own[frame] / rate,
                     "inclus (s)": inclusive[frame] / rate, "part propre": own[frame] / total})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("propre (s)", ascending=False).head(top).reset_index(drop=True)
