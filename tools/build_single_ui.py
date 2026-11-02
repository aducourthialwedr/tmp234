"""Construit single/reconciliation_ui.py, l'interface en un fichier, depuis src/ui (outil de développement).

- src/ui/common.py → section « Éléments communs » ;
- chaque vue src/ui/views/<nom>.py → fonction `page_<nom>()` ;
- src/ui/app.py → `main()`, les pages passées comme fonctions ;
- imports `from src...` regroupés en un seul `from reconciliation import (...)`, y compris ceux placés dans une
  fonction ; imports externes regroupés en tête ;
- code propre au jeu synthétique retiré, comme dans single/reconciliation.py : blocs « <synthétique> », lignes
  « synthétique seulement » ; « portable: … » remplace la ligne marquée.

    python tools/build_single_ui.py
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "src" / "ui"
TARGET = ROOT / "single" / "reconciliation_ui.py"
PAGES = ["home", "config", "ingestion", "load", "split", "allocation", "rules", "ml", "evaluation", "analysis",
         "runtime"]
BAR = "# " + "═" * 100
HEADER = '''"""Interface de pilotage — version en un fichier.

    streamlit run reconciliation_ui.py

Exige reconciliation.py dans le même dossier. Écoute locale,
télémétrie désactivée par .streamlit/config.toml (créé par reconciliation.py).

Généré par tools/build_single_ui.py depuis src/ui.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
'''
HEADER_IMPORTS = {"import sys", "from pathlib import Path"}
# Substitutions propres à la navigation : une page est une fonction, plus un fichier de src/ui/views.
APP_SUBSTITUTIONS = [
    (r"def page\(key: str, file: str, ", "def page(key: str, fn, "),
    (r"st\.Page\(VIEWS / file, ", "st.Page(fn, "),
    (r'st\.Page\(VIEWS / "(\w+)\.py", ', r"st.Page(page_\1, "),
    (r'page\("(\w+)", "(\w+)\.py"\)', r'page("\1", page_\2)'),
]


def portable(lines: list[str]) -> list[str]:
    out, skip = [], False
    for line in lines:
        if line.strip() == "# <synthétique>":
            skip = True
            continue
        if line.strip() == "# </synthétique>":
            skip = False
            continue
        if skip or "# synthétique seulement" in line:
            continue
        if "# portable: " in line:
            indent = line[:len(line) - len(line.lstrip())]
            line = indent + line.split("# portable: ", 1)[1].rstrip()
        out.append(line)
    return out


class Module:
    """Code d'un fichier de src/ui sans ses imports ni sa docstring ; imports collectés à part."""

    def __init__(self, path: Path, drop_statement=lambda source: False):
        text = "\n".join(portable(path.read_text(encoding="utf-8").splitlines())) + "\n"
        tree = ast.parse(text)
        self.external: set[str] = set()
        self.names: set[str] = set()
        drop: set[int] = set()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            span = range(node.lineno, node.end_lineno + 1)
            module = node.module or "" if isinstance(node, ast.ImportFrom) else ""
            if isinstance(node, ast.ImportFrom) and module.split(".")[0] == "src":
                if not module.startswith("src.ui"):
                    self.names |= {a.name if a.asname is None else f"{a.name} as {a.asname}" for a in node.names}
                drop.update(span)
            elif node in tree.body:
                if isinstance(node, ast.Import):
                    self.external |= {("", ast.unparse(ast.Import(names=[a]))) for a in node.names}
                elif module != "__future__":
                    self.external |= {(module, a.name if a.asname is None else f"{a.name} as {a.asname}")
                                      for a in node.names}
                drop.update(span)
        for i, node in enumerate(tree.body):
            docstring = i == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
            if docstring or drop_statement(ast.get_source_segment(text, node) or ""):
                drop.update(range(node.lineno, node.end_lineno + 1))
        self.lines = [line for n, line in enumerate(text.splitlines(), 1) if n not in drop]
        # Lignes intérieures des chaînes sur plusieurs lignes : à ne pas réindenter.
        code = "\n".join(self.lines) + "\n"
        self.inside_strings: set[int] = set()
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            if tok.type == tokenize.STRING and tok.end[0] > tok.start[0]:
                self.inside_strings.update(range(tok.start[0] + 1, tok.end[0] + 1))

    def body(self, indent: str = "") -> str:
        out = [line if not line.strip() or n in self.inside_strings else indent + line
               for n, line in enumerate(self.lines, 1)]
        text = "\n".join(line.rstrip() if not line.strip() else line for line in out).strip("\n")
        return re.sub(r"\n{3,}" if indent else r"\n{4,}", "\n\n" if indent else "\n\n\n", text)


def banner(title: str) -> str:
    return f"\n\n{BAR}\n# {title}\n{BAR}\n\n"


def build() -> str:
    common = Module(UI / "common.py")
    views = {name: Module(UI / "views" / f"{name}.py") for name in PAGES}
    app = Module(UI / "app.py", drop_statement=lambda s: "_ROOT" in s or s.startswith("VIEWS ="))
    modules = [common, *views.values(), app]

    names = sorted(set().union(*(m.names for m in modules)))
    external = set().union(*(m.external for m in modules))
    plain = sorted(line for module, line in external if not module)
    grouped: dict[str, set[str]] = {}
    for module, name in external:
        if module:
            grouped.setdefault(module, set()).add(name)
    froms = [f"from {module} import {', '.join(sorted(grouped[module]))}" for module in sorted(grouped)]
    imports = [line for line in plain + froms if line not in HEADER_IMPORTS]
    parts = [HEADER, "\n", "\n".join(imports), "\n\nfrom reconciliation import (\n"
             + "".join(f"    {n},\n" for n in names) + ")\n"]

    common_body = common.body()
    target = "from src.api import run_task"
    assert target in common_body, "run_task : ligne de commande introuvable dans common.py"
    parts += [banner("Éléments communs"), common_body.replace(target, "from reconciliation import run_task")]

    parts.append(banner("Pages"))
    for name, view in views.items():
        parts.append(f"\n\ndef page_{name}() -> None:\n{view.body('    ')}\n")

    main = app.body("    ")
    for pattern, repl in APP_SUBSTITUTIONS:
        main, n = re.subn(pattern, repl, main)
        assert n, f"navigation : motif absent de app.py : {pattern}"
    parts += [banner("Navigation"), f"\ndef main() -> None:\n{main}\n\n\nif __name__ == \"__main__\":\n    main()\n"]
    return "".join(parts)


def main() -> Path:
    text = build()
    compile(text, str(TARGET), "exec")
    TARGET.write_text(text, encoding="utf-8")
    return TARGET


if __name__ == "__main__":
    path = main()
    print(f"{path} — {len(path.read_text(encoding='utf-8').splitlines())} lignes")
