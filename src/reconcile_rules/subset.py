"""Recherche bornée de sous-ensembles à somme exacte (règle R4, ensembles de l'étape 5).

Parcours en profondeur sur les montants triés, avec élagage :
- la somme courante dépasse la cible → on coupe (montants croissants) ;
- même en ajoutant les plus grands montants restants, la cible est hors d'atteinte → on coupe ;
- profondeur > `max_size` → on coupe ;
- budget de nœuds épuisé → résultat « indéterminé ».

On s'arrête dès la deuxième solution : l'unicité est tout ce qui compte pour valider.

Le parcours ne dépend que des montants triés et des paramètres : il est mémorisé sur ces valeurs. Un paiement
du reliquat, retesté chaque jour avec les mêmes factures ouvertes, n'est ainsi parcouru qu'une fois.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

UNIQUE, NONE, AMBIGUOUS, BUDGET = "unique", "none", "ambiguous", "budget"
CACHE_SIZE = 200_000          # parcours mémorisés (quelques centaines d'octets chacun)


@dataclass(frozen=True)
class SubsetResult:
    status: str
    indices: tuple[int, ...] = ()     # indices dans le tableau d'entrée (si unique)
    nodes: int = 0


def _sorted(amounts: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    values = np.asarray(amounts, dtype=np.int64)
    order = np.argsort(values, kind="stable")
    return order, tuple(values[order].tolist())


@lru_cache(maxsize=CACHE_SIZE)
def _exact_sorted(a: tuple[int, ...], target: int, max_size: int, min_size: int,
                  node_budget: int) -> tuple[str, tuple[int, ...], int]:
    """Parcours sur les montants triés `a` : (statut, solution en positions de `a`, nœuds)."""
    n = len(a)
    # Borne supérieure = somme des k plus grands à partir de i, soit les k derniers éléments (tableau trié).
    top = [0] * (max_size + 1)
    for k in range(1, max_size + 1):
        top[k] = sum(a[-k:]) if k <= n else sum(a)
    # Bornes indexées par la profondeur : top[min(max_size - profondeur - 1, n)].
    bound = [top[min(max_size - d - 1, n)] if d + 1 < max_size else 0 for d in range(max_size)]
    solutions: list[tuple[int, ...]] = []
    nodes = 0
    # Parcours en profondeur itératif (même ordre de visite et même décompte de nœuds que la version
    # récursive) : `chosen` = indices retenus, `frames` = (index, total) du parent à chaque profondeur.
    chosen: list[int] = []
    frames: list[tuple[int, int]] = []
    i, total, completed = 0, 0, True
    while True:
        if i >= n:                                      # fin de boucle à cette profondeur : retour au parent
            if not frames:
                break
            i, total = frames.pop()
            chosen.pop()
            i += 1
            continue
        nodes += 1
        if nodes > node_budget:
            completed = False
            break
        s = total + a[i]
        if s > target:                                  # montants croissants : rien au-delà
            i = n
            continue
        depth = len(frames)
        chosen.append(i)
        if s == target and depth + 1 >= min_size:
            solutions.append(tuple(chosen))
            if len(solutions) > 1:
                completed = False
                break
        elif depth + 1 < max_size and s + bound[depth] >= target:
            frames.append((i, total))                   # descente
            i, total = i + 1, s
            continue
        chosen.pop()
        i += 1
    if len(solutions) > 1:
        return AMBIGUOUS, (), nodes
    if not completed:
        return BUDGET, (), nodes
    if not solutions:
        return NONE, (), nodes
    return UNIQUE, solutions[0], nodes


def exact_subset(amounts: np.ndarray, target: int, max_size: int = 5, min_size: int = 1,
                 node_budget: int = 100_000) -> SubsetResult:
    """Sous-ensemble unique de `amounts` (entiers > 0) sommant exactement à `target`."""
    order, a = _sorted(amounts)
    if len(a) == 0 or target <= 0:
        return SubsetResult(NONE)
    status, solution, nodes = _exact_sorted(a, int(target), int(max_size), int(min_size), int(node_budget))
    if status != UNIQUE:
        return SubsetResult(status, nodes=nodes)
    return SubsetResult(UNIQUE, tuple(sorted(int(order[i]) for i in solution)), nodes)


@lru_cache(maxsize=CACHE_SIZE)
def _near_sorted(a: tuple[int, ...], target: int, tolerance: int, max_size: int, min_size: int, node_budget: int,
                 max_solutions: int) -> tuple[tuple[tuple[int, ...], ...], bool]:
    """Parcours sur les montants triés `a` : (solutions en positions de `a`, dans l'ordre du parcours ; complet ?)."""
    n = len(a)
    lo, hi = target - tolerance, target + tolerance
    top = [sum(a[-k:]) if 0 < k <= n else sum(a) for k in range(max_size + 1)]
    bound = [top[min(max_size - d - 1, n)] if d + 1 < max_size else 0 for d in range(max_size)]
    solutions: list[tuple[int, ...]] = []
    nodes = 0
    # Même parcours que `_exact_sorted` (itératif, ordre et nœuds identiques à la version récursive), mais une
    # solution n'arrête pas la descente : un ensemble plus grand peut aussi tomber dans la tolérance.
    chosen: list[int] = []
    frames: list[tuple[int, int]] = []
    i, total, complete = 0, 0, True
    while True:
        if i >= n:
            if not frames:
                break
            i, total = frames.pop()
            chosen.pop()
            i += 1
            continue
        nodes += 1
        if nodes > node_budget:
            complete = False
            break
        s = total + a[i]
        if s > hi:
            i = n
            continue
        depth = len(frames)
        chosen.append(i)
        if s >= lo and depth + 1 >= min_size:
            solutions.append(tuple(chosen))
            if len(solutions) >= max_solutions:
                complete = False
                break
        if depth + 1 < max_size and s + bound[depth] >= lo:
            frames.append((i, total))
            i, total = i + 1, s
            continue
        chosen.pop()
        i += 1
    return tuple(solutions), complete


def near_subsets(amounts: np.ndarray, target: int, tolerance: int, max_size: int = 5, min_size: int = 2,
                 node_budget: int = 100_000, max_solutions: int = 20) -> tuple[list[tuple[int, ...]], bool]:
    """Sous-ensembles dont la somme est à `tolerance` près de `target` (ensembles de l'étape 5).

    Retourne (solutions en indices de `amounts`, parcours complet ?). Solutions triées par
    cardinalité croissante : la parcimonie évite les combinaisons fortuites.
    """
    order, a = _sorted(amounts)
    found, complete = _near_sorted(a, int(target), int(tolerance), int(max_size), int(min_size), int(node_budget),
                                   int(max_solutions))
    solutions = [tuple(sorted(int(order[j]) for j in sol)) for sol in found]
    solutions.sort(key=len)
    return solutions, complete
