"""Recherches par clé vectorisées via pyarrow.

Sur les colonnes de chaînes pyarrow, `Series.isin` et `Series.map(Series)` de
pandas repassent par des objets Python : ~10 s sur 2 M de lignes, contre
< 1 s ici. À utiliser pour toute jointure par identifiant à grande échelle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc


def to_arrow(values: pd.Series | pd.Index | np.ndarray | list) -> pa.Array:
    if isinstance(values, (pd.Series, pd.Index)):
        values = values.array
    arr = pa.array(values, from_pandas=True)
    return arr.combine_chunks() if isinstance(arr, pa.ChunkedArray) else arr


def isin(values: pd.Series, reference: pd.Series) -> np.ndarray:
    """Masque booléen : valeur présente dans `reference` (NA → False)."""
    value_set = to_arrow(pd.Series(reference).dropna().drop_duplicates())
    mask = pc.is_in(to_arrow(values), value_set=value_set).fill_null(False)
    return mask.to_numpy(zero_copy_only=False)


def lookup(keys: pd.Series, index_keys: pd.Series, values: pd.Series) -> pd.Series:
    """Pour chaque clé, la valeur de `values` à la première position où `index_keys` vaut la clé.

    Clé absente ou manquante → NA. Résultat aligné sur l'index de `keys`.
    """
    first = ~pd.Series(index_keys).duplicated().to_numpy()
    idx_keys = pd.Series(index_keys)[first]
    vals = pd.Series(values)[first].reset_index(drop=True)
    pos = pc.index_in(to_arrow(keys), value_set=to_arrow(idx_keys)).fill_null(-1)
    pos = pos.to_numpy(zero_copy_only=False)
    found = pos >= 0
    out = vals.iloc[np.where(found, pos, 0)] if len(vals) else pd.Series([pd.NA] * len(pos))
    out = out.reset_index(drop=True).where(found)
    out.index = keys.index
    return out


# --- Colonnes de listes -----------------------------------------------------------------------------
# Les colonnes de listes (clés de référence, nombres des libellés) sont relues en Arrow : quelques
# octets par élément au lieu d'une liste Python et d'un objet chaîne par élément (÷ 8 environ). Les
# fonctions ci-dessous acceptent les deux formes (Arrow ou listes Python, comme dans les tests).

def _list_array(values: pd.Series) -> pa.Array | pa.ChunkedArray | None:
    """Tableau Arrow de listes si la colonne est en Arrow, sinon None.

    Les blocs ne sont pas fusionnés : `combine_chunks` recopierait toute la colonne à chaque appel, et
    `list_take` est appelé chaque jour du replay, warm-up compris (coût proportionnel à la taille des tables)."""
    if isinstance(values, pd.Series) and isinstance(values.dtype, pd.ArrowDtype):
        return values.array._pa_array
    return None


def list_lengths(values: pd.Series) -> np.ndarray:
    """Longueur de chaque liste (0 si nulle)."""
    arr = _list_array(values)
    if arr is not None:
        return pc.list_value_length(arr).fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
    return np.fromiter((0 if v is None or isinstance(v, float) else len(v) for v in values),
                       dtype=np.int64, count=len(values))


def flatten_lists(values: pd.Series) -> tuple[np.ndarray, np.ndarray] | None:
    """(longueurs, valeurs aplaties en objets) pour une colonne Arrow ; None si listes Python."""
    arr = _list_array(values)
    if arr is None:
        return None
    lengths = pc.list_value_length(arr).fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
    flat = pc.list_flatten(arr).to_numpy(zero_copy_only=False).astype(object)
    return lengths, flat


def list_take(values: pd.Series, positions: np.ndarray) -> list:
    """Listes Python des lignes demandées (pour un petit nombre de lignes)."""
    arr = _list_array(values)
    if arr is not None:
        return [v or [] for v in arr.take(pa.array(np.asarray(positions, dtype=np.int64))).to_pylist()]
    col = values.to_numpy()
    return [col[p] if isinstance(col[p], (list, tuple, np.ndarray)) else [] for p in positions]


def lists_source(values: pd.Series):
    """Colonne de listes prête pour des lectures répétées : tableau Arrow d'un seul bloc (fusionné une fois),
    ou la série elle-même si les listes sont en Python. À calculer une fois, pas à chaque lecture : un `take`
    sur un tableau en plusieurs blocs recopie la colonne entière."""
    arr = _list_array(values)
    if arr is None:
        return values
    return arr.combine_chunks() if isinstance(arr, pa.ChunkedArray) else arr


def lists_intersect(a, a_pos: np.ndarray, b, b_pos: np.ndarray) -> np.ndarray:
    """Pour chaque i : les listes a[a_pos[i]] et b[b_pos[i]] ont-elles un élément commun ?

    `a`, `b` : sorties de `lists_source`. En Arrow, sans aucune liste Python : on aplatit les seules lignes
    demandées et on joint (ligne, valeur)."""
    n = len(a_pos)
    if n == 0:
        return np.zeros(0, dtype=bool)
    if not (isinstance(a, pa.Array) and isinstance(b, pa.Array)):
        la = list_take(a, a_pos) if isinstance(a, pd.Series) else [v or [] for v in a.take(pa.array(a_pos)).to_pylist()]
        lb = list_take(b, b_pos) if isinstance(b, pd.Series) else [v or [] for v in b.take(pa.array(b_pos)).to_pylist()]
        return np.fromiter((bool(set(x) & set(y)) for x, y in zip(la, lb)), dtype=bool, count=n)

    def flat(arr: pa.Array, pos: np.ndarray) -> tuple[np.ndarray, pa.Array]:
        taken = arr.take(pa.array(np.asarray(pos, dtype=np.int64)))
        values = pc.fill_null(pc.list_flatten(taken).cast(pa.large_string()), "\x00")     # null = null, comme un set
        return pc.list_parent_indices(taken).to_numpy(zero_copy_only=False).astype(np.int64), values

    (la, va), (lb, vb) = flat(a, a_pos), flat(b, b_pos)
    out = np.zeros(n, dtype=bool)
    if len(la) and len(lb):
        # Valeurs codées en entiers (égalité exacte) ; une ligne est citée si un couple (ligne, code) est commun.
        codes = pc.dictionary_encode(pa.concat_arrays([va, vb])).indices.to_numpy(zero_copy_only=False)
        codes = codes.astype(np.int64)
        width = int(codes.max()) + 1
        left, right = la * width + codes[:len(la)], lb * width + codes[len(la):]
        out[la[np.isin(left, right)]] = True
    return out
