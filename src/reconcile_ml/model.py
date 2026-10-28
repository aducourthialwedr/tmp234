"""Modèle de scoring des paires (brief §7.2) : LightGBM binaire en deux passes + calibration isotonique.

- Passe 1 : features des familles actives.
- Features de compétition dérivées des scores de passe 1 au sein d'un même paiement : rang,
  marge au meilleur autre candidat, meilleur score concurrent. En entraînement, les scores de
  passe 1 sont obtenus hors échantillon pour ne pas biaiser la passe 2 :
  - `walk_forward` (par défaut dans les réglages) : chaque mois est scoré par un modèle appris sur
    le seul passé, avec les labels connus à cette date, comme la production le verrait ;
  - `random_folds` (version précédente) : 3 folds tirés par paiement ; un modèle de fold voit le
    futur du train (fuite F4, docs/fuites_ml.md).
- Passe 2 : features actives + compétition.
- Calibration isotonique des scores de passe 2 sur la période de validation.

Le modèle est sauvegardé avec l'empreinte du journal, la version de la featurisation et les
paramètres, pour être rejouable à l'identique.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from src import memory, progress
from src.memory import available_cpus
from src.reconcile_ml.features import CATEGORICAL, COMPETITION
from src.settings import TrainingSettings

GROUP = "group"      # identifiant du paiement dans le jeu (paiement × jour de scoring)


def competition_features(scores: np.ndarray, group: np.ndarray) -> pd.DataFrame:
    """Rang, marge au meilleur concurrent et meilleur score concurrent, au sein de chaque groupe."""
    df = pd.DataFrame({"g": group, "s": scores})
    order = np.lexsort((-df["s"].to_numpy(), df["g"].to_numpy()))
    g, s = df["g"].to_numpy()[order], df["s"].to_numpy()[order]
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]]) if len(g) else np.array([], dtype=np.int64)
    sizes = np.diff(np.r_[starts, len(g)])
    first = np.repeat(starts, sizes)
    rank = np.arange(len(g)) - first + 1
    best = s[first]
    second = np.where(sizes > 1, s[np.minimum(starts + 1, len(g) - 1)], 0.0)
    second = np.repeat(second, sizes)
    best_other = np.where(rank == 1, second, best)
    out = np.empty((len(g), 3), dtype=np.float32)
    out[order, 0] = rank
    out[order, 1] = s - best_other
    out[order, 2] = best_other
    return pd.DataFrame(out, columns=COMPETITION)


def _params(cfg: TrainingSettings) -> dict:
    return {"objective": "binary", "learning_rate": cfg.learning_rate, "num_leaves": cfg.num_leaves,
            "min_data_in_leaf": cfg.min_data_in_leaf, "feature_fraction": 0.9, "seed": cfg.seed,
            "deterministic": True, "force_row_wise": True, "verbosity": -1, "num_threads": threads(cfg.num_threads)}


def threads(requested: int = 0) -> int:
    """Threads LightGBM : `requested` s'il est fixé, sinon les CPU utilisables du pod (jamais les cœurs
    du nœud, que LightGBM prendrait avec num_threads=0 : threads bridés, calcul très ralenti)."""
    return requested if requested > 0 else available_cpus()


def _train(X: pd.DataFrame, y: np.ndarray, Xv: pd.DataFrame | None, yv: np.ndarray | None,
           cfg: TrainingSettings, label: str = "LightGBM") -> lgb.Booster:
    cats = [c for c in CATEGORICAL if c in X.columns]
    train = lgb.Dataset(X, label=y, categorical_feature=cats, free_raw_data=True)
    valid = [lgb.Dataset(Xv, label=yv, categorical_feature=cats, reference=train)] if Xv is not None else []
    memory.mark(label)
    with progress.task(f"{label} ({len(X)} paires)", cfg.num_boost_round, "itérations") as task:
        callbacks = [progress.lightgbm_callback(task)]
        if valid:
            callbacks.append(lgb.early_stopping(30, verbose=False))
        return lgb.train(_params(cfg), train, num_boost_round=cfg.num_boost_round, valid_sets=valid,
                         callbacks=callbacks)


OOF_SCHEMES = ("walk_forward", "random_folds")


def _labels_known(pairs: pd.DataFrame, imputations: pd.DataFrame) -> np.ndarray:
    """Label de chaque paire (paiement, facture) d'après les seules imputations données."""
    truth = imputations[["payment_id", "invoice_id"]].drop_duplicates().assign(_hit=1)
    merged = pairs[["payment_id", "invoice_id"]].merge(truth, on=["payment_id", "invoice_id"], how="left")
    return merged["_hit"].fillna(0).to_numpy(dtype=np.int8)


def walk_forward_scores(train: pd.DataFrame, X: pd.DataFrame, rounds: int, cfg: TrainingSettings,
                        imputations: pd.DataFrame, purge_days: int, retention_days: int) -> tuple[np.ndarray, list[dict]]:
    """Scores de pass 1 du train, « comme en production ».

    Aux dates Dₖ (débuts de mois, tous les `walk_forward_step_months`), un modèle est appris sur les paiements
    arrivés avant Dₖ − purge (dans la fenêtre `walk_forward_window_months` si elle est fixée), avec les seuls
    labels connus à Dₖ : imputations d'horodatage antérieur à Dₖ. Un paiement sans imputation connue à Dₖ et plus
    récent que la rétention a une issue inconnue : il est écarté, pas compté négatif. Le modèle score les paires
    des paiements arrivés dans [Dₖ, Dₖ₊₁). NaN : paire sans modèle (premier mois, trop peu de positifs).
    """
    day = pd.to_datetime(train["day"]).reset_index(drop=True)
    month = day.dt.to_period("M").dt.start_time
    starts = sorted(month.unique())
    cutoffs = [pd.Timestamp(m) for m in starts[1::cfg.walk_forward_step_months]]
    imp = imputations[["payment_id", "invoice_id", "updated_at"]].dropna()
    imp_ts = pd.to_datetime(imp["updated_at"]).to_numpy()
    purge, retention = pd.Timedelta(days=purge_days), pd.Timedelta(days=retention_days)
    pairs = train[["payment_id", "invoice_id"]].reset_index(drop=True)
    fixed = cfg.model_copy(update={"num_boost_round": rounds})
    scores, log = np.full(len(train), np.nan), []
    for k, d_k in enumerate(cutoffs):
        end = cutoffs[k + 1] if k + 1 < len(cutoffs) else None
        target = ((day >= d_k) & (day < end) if end is not None else (day >= d_k)).to_numpy()
        seen = day < d_k - purge
        if cfg.walk_forward_window_months:
            seen &= day >= d_k - pd.DateOffset(months=cfg.walk_forward_window_months)
        seen = seen.to_numpy()
        known = imp[imp_ts < d_k.to_datetime64()]
        labels = _labels_known(pairs[seen], known)
        outcome = (pairs.loc[seen, "payment_id"].isin(known["payment_id"]).to_numpy()
                   | (day[seen] < d_k - retention).to_numpy())
        rows = np.flatnonzero(seen)[outcome]
        n_pos = int(labels[outcome].sum())
        entry = {"date": str(d_k.date()), "paires_apprises": int(len(rows)), "positifs": n_pos,
                 "paires_scorees": int(target.sum())}
        if n_pos < cfg.walk_forward_min_positives or not target.any():
            log.append({**entry, "modele": False})
            continue
        model = _train(X.iloc[rows], labels[outcome], None, None, fixed,
                       f"LightGBM passe 1, walk-forward {d_k.date()} ({k + 1}/{len(cutoffs)})")
        scores[target] = model.predict(X[target], num_threads=threads(cfg.num_threads))
        log.append({**entry, "modele": True})
    return scores, log


@dataclass
class PairModel:
    features: list[str]
    pass1: lgb.Booster
    pass2: lgb.Booster | None
    calibrator: IsotonicRegression | None
    meta: dict = field(default_factory=dict)
    threads: int = field(default_factory=threads)

    # --- Scoring -------------------------------------------------------------------------------------

    def raw(self, X: pd.DataFrame, group: np.ndarray) -> np.ndarray:
        s1 = self.pass1.predict(X[self.features], num_threads=self.threads)
        if self.pass2 is None:
            return s1
        comp = competition_features(s1, group)
        X2 = pd.concat([X[self.features].reset_index(drop=True), comp], axis=1)
        return self.pass2.predict(X2, num_threads=self.threads)

    def calibrate(self, raw: np.ndarray) -> np.ndarray:
        return self.calibrator.predict(raw) if self.calibrator is not None else raw

    def predict(self, X: pd.DataFrame, group: np.ndarray) -> np.ndarray:
        """Probabilité calibrée que la paire soit une vraie imputation."""
        return self.calibrate(self.raw(X, group))

    # --- Entraînement ----------------------------------------------------------------------------------

    @classmethod
    def fit(cls, train: pd.DataFrame, valid: pd.DataFrame, features: list[str], cfg: TrainingSettings,
            second_pass: bool = True, calibration: bool = True, folds: int = 3, oof: str = "random_folds",
            imputations: pd.DataFrame | None = None, purge_days: int = 5, retention_days: int = 60) -> PairModel:
        """`oof` : schéma des scores de passe 1 du train pour la pass 2 (`OOF_SCHEMES`). `walk_forward` exige
        `imputations` (payment_id, invoice_id, updated_at) et la colonne `day` du jeu."""
        if oof not in OOF_SCHEMES:
            raise ValueError(f"schéma OOF inconnu : {oof} ({', '.join(OOF_SCHEMES)})")
        train = train.reset_index(drop=True)
        y, yv = train["label"].to_numpy(), valid["label"].to_numpy()
        X, Xv = train[features], valid[features]
        pass1 = _train(X, y, Xv, yv, cfg, "LightGBM passe 1")
        pass2, info = None, {"schema": oof}
        if second_pass:
            rounds = max(pass1.best_iteration, 10)
            used = np.ones(len(train), dtype=bool)
            if oof == "walk_forward":
                if imputations is None:
                    raise ValueError("walk_forward : les imputations (dates des labels) sont nécessaires")
                scores, log = walk_forward_scores(train, X, rounds, cfg, imputations, purge_days, retention_days)
                used = ~np.isnan(scores)
                info.update(mois=log, paires_pass2=int(used.sum()), paires_sans_score=int((~used).sum()))
            if oof == "random_folds" or not used.any():
                if oof == "walk_forward":            # aucun mois scoré (historique trop court) : repli signalé
                    info["repli"] = "random_folds : aucun mois n'a assez d'historique pour un modèle walk-forward"
                    used[:] = True
                # Scores de passe 1 hors échantillon sur l'entraînement (plis par paiement).
                fold = (pd.util.hash_array(train[GROUP].astype(str).to_numpy()) % folds).astype(int)
                scores = np.zeros(len(train))
                fixed = cfg.model_copy(update={"num_boost_round": rounds})
                for k in range(folds):
                    m = _train(X[fold != k], y[fold != k], None, None, fixed,
                               f"LightGBM passe 1, pli {k + 1}/{folds} (scores hors échantillon)")
                    scores[fold == k] = m.predict(X[fold == k], num_threads=threads(cfg.num_threads))
            X2 = pd.concat([X[used].reset_index(drop=True),
                            competition_features(scores[used], train.loc[used, GROUP].to_numpy())], axis=1)
            s1v = pass1.predict(Xv, num_threads=threads(cfg.num_threads))
            Xv2 = pd.concat([Xv.reset_index(drop=True), competition_features(s1v, valid[GROUP].to_numpy())], axis=1)
            pass2 = _train(X2, y[used], Xv2, yv, cfg, "LightGBM passe 2")
        model = cls(features, pass1, pass2, None, {"oof": info})
        if calibration:
            raw_v = model.raw(valid, valid[GROUP].to_numpy())
            model.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw_v, yv)
        return model

    # --- Persistance -------------------------------------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.pass1.save_model(str(d / "pass1.txt"))
        if self.pass2 is not None:
            self.pass2.save_model(str(d / "pass2.txt"))
        if self.calibrator is not None:
            (d / "calibrator.pkl").write_bytes(pickle.dumps(self.calibrator))
        meta = {**self.meta, "features": self.features}
        (d / "model.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> PairModel:
        d = Path(directory)
        meta = json.loads((d / "model.json").read_text(encoding="utf-8"))
        pass2 = lgb.Booster(model_file=str(d / "pass2.txt")) if (d / "pass2.txt").exists() else None
        cal = pickle.loads((d / "calibrator.pkl").read_bytes()) if (d / "calibrator.pkl").exists() else None
        return cls(meta["features"], lgb.Booster(model_file=str(d / "pass1.txt")), pass2, cal, meta)


def pair_metrics(df: pd.DataFrame, score: np.ndarray) -> dict:
    """Diagnostics du scoring : AUC, precision@1, MRR (sur les groupes ayant au moins un positif)."""
    d = pd.DataFrame({"g": df[GROUP].to_numpy(), "y": df["label"].to_numpy(), "s": score})
    d = d.sort_values(["g", "s"], ascending=[True, False], kind="mergesort")
    d["rank"] = d.groupby("g").cumcount() + 1
    pos = d[d["y"] == 1]
    first = pos.groupby("g")["rank"].min()
    return {
        "auc": float(roc_auc_score(d["y"], d["s"])) if d["y"].nunique() == 2 else None,
        "precision_at_1": float((first == 1).mean()) if len(first) else None,
        "mrr": float((1.0 / first).mean()) if len(first) else None,
        "groupes_avec_positif": int(len(first)),
    }


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    df = pd.DataFrame({"bin": b, "p": p, "y": y})
    out = df.groupby("bin").agg(paires=("y", "size"), score_moyen=("p", "mean"), taux_observé=("y", "mean"))
    out.index = [f"{edges[i]:.1f}–{edges[i + 1]:.1f}" for i in out.index]
    return out.rename_axis("tranche").reset_index()


def fingerprint(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]
