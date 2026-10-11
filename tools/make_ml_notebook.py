"""Génère notebooks/exploration_ml.ipynb (et sa copie single/exploration_ml.ipynb) : exploration
orientée ML du jeu de paires et du modèle (outil de développement, hors livrable).

    python tools/make_ml_notebook.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CELLS: list[tuple[str, str]] = [
    ("md", """# Exploration ML — jeu de paires, features, modèle, erreurs, seuils

Ce notebook sert à **comprendre et améliorer l'étape 5** : qualité des candidats, pouvoir des features,
redondances, contributions, ablations par famille, nouvelles features, erreurs du modèle, calibration et
seuils. Il ne modifie rien : il lit ce que `project.train()` a produit.

**Prérequis** : un apprentissage terminé, qui a écrit :
- `data/interim/ml/dataset/` : jeu de paires (paiement, facture) avec features, `label`, `period` ;
- `models/<jeu>/pair_model/` : modèle et seuils ;
- `data/interim/ml/validation_daily_proposals.parquet` : propositions quotidiennes de la calibration en ligne.

**Garde-fous** :
- les features du jeu sont calculées sur l'état de la veille : on peut les combiner librement ;
- `label` est la vérité : ne jamais en dériver une feature ;
- comparer les variantes sur la **validation** ; la **période de test** reste réservée au backtest final.

Choix de conception de l'étape ML : `docs/choix_etape_ml.md`. Détail des features : `docs/codebase/04_modeles_ml.md`."""),
    ("code", """import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

import json

import altair as alt
import numpy as np
import pandas as pd
import pyarrow.dataset as pads
from IPython.display import Markdown, display
from sklearn.metrics import roc_auc_score

try:                                   # dépôt complet
    from src.api import Project
    from src.reconcile_ml.features import BASE, CATEGORICAL, FAMILIES
    from src.reconcile_ml.model import GROUP, PairModel, _train, calibration_table, pair_metrics
    from src.settings import TrainingSettings
except ModuleNotFoundError:            # version portable (single/)
    from reconciliation import (BASE, CATEGORICAL, FAMILIES, GROUP, PairModel, Project, TrainingSettings, _train,
                                calibration_table, pair_metrics)

pd.set_option("display.max_columns", 80)
alt.data_transformers.disable_max_rows()

project = Project(dataset="real", log=print)       # "synthetic" pour le jeu de démonstration
DATASET_DIR = project.interim_dir / "ml" / "dataset"
PROPOSALS = project.interim_dir / "ml" / "validation_daily_proposals.parquet"
MODEL_DIR = project.model_dir
SAMPLE = 0.2          # part des paiements chargés (tirage déterministe sur payment_id) ; 1.0 = tout
print(DATASET_DIR, MODEL_DIR, sep="\\n")"""),
    ("md", """## 1. Chargement du jeu de paires

Une ligne par paire (paiement, facture candidate) : les features du modèle, plus `payment_id`, `invoice_id`,
`balance` (restant dû, centimes), `payment_amount` (centimes), `day`, `label` (1 = vraie imputation),
`group` (= paiement), `period` (train / validation), `month`. Le tirage garde **tous** les candidats des paiements
retenus, pour que les analyses par paiement restent justes."""),
    ("code", """def load_pairs(path: Path, share: float = 1.0) -> pd.DataFrame:
    dataset = pads.dataset(path, format="parquet", partitioning="hive")
    parts = []
    for batch in dataset.to_batches(batch_size=500_000):
        df = batch.to_pandas()
        if share < 1:
            h = pd.util.hash_array(df["payment_id"].astype(str).to_numpy()) % 10_000
            df = df[h < share * 10_000]
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df["month"] = df["month"].astype(str)
    return df.drop_duplicates(["payment_id", "invoice_id"]).reset_index(drop=True)


pairs = load_pairs(DATASET_DIR, SAMPLE)
meta = json.loads((MODEL_DIR / "model.json").read_text(encoding="utf-8"))
FEATURES = meta["features"]
FAMILY_OF = {f: "base" for f in BASE} | {f: fam for fam, cols in FAMILIES.items() for f in cols}
train = pairs[pairs["period"] == "train"].reset_index(drop=True)
valid = pairs[pairs["period"] == "validation"].reset_index(drop=True)

print(f"{len(pairs):,} paires · {pairs['payment_id'].nunique():,} paiements · {len(FEATURES)} features".replace(",", " "))
display(pairs.groupby("period").agg(paires=("label", "size"), positifs=("label", "sum"),
                                    paiements=("payment_id", "nunique")))"""),
    ("md", """## 2. Le modèle entraîné

Métriques de validation (paires), seuils retenus par type de proposition, durées, importance des features
(gain, passe 1)."""),
    ("code", """display(pd.Series(meta["metrics"]["validation"], name="validation (paires)"))
display(pd.Series({k: v for k, v in meta["metrics"].items() if k != "validation"}, name="autres"))
display(pd.Series(meta["thresholds"]["kinds"], name="τ_high par type"))
importance = pd.read_csv(MODEL_DIR / "feature_importance.csv")
importance["famille"] = importance["feature"].map(FAMILY_OF)
alt.Chart(importance.head(30)).mark_bar().encode(
    x="gain:Q", y=alt.Y("feature:N", sort="-x"), color="famille:N").properties(height=500, title="Gain (passe 1)")"""),
    ("md", """## 3. Structure des candidats

- **Positifs par paiement** : 0 = la vraie facture n'est pas parmi les candidats (le modèle ne peut pas la
  trouver : plafond de rappel) ; plus de 1 = paiement groupé.
- **Sources** : quelle combinaison de clés (débiteur, référence, montant, client file) a proposé la paire, et avec
  quel taux de vraies imputations. Les sources peu productives coûtent des candidats sans apporter de positifs."""),
    ("code", """g = pairs.groupby("payment_id")
per_pay = pd.DataFrame({"candidats": g.size(), "positifs": g["label"].sum()})
print(f"paiements sans vraie facture candidate : {(per_pay['positifs'] == 0).mean():.1%}")
print(f"paiements groupés (> 1 positif)        : {(per_pay['positifs'] > 1).mean():.1%}")
display(per_pay["candidats"].describe(percentiles=[0.5, 0.9, 0.99]).to_frame("candidats par paiement"))

SRC = ["src_debtor", "src_reference", "src_amount", "src_client_file"]
bits = sum(pairs[c].to_numpy().astype(int) << i for i, c in enumerate(SRC))
names = {b: "+".join(c[4:] for i, c in enumerate(SRC) if b >> i & 1) for b in np.unique(bits)}
by_source = (pairs.assign(sources=pd.Series(bits).map(names)).groupby("sources")
             .agg(paires=("label", "size"), positifs=("label", "sum")))
by_source["taux_positif"] = by_source["positifs"] / by_source["paires"]
by_source["part_des_positifs"] = by_source["positifs"] / by_source["positifs"].sum()
display(by_source.sort_values("paires", ascending=False))"""),
    ("md", """## 4. Pouvoir de chaque feature, seule

- `auc` : AUC de la feature seule sur les paires de validation (0,5 = aucun pouvoir ; < 0,5 = pouvoir inversé) ;
- `top1` : part des paiements (avec au moins un positif) dont le meilleur candidat **selon cette seule feature**
  est une vraie facture — c'est la mesure qui compte pour un classement au sein du paiement ;
- `manquants` : part de valeurs absentes."""),
    ("code", """def top1_rate(df: pd.DataFrame, score: np.ndarray) -> float:
    # Part des paiements avec un positif dont le candidat de score maximal est positif (ex æquo : premier rencontré).
    d = pd.DataFrame({"g": df["payment_id"].to_numpy(), "s": np.nan_to_num(score, nan=-np.inf),
                      "y": df["label"].to_numpy()})
    has_pos = d.groupby("g")["y"].transform("max").to_numpy() == 1
    d = d[has_pos].sort_values(["g", "s"], ascending=[True, False], kind="mergesort")
    return float(d.drop_duplicates("g")["y"].mean()) if len(d) else float("nan")


def univariate(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    y = df["label"].to_numpy()
    rows = []
    for f in features:
        x = df[f].to_numpy(dtype=np.float64)
        ok = ~np.isnan(x)
        auc = np.nan
        if ok.sum() and len(np.unique(y[ok])) == 2 and np.nanstd(x[ok]) > 0:
            auc = roc_auc_score(y[ok], x[ok])
        sign = -1.0 if (not np.isnan(auc) and auc < 0.5) else 1.0
        rows.append({"feature": f, "famille": FAMILY_OF.get(f, "?"), "manquants": 1 - ok.mean(), "auc": auc,
                     "top1": top1_rate(df, sign * x), "moyenne_pos": np.nanmean(x[y == 1]) if (y == 1).any() else np.nan,
                     "moyenne_neg": np.nanmean(x[y == 0]) if (y == 0).any() else np.nan})
    out = pd.DataFrame(rows)
    out["auc_orientée"] = np.maximum(out["auc"], 1 - out["auc"])
    return out.sort_values("top1", ascending=False).reset_index(drop=True)


uni = univariate(valid, [f for f in FEATURES if f in valid.columns])
display(uni)"""),
    ("md", """## 5. Distribution d'une feature, positifs contre négatifs

Changer `FEATURE` pour explorer. Les valeurs extrêmes sont rognées aux quantiles 0,5 % et 99,5 %."""),
    ("code", """FEATURE = "amount_diff_rel"

x = valid[FEATURE].astype(float)
lo, hi = x.quantile([0.005, 0.995])
edges = np.linspace(lo, hi, 41) if hi > lo else np.array([lo - 0.5, lo + 0.5])
rows = []
for lab, part in valid.groupby("label"):
    counts, _ = np.histogram(part[FEATURE].astype(float).clip(lo, hi).dropna(), bins=edges)
    rows.append(pd.DataFrame({"valeur": (edges[:-1] + edges[1:]) / 2, "part": counts / max(counts.sum(), 1),
                              "label": "vraie imputation" if lab == 1 else "autre candidat"}))
alt.Chart(pd.concat(rows)).mark_line(interpolate="step-after").encode(
    x=alt.X("valeur:Q", title=FEATURE), y="part:Q", color="label:N").properties(height=260, title=FEATURE)"""),
    ("md", """## 6. Redondances

Corrélations de rang (Spearman) entre features sur un échantillon de validation : les paires au-dessus du seuil
portent la même information (candidates à la suppression, ou signe d'une feature mal construite)."""),
    ("code", """sample = valid[[f for f in FEATURES if f in valid.columns]].sample(min(len(valid), 200_000), random_state=0)
corr = sample.rank().corr()
upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
redundant = upper.stack().rename("spearman").reset_index()
redundant = redundant[redundant["spearman"].abs() >= 0.95].sort_values("spearman", key=np.abs, ascending=False)
display(redundant.rename(columns={"level_0": "feature_a", "level_1": "feature_b"}))"""),
    ("md", """## 7. Contributions (valeurs SHAP de LightGBM)

`pred_contrib=True` donne, pour chaque paire, la contribution de chaque feature au score (en log-odds) de la
**passe 1**. Moyenne des valeurs absolues = importance « réelle » sur ces données, à comparer au gain."""),
    ("code", """model = PairModel.load(MODEL_DIR)
X = valid[model.features].sample(min(len(valid), 50_000), random_state=0)
contrib = model.pass1.predict(X, pred_contrib=True)
shap = pd.DataFrame({"feature": model.features, "shap_abs_moyen": np.abs(contrib[:, :-1]).mean(axis=0)})
shap = shap.merge(importance[["feature", "gain"]], on="feature", how="left")
shap["famille"] = shap["feature"].map(FAMILY_OF)
display(shap.sort_values("shap_abs_moyen", ascending=False).head(30))"""),
    ("md", """## 8. Ablation par famille de features

Réentraînement **rapide** (passe 1 seule, moins d'itérations) sur l'échantillon, en retirant une famille à la fois :
la baisse d'AUC, de precision@1 et de MRR mesure ce que la famille apporte. Comparaison relative uniquement : les
chiffres absolus ne sont pas ceux du modèle complet (pas de passe 2, échantillon, hyperparamètres rapides)."""),
    ("code", """QUICK = TrainingSettings(num_boost_round=150, learning_rate=0.1, num_leaves=63, min_data_in_leaf=50, seed=42)


def quick_eval(features: list[str], tr: pd.DataFrame = train, va: pd.DataFrame = valid) -> dict:
    booster = _train(tr[features], tr["label"].to_numpy(), va[features], va["label"].to_numpy(), QUICK, "rapide")
    return pair_metrics(va, booster.predict(va[features]))


base_features = [f for f in FEATURES if f in train.columns]
reference = quick_eval(base_features)
rows = [{"retirée": "(aucune)", **reference}]
for fam in sorted({FAMILY_OF.get(f) for f in base_features} - {"base"}):
    kept = [f for f in base_features if FAMILY_OF.get(f) != fam]
    rows.append({"retirée": fam, **quick_eval(kept)})
ablation = pd.DataFrame(rows)
for k in ("auc", "precision_at_1", "mrr"):
    ablation[f"Δ {k}"] = ablation[k] - reference[k]
display(ablation)"""),
    ("md", """## 9. Tester une nouvelle feature

Gabarit : construire des features **à partir des colonnes du jeu** (jamais de `label`), puis comparer avec
`quick_eval`. Les exemples ci-dessous sont des pistes, pas des recommandations.

Si une feature apporte un gain net sur la validation :
1. l'implémenter dans `Featurizer.features` (`src/reconcile_ml/features.py`), dans une famille de `FAMILIES` ;
2. incrémenter `FEATURIZATION_VERSION` ;
3. relancer `project.train()` (le jeu enregistré sera reconstruit), puis le backtest sur le test.

Une feature qui demande l'état du grand livre (encours, historique du débiteur…) ne peut pas être prototypée
ici : il faut la calculer dans le `Featurizer`, qui a accès à `LedgerState` à la date du jour."""),
    ("code", """def add_experimental(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    paid, bal = df["payment_amount"].astype(float), df["balance"].astype(float)
    out["x_payment_is_round"] = ((df["payment_amount"] % 10_000) == 0).astype("float32")      # multiple de 100 €
    out["x_log_amount_ratio"] = (np.log1p(paid.clip(lower=0)) - np.log1p(bal.clip(lower=0))).astype("float32")
    out["x_balance_rank_desc"] = df.groupby("payment_id")["balance"].rank(ascending=False, method="first")
    out["x_share_of_open_in_payment"] = (bal / df.groupby("payment_id")["balance"].transform("sum")).astype("float32")
    return out


NEW = ["x_payment_is_round", "x_log_amount_ratio", "x_balance_rank_desc", "x_share_of_open_in_payment"]
train_x, valid_x = add_experimental(train), add_experimental(valid)
display(univariate(valid_x, NEW))
with_new = quick_eval(base_features + NEW, train_x, valid_x)
display(pd.DataFrame([{"variante": "référence", **reference}, {"variante": "+ nouvelles features", **with_new}]))"""),
    ("md", """## 10. Analyse d'erreurs (validation, modèle complet)

Paiements à **une** vraie facture candidate dont le meilleur candidat du modèle n'est pas la bonne : on compare la
facture choisie et la vraie sur quelques features clés. C'est la source la plus directe d'idées de features."""),
    ("code", """valid_scored = valid.assign(score=model.raw(valid, valid[GROUP].to_numpy()))
n_pos = valid_scored.groupby("payment_id")["label"].transform("sum")
single = valid_scored[n_pos == 1].sort_values(["payment_id", "score"], ascending=[True, False], kind="mergesort")
top = single.drop_duplicates("payment_id")
wrong = top[top["label"] == 0]["payment_id"]
print(f"precision@1 (paiements à une vraie facture) : {(top['label'] == 1).mean():.2%} · erreurs : {len(wrong):,}"
      .replace(",", " "))

KEY = ["score", "payment_amount", "balance", "amount_diff_rel", "ref_key_in_label", "ref_full_in_label",
       "alloc_rank", "alloc_is_firm_debtor", "days_to_due", "src_reference", "src_amount", "n_candidates"]
KEY = [k for k in KEY if k in single.columns]
errors = single[single["payment_id"].isin(wrong.head(20))]
chosen = errors.drop_duplicates("payment_id").set_index("payment_id")[KEY].add_suffix(" (choisie)")
truth = errors[errors["label"] == 1].set_index("payment_id")[KEY].add_suffix(" (vraie)")
display(chosen.join(truth).T)"""),
    ("md", """## 11. Calibration (validation)

Score calibré (isotonique) contre taux observé de vraies imputations, par tranche. La diagonale = calibration
parfaite."""),
    ("code", """calib = calibration_table(valid_scored["label"].to_numpy(), model.calibrate(valid_scored["score"].to_numpy()))
display(calib)
diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash=[4, 4], color="gray").encode(x="x", y="y")
points = alt.Chart(calib).mark_line(point=True).encode(x=alt.X("score_moyen:Q", title="score calibré moyen"),
                                                       y=alt.Y("taux_observé:Q", title="taux observé"),
                                                       tooltip=["tranche", "paires"])
(diag + points).properties(height=300, width=300)"""),
    ("md", """## 12. Propositions et seuils (calibration en ligne)

Les propositions quotidiennes de la validation (sans auto-validation ML). Pour chaque type, la meilleure
proposition de chaque paiement sur toute sa présence dans le reliquat : précision cumulée en fonction du score.
C'est une approximation de la calibration en ligne (qui simule le premier jour de franchissement du seuil) ;
le trait vertical est le τ_high retenu."""),
    ("code", """props = pd.read_parquet(PROPOSALS)
best = props.sort_values(["payment_id", "score"], ascending=[True, False]).drop_duplicates(["payment_id", "kind"])
curves = []
for kind, part in best.groupby("kind"):
    part = part.sort_values("score", ascending=False)
    curves.append(pd.DataFrame({"kind": kind, "score": part["score"].to_numpy(),
                                "propositions": np.arange(1, len(part) + 1),
                                "précision": np.cumsum(part["correct"].to_numpy()) / np.arange(1, len(part) + 1)}))
curves = pd.concat(curves)
display(best.groupby("kind").agg(paiements=("payment_id", "nunique"), exactes=("correct", "mean")))
taus = pd.DataFrame([{"kind": k, "tau": v} for k, v in meta["thresholds"]["kinds"].items() if v <= 1])
line = alt.Chart(curves).mark_line().encode(x=alt.X("score:Q", scale=alt.Scale(reverse=True)),
                                            y=alt.Y("précision:Q", scale=alt.Scale(zero=False)), color="kind:N")
rule = alt.Chart(taus).mark_rule(strokeDash=[4, 4]).encode(x="tau:Q", color="kind:N")
(line + rule).properties(height=300, title="Précision cumulée par score décroissant (validation)")"""),
]


def build() -> dict:
    cells = []
    for kind, text in CELLS:
        lines = text.splitlines(keepends=True)
        if kind == "md":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": lines})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines})
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}}}


def main() -> None:
    text = json.dumps(build(), indent=1, ensure_ascii=False)
    for out in (ROOT / "notebooks" / "exploration_ml.ipynb", ROOT / "single" / "exploration_ml.ipynb"):
        out.parent.mkdir(exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(out)


if __name__ == "__main__":
    main()
