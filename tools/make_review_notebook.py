"""Génère notebooks/review_ml.ipynb : le notebook unique de la review de l'étape ML, à dérouler en séance.

Deux temps :
- la veille, `PREPARE = True` : calcule tout ce qui est long (expériences de leakage, intervalles de confiance,
  pire cas F2, exemples) et le met en cache dans `reports/<jeu>/review/` ;
- en séance, `PREPARE = False` : tout se relit du cache, en quelques secondes.

    python tools/make_review_notebook.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CELLS: list[tuple[str, str]] = [
    ("md", r"""# Review de l'étape ML : résultat, fonctionnement, leakage, évolutions

**Mode d'emploi.**
- **La veille** : `PREPARE = True`, puis *Run All*. Les calculs longs (replay du test pour la calibration par la
  précision, expériences de leakage, intervalles de confiance, pire cas F2, exemples) sont mis en cache dans
  `reports/<jeu>/review/`. Compter la durée de la « calibration en ligne » de `model.json` (replay du test), plus 10 à
  20 minutes.
- **En séance** : `PREPARE = False`, puis *Run All*. Tout se relit du cache en quelques secondes, sans entraînement ni
  replay.

Prérequis, vérifiés par la cellule de contrôle : données chargées, modèle entraîné, backtest du test et export des
décisions, tous sur le même journal.

| § | Sujet | Ce qu'on montre |
|---|---|---|
| 1 | Le résultat | cascade actuel → règles → règles + ML, précision et intervalle de confiance, stabilité par mois |
| 2 | La pipeline | boucle quotidienne, split temporel, allocation, règles |
| 3 | Deep dive ML | candidats, features, stacking, décision, un paiement de bout en bout, validation contre test, calibration par la précision, challenger AutoGluon |
| 4 | Leakage | la théorie, puis la vérification : pas de leakage grossier, impact du leakage résiduel |
| 5 | Évolutions | plan de correction et décisions à prendre |"""),
    ("code", r"""import json
import pickle
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

import altair as alt
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pads
import pyarrow.parquet as pq
from IPython.display import Markdown, display
from scipy.stats import beta
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss

try:                                   # dépôt complet
    from src.api import Project
    from src.evaluation.metrics import ground_truth
    from src.load.interim import load_interim
    from src.memory import available_cpus
    from src.reconcile_ml.decision import calibrate_thresholds, decide
    from src.reconcile_ml.features import BASE, CATEGORICAL, FAMILIES
    from src.reconcile_ml.model import GROUP, PairModel, _params, competition_features, pair_metrics
    from src.reconcile_ml.pipeline import label_pairs, offline_proposals, online_thresholds
except ModuleNotFoundError:            # version portable (single/)
    from reconciliation import (BASE, CATEGORICAL, FAMILIES, GROUP, PairModel, Project, _params, calibrate_thresholds,
                                competition_features, decide, ground_truth, label_pairs, load_interim, memory,
                                offline_proposals, online_thresholds, pair_metrics)
    available_cpus = memory.available_cpus

import warnings
warnings.filterwarnings("ignore", message="invalid value encountered in cast")
pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 200)
alt.data_transformers.disable_max_rows()"""),
    ("code", r"""DATASET = "real"          # "synthetic" pour répéter sur le jeu synthétique
PREPARE = False           # True la veille (calculs longs mis en cache) ; False en séance (lecture du cache)
TRAIN_PAYMENTS = 40_000   # paiements du train tirés pour les expériences de leakage (§ 4)
VALID_PAYMENTS = 15_000   # paiements de la validation tirés pour ces expériences
AUTOGLUON_PYTHON = None   # challenger (§ 3.8) : Python d'un environnement séparé où AutoGluon est installé
AUTOGLUON_PRESET = "medium_quality"   # presets avec bagging refusés (folds non groupés par paiement)
AUTOGLUON_TIME_LIMIT = 1200           # secondes par modèle AutoGluon (deux modèles)
SEED = 42"""),
    ("code", r"""project = Project(dataset=DATASET, log=print)
settings = project.settings
meta = project.meta()
assert meta is not None, "aucune donnée chargée : lancer project.load() d'abord"
JOURNAL = meta["journal_sha256"]
REP, MODEL_DIR, INTERIM = project.reports_dir, project.model_dir, project.interim_dir
REVIEW_DIR = REP / "review"
REVIEW_DIR.mkdir(parents=True, exist_ok=True)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def cached(name: str, compute):
    # PREPARE : calcule et enregistre ; sinon relit, ou signale ce qui manque sans interrompre la séance.
    path = REVIEW_DIR / f"{name}.pkl"
    if path.exists() and not PREPARE:
        saved = pickle.loads(path.read_bytes())
        if saved["journal"] != JOURNAL:
            display(Markdown(f"⚠ `{name}` a été préparé sur un autre journal : relancer avec `PREPARE = True`."))
        return saved["value"]
    if not PREPARE:
        display(Markdown(f"⚠ `{name}` n'est pas préparé : relancer le notebook avec `PREPARE = True` avant la séance."))
        return None
    print(f"… calcul de {name}")
    try:
        value = compute()
    except Exception as exc:                       # la préparation continue : le point manquant est signalé
        display(Markdown(f"⚠ `{name}` : échec du calcul ({type(exc).__name__} : {exc})."))
        return None
    path.write_bytes(pickle.dumps({"journal": JOURNAL, "value": value}))
    return value


def pct(v, digits: int = 2) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{100 * v:.{digits}f} %".replace(".", ",")


def num(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{int(v):,}".replace(",", " ")


utc = lambda v: pd.to_datetime(v, utc=True).dt.tz_localize(None)
EVAL = read_json(REP / "evaluation_pipeline_test.json")
MODEL_META = read_json(MODEL_DIR / "model.json")
EXPORT = REP / "decisions_pipeline_test.parquet"
DS_DIR = INTERIM / "ml" / "dataset"
DS = pads.dataset(DS_DIR, partitioning="hive") if DS_DIR.exists() else None
checks = [
    ("données chargées", True, meta.get("source")),
    ("modèle entraîné sur ce journal", MODEL_META is not None and MODEL_META.get("journal_sha256") == JOURNAL,
     "project.train()"),
    ("backtest du test sur ce journal", EVAL is not None and EVAL["context"]["journal_sha256"] == JOURNAL,
     'project.backtest("test", matcher="pipeline")'),
    ("export des décisions du test", EXPORT.exists(), 'project.export_decisions("test", "pipeline")'),
    ("jeu de paires du ML", DS is not None, "project.train()"),
]
display(pd.DataFrame([(a, "oui" if b else "NON", c) for a, b, c in checks],
                     columns=["prérequis", "ok", "détail ou commande à lancer"]))

FEATS = MODEL_META["features"] if MODEL_META else []
CATS = [c for c in CATEGORICAL if c in FEATS]


def sample_pairs(period: str, n: int, columns=None) -> pd.DataFrame:
    # Paiements entiers tirés au hasard (graine fixe) dans une période du dataset de training.
    ids = pc.unique(DS.to_table(columns=["payment_id"], filter=pads.field("period") == period)["payment_id"])
    take = np.sort(np.random.default_rng(SEED).choice(len(ids), size=min(n, len(ids)), replace=False))
    df = DS.to_table(columns=columns, filter=(pads.field("period") == period)
                     & pads.field("payment_id").isin(ids.take(pa.array(take)))).to_pandas()
    df["month"] = df["month"].astype(str)
    return (df.drop_duplicates(["payment_id", "invoice_id"])
            .sort_values(["day", "payment_id", "invoice_id"], kind="mergesort").reset_index(drop=True))


def lgb_fit(Xd, yd, rounds=None, Xval=None, yval=None):
    # LightGBM avec les hyperparamètres de production ; early stopping si un jeu de validation est donné.
    cfg = settings.reconcile_ml.training

    def dset(Xs, ys, ref=None):
        return lgb.Dataset(Xs, label=ys, categorical_feature=[c for c in CATS if c in Xs.columns], free_raw_data=False,
                           reference=ref)

    d = dset(Xd, yd)
    if Xval is None:
        return lgb.train(_params(cfg), d, num_boost_round=rounds)
    return lgb.train(_params(cfg), d, num_boost_round=cfg.num_boost_round, valid_sets=[dset(Xval, yval, d)],
                     callbacks=[lgb.early_stopping(30, verbose=False)])


def scores(df: pd.DataFrame, s: np.ndarray) -> dict:
    return {**pair_metrics(df, s), "logloss": log_loss(df["label"], np.clip(s, 1e-7, 1 - 1e-7), labels=[0, 1])}"""),

    # --- 1. Résultat -----------------------------------------------------------------------------------------
    ("md", r"""## 1. Le résultat

La métrique métier : le **taux d'automatisation à précision fixée** (cible 99,5 %), mesuré sur le **test** (les
2 derniers mois), que ni le training ni le choix des seuils n'ont vu. Une auto-validation n'est juste que si
l'ensemble de factures proposé est **exactement** celui qui a été réellement imputé."""),
    ("code", r"""S = EVAL["summary"]
cascade = pd.read_csv(REP / "evaluation_pipeline_test_cascade.csv")
current = settings.evaluation.current_automation_rate
if current is not None:
    cascade.loc[cascade["niveau"] == "Algorithme actuel", "taux_automatisation"] = current
cascade["gain (pts)"] = cascade["taux_automatisation"].diff() * 100
display(Markdown(f"**{num(S['auto'])} paiements auto-validés sur {num(S['paiements_périmètre'])}**, "
                 f"précision **{pct(S['précision'])}** (cible {pct(S['précision_cible'], 1)})."
                 + ("" if current is not None else
                    "  \n⚠ Taux actuel non renseigné : `evaluation.current_automation_rate` dans `config/settings.yaml`.")))
display(cascade[["niveau", "taux_automatisation", "précision", "gain (pts)"]].style.format(
    {"taux_automatisation": pct, "précision": pct, "gain (pts)": "{:+.1f}"}, na_rep="—").hide(axis="index"))
bars = cascade.dropna(subset=["taux_automatisation"])
chart = alt.Chart(bars).mark_bar(color="#245C99").encode(
    x=alt.X("taux_automatisation:Q", title="taux d'automatisation", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
    y=alt.Y("niveau:N", sort=None, title=None))
labels = chart.mark_text(align="left", dx=4).encode(text=alt.Text("taux_automatisation:Q", format=".1%"))
(chart + labels).properties(width=560, height=40 * len(bars), title="Cascade sur le test")"""),
    ("md", r"""### La précision tient-elle statistiquement ?

Borne basse de **Clopper-Pearson unilatérale à 95 %** de la précision des auto-validations : la vraie précision est
au-dessus avec 95 % de confiance."""),
    ("code", r"""def _precision_ci():
    d = pd.read_parquet(EXPORT, columns=["payment_id", "action", "step", "decision_retenue", "decision_retenue_exacte"])
    auto = d[d["decision_retenue"] & (d["action"] == "auto")].drop_duplicates("payment_id")

    def bound(g: pd.DataFrame) -> dict:
        n, k = len(g), int(g["decision_retenue_exacte"].astype(float).sum())
        return {"auto-validations": n, "erreurs": n - k, "précision": k / n if n else np.nan,
                "borne basse 95 %": beta.ppf(0.05, k, n - k + 1) if k else 0.0}

    rows = {step: bound(g) for step, g in auto.groupby("step")}
    rows["total"] = bound(auto)
    return pd.DataFrame(rows).T


CI = cached("precision_ci", _precision_ci) if EXPORT.exists() else None
if CI is not None:
    display(CI.style.format({"auto-validations": num, "erreurs": num, "précision": pct, "borne basse 95 %": pct}))
    ok = CI.loc["total", "borne basse 95 %"] >= settings.evaluation.target_precision
    display(Markdown(f"Borne basse totale : **{pct(CI.loc['total', 'borne basse 95 %'])}**, "
                     + ("au-dessus de la cible." if ok else "**sous la cible** : voir § 4.5 (F1).")))"""),
    ("md", r"""### Stabilité dans le temps et par type de groupe

Le chiffre `taux_automatisation_à_précision_cible` du rapport est un **plafond a posteriori** : il choisit son seuil
sur le test (seuil oracle). Il n'est donné ici qu'à titre indicatif ; le résultat est celui de la cascade, obtenu
avec les seuils de la validation."""),
    ("code", r"""by_month = pd.read_csv(REP / "evaluation_pipeline_test_by_month.csv")
by_group = pd.read_csv(REP / "evaluation_pipeline_test_by_group.csv")
fmt = {"taux_automatisation": pct, "précision": pct, "paiements": num, "auto": num, "corrects": num}
display(Markdown("**Par mois**"), by_month.style.format(fmt).hide(axis="index"))
display(Markdown("**Par type de groupe réel**"), by_group.style.format(fmt).hide(axis="index"))
display(Markdown(f"Plafond a posteriori (seuil oracle, indicatif) : {pct(S.get('taux_automatisation_à_précision_cible'))}"))"""),

    # --- 2. Pipeline -----------------------------------------------------------------------------------------
    ("md", r"""## 2. La pipeline

```
                        chaque jour D (même code en replay et en production)
 Journal ──► LedgerState ──► Allocation ──────► Règles R1–R5 ──────► ML ─────────────► Décision
 événements  état figé       paiement →         solution unique      candidats,         auto si p ≥ τ_high
 datés       à la veille     débiteurs          seulement            pass 1 → pass 2    et marge ≥ δ ; revue ;
             (as_of = D)     candidats                               ensembles          sinon manuel
                                       └── chaque étape ne traite que le résiduel de la précédente ──┘
   les paiements non résolus restent en reliquat (60 jours) et sont retraités le lendemain

 Training : le replay de cette boucle sur train + validation produit le dataset de paires (features à D),
            labellisé dans une seconde passe à partir de la table imputation.
```"""),
    ("code", r"""split = project.split()
display(pd.DataFrame([{"période": p.name, "du": p.start, "au": p.end, "jours": p.days} for p in split.periods])
        .style.hide(axis="index"))
volumes = {name: pq.ParquetFile(INTERIM / f"{name}.parquet").metadata.num_rows
           for name in ("payment", "invoice", "imputation", "client_file") if (INTERIM / f"{name}.parquet").exists()}
display(pd.Series(volumes).map(num).to_frame("lignes").T)

ALLOC = read_json(REP / "allocation_test.json")
if ALLOC:
    a = ALLOC["summary"]
    display(Markdown("**Allocation (test)**" + ("" if ALLOC["context"].get("journal_sha256") == JOURNAL else
                                                 " — ⚠ mesurée sur un autre journal : relancer `project.measure_allocation('test')`")))
    display(pd.Series({"rappel (le vrai débiteur est candidat)": pct(a["rappel_premier_passage"]),
                       "allocations fermes": pct(a["taux_ferme"]), "précision des allocations fermes": pct(a["précision_ferme"]),
                       "sans aucun candidat": pct(a["sans_candidat"]), "client file rattaché": pct(a["avec_client_file"])})
            .to_frame("valeur"))
by_rule = pd.read_csv(REP / "evaluation_pipeline_test_by_rule.csv").sort_values("taux_automatisation", ascending=False)
display(Markdown("**Par règle et par type de proposition ML** (taux sur l'ensemble des paiements du test)"),
        by_rule.style.format(fmt).hide(axis="index"))"""),

    # --- 3. Deep dive ML -------------------------------------------------------------------------------------
    ("md", r"""## 3. Deep dive ML

Le ML ne traite que le **résiduel des règles**. Pour chaque paiement : des candidats, une ligne de features par paire
(paiement, facture candidate), un score en deux passes, puis une proposition et une décision.

### 3.1 Candidats

Union de quatre sources, avec filtres durs (devise, facture ouverte à D, contrat actif) et au plus
`max_per_payment` candidats :
- les factures des débiteurs alloués, échéance de D − 180 j à D + 30 j ;
- les factures dont une référence est citée dans le libellé, sans fenêtre ;
- les factures au montant exact, à ± 90 j ;
- les factures citées par le client file.

Le **recall des candidats** est le plafond du ML : un paiement dont la vraie facture n'est pas candidate ne peut pas
être rapproché juste."""),
    ("code", r"""M = MODEL_META["metrics"]
display(pd.Series({"paiements passés en ML (test)": num(S.get("ml_paiements_en_ml")),
                   "recall des candidats, test": pct(S.get("ml_rappel_candidats")),
                   "recall des candidats, validation": pct(M["rappel_candidats_validation"])}).to_frame("valeur"))


def _sources():
    src = ["src_debtor", "src_reference", "src_amount", "src_client_file"]
    t = DS.to_table(columns=["label", *src], filter=pads.field("period") == "validation").to_pandas()
    flags = t[src].astype(bool)
    pos = flags[t["label"].to_numpy() == 1]
    only = pos[(pos.sum(axis=1) == 1).to_numpy()].sum() / max(len(pos), 1)
    out = pd.DataFrame({"part des paires candidates": flags.mean(), "part des vraies factures trouvées": pos.mean(),
                        "vraies factures trouvées par cette seule source": only})
    out.index = ["débiteur alloué", "référence citée", "montant exact", "client file"]
    return out


SOURCES = cached("sources", _sources) if DS is not None else None
if SOURCES is not None:
    display(Markdown("**Sources des candidats (validation)**"), SOURCES.style.format(pct))"""),
    ("md", r"""### 3.2 Features

54 features en huit familles plus la base, **toutes calculées sur l'état figé à D** ; le comportemental est une
fenêtre strictement antérieure de 180 jours. Importance : gain de la pass 1."""),
    ("code", r"""imp = pd.read_csv(MODEL_DIR / "feature_importance.csv")
family_of = {f: fam for fam, cols in FAMILIES.items() for f in cols} | {f: "base" for f in BASE}
imp["famille"] = imp["feature"].map(family_of).fillna("autre")
imp["part du gain"] = imp["gain"] / imp["gain"].sum()
fam = (imp.groupby("famille").agg(features=("feature", "size"), part=("part du gain", "sum"))
       .sort_values("part", ascending=False).reset_index())
display(alt.Chart(fam).mark_bar(color="#245C99").encode(
    x=alt.X("part:Q", title="part du gain (pass 1)", axis=alt.Axis(format="%")),
    y=alt.Y("famille:N", sort="-x", title=None), tooltip=["famille", "features", alt.Tooltip("part:Q", format=".1%")])
    .properties(width=480, height=260, title="Importance par famille"))
display(Markdown("**Les 15 premières features**"),
        imp.head(15)[["feature", "famille", "part du gain"]].style.format({"part du gain": pct}).hide(axis="index"))"""),
    ("md", r"""### 3.3 Le modèle : stacking en deux passes

```
pass 1      LightGBM binaire, 54 features, entraîné sur le train, early stopping sur la validation
folds OOF   GroupKFold par paiement (3 folds) : chaque paire du train scorée par un modèle qui n'a pas vu son paiement
compétition rang, marge au meilleur concurrent, meilleur concurrent — calculés dans chaque paiement
            (train : à partir des scores OOF ; validation, test, production : à partir de la pass 1)
pass 2      LightGBM binaire, 57 features, early stopping sur la validation → score brut, utilisé par la décision
calibration isotonique sur la validation, pour la lecture seulement
```

Pourquoi du **pointwise binaire** plutôt qu'un objectif de ranking : la décision compare le score à un seuil
**absolu**, commun à tous les paiements. Un objectif listwise optimise l'ordre au sein du groupe, pas l'échelle. Les
features de compétition de la pass 2 apportent le contexte du groupe."""),
    ("code", r"""p1 = lgb.Booster(model_file=str(MODEL_DIR / "pass1.txt"))
p2 = lgb.Booster(model_file=str(MODEL_DIR / "pass2.txt")) if (MODEL_DIR / "pass2.txt").exists() else None
tr = settings.reconcile_ml.training
display(pd.Series({"pass 1 : arbres": p1.num_trees(), "pass 2 : arbres": p2.num_trees() if p2 else "—",
                   "learning rate": tr.learning_rate, "feuilles": tr.num_leaves,
                   "lignes minimum par feuille": tr.min_data_in_leaf, "early stopping (patience)": 30,
                   "part des paiements du résiduel gardés": tr.payment_sample}, dtype=object).to_frame("valeur"))
v = M["validation"]
display(Markdown("**Validation**"), pd.Series({"AUC": f"{v['auc']:.4f}", "precision@1": pct(v["precision_at_1"]),
                                                "MRR": f"{v['mrr']:.3f}", "paiements avec un positif": num(v["groupes_avec_positif"])})
        .to_frame("valeur"))"""),
    ("md", r"""### 3.4 Validation contre test : pas de dégradation

La precision@1 du rapport de test (`ml_precision_at_1`) compte comme un échec tout paiement **sans vraie facture
candidate**. Celle de la validation ne porte que sur les paiements qui en ont une. Pour comparer à définition égale :"""),
    ("code", r"""def _test_ranking():
    path = INTERIM / "replay" / "replay_pipeline_test_ml_candidates.parquet"
    score = "raw" if "raw" in pq.read_schema(path).names else "p"       # score brut : pas de paliers isotoniques
    c = pd.read_parquet(path, columns=["payment_id", "invoice_id", score])
    imp_ = pd.read_parquet(INTERIM / "imputation.parquet", columns=["payment_id", "invoice_id"]).drop_duplicates()
    c = c[c["payment_id"].isin(imp_["payment_id"].unique())]
    c = c.merge(imp_.assign(label=True), on=["payment_id", "invoice_id"], how="left")
    c["label"] = c["label"].fillna(False).astype(bool)
    c = c.sort_values(["payment_id", score, "invoice_id"], ascending=[True, False, True], kind="mergesort")
    c["rank"] = c.groupby("payment_id").cumcount() + 1
    has_pos = c.groupby("payment_id")["label"].any()
    first = c[c["label"]].groupby("payment_id")["rank"].min()
    return pd.DataFrame({
        "test · tous les paiements en ML (définition du rapport)": {
            "paiements": len(has_pos), "precision@1": float((first.reindex(has_pos.index) == 1).mean()), "MRR": np.nan},
        "test · paiements avec une vraie facture candidate": {
            "paiements": int(has_pos.sum()), "precision@1": float((first == 1).mean()), "MRR": float((1 / first).mean())},
        "validation · paiements avec une vraie facture candidate": {
            "paiements": v["groupes_avec_positif"], "precision@1": v["precision_at_1"], "MRR": v["mrr"]}}).T


RANKING = cached("test_ranking", _test_ranking)
if RANKING is not None:
    display(RANKING.style.format({"paiements": num, "precision@1": pct, "MRR": "{:.3f}"}, na_rep="—"))"""),
    ("md", r"""### 3.5 Un paiement de bout en bout

Un paiement de la validation, pris au hasard parmi ceux que le modèle auto-valide justement : ses candidats, leurs
features clés, le score de pass 1, les features de compétition, le score de pass 2, la proposition et la décision."""),
    ("code", r"""def _examples():
    model = PairModel.load(MODEL_DIR)
    light = DS.to_table(columns=["payment_id", "label", "n_candidates"], filter=pads.field("period") == "validation").to_pandas()
    g = light.groupby("payment_id").agg(pos=("label", "sum"), n=("n_candidates", "first"))
    rng = np.random.default_rng(SEED)
    singles = g[(g["pos"] == 1) & g["n"].between(4, 12)].index.to_numpy()
    sets = g[g["pos"].between(2, 4) & g["n"].between(4, 15)].index.to_numpy()
    pick = np.concatenate([rng.choice(singles, min(200, len(singles)), replace=False),
                           rng.choice(sets, min(200, len(sets)), replace=False)])
    id_type = DS.schema.field("payment_id").type
    sub = DS.to_table(filter=(pads.field("period") == "validation")
                      & pads.field("payment_id").isin(pa.array(list(pick), type=id_type))).to_pandas()
    sub = sub.drop_duplicates(["payment_id", "invoice_id"]).sort_values(["payment_id", "invoice_id"]).reset_index(drop=True)
    s1 = model.pass1.predict(sub[model.features])
    comp = competition_features(s1, sub[GROUP].to_numpy())
    raw = model.raw(sub, sub[GROUP].to_numpy())
    props, _ = offline_proposals(sub, raw, settings)
    props["action"] = decide(props, MODEL_META["thresholds"])
    truth = sub[sub["label"] == 1].groupby("payment_id")["invoice_id"].agg(lambda s: tuple(sorted(s)))
    props["correcte"] = [truth.get(pid) == inv for pid, inv in zip(props["payment_id"], props["invoice_ids"])]
    keys = ["invoice_id", "label", "payment_amount", "balance", "src_debtor", "src_reference", "src_amount",
            "amount_diff", "ref_key_in_label", "days_to_due", "alloc_score"]
    table = sub[["payment_id", *keys]].assign(s1=s1, rang=comp["rank_in_payment"].to_numpy(),
                                              marge=comp["score_margin"].to_numpy(), s2=raw, p=model.calibrate(raw))
    out = {}
    for kind in ("single", "set"):
        good = props[(props["kind"] == kind) & (props["action"] == "auto") & props["correcte"]]
        chosen = good if len(good) else props[props["kind"] == kind]
        if chosen.empty:
            continue
        prop = chosen.iloc[0]
        out[kind] = {"candidats": table[table["payment_id"] == prop["payment_id"]].sort_values("s2", ascending=False)
                     .drop(columns="payment_id").reset_index(drop=True),
                     "proposition": prop[["payment_id", "kind", "invoice_ids", "payment_amount", "score", "margin",
                                          "action", "correcte"]]}
    return out


EXAMPLES = cached("examples", _examples) if DS is not None else None
TH = MODEL_META["thresholds"]
for kind, ex in (EXAMPLES or {}).items():
    prop = ex["proposition"]
    display(Markdown(f"#### Proposition `{kind}` · paiement `{prop['payment_id']}` · {prop['payment_amount'] / 100:,.2f} €"
                     .replace(",", " ")))
    tbl = ex["candidats"].head(10).assign(payment_amount=lambda d: d["payment_amount"] / 100, balance=lambda d: d["balance"] / 100)
    ints = ["label", "src_debtor", "src_reference", "src_amount", "ref_key_in_label", "days_to_due", "rang"]
    display(tbl.rename(columns={"payment_amount": "paiement (€)", "balance": "restant dû (€)"}).style.format(
        {"s1": "{:.4f}", "marge": "{:+.4f}", "s2": "{:.4f}", "p": "{:.4f}", "amount_diff": "{:.2f}", "alloc_score": "{:.2f}",
         "paiement (€)": "{:.2f}", "restant dû (€)": "{:.2f}", **{c: "{:.0f}" for c in ints}}).hide(axis="index"))
    display(Markdown(f"Proposition : factures {list(prop['invoice_ids'])}, score {prop['score']:.4f}, marge "
                     f"{prop['margin']:.4f} → seuil `{kind}` {TH['kinds'].get(kind, TH['tau_high']):.4f}, marge minimale "
                     f"{TH['min_margin']} → **{prop['action'] or 'manuel'}**, {'juste' if prop['correcte'] else 'fausse'}."))"""),
    ("md", r"""### 3.6 Propositions et décision

Pour chaque paiement, des propositions concurrentes :
- `single` : une facture qui absorbe le paiement, à 5 € ou 3 % près ;
- `partial` : une facture plus grande que le paiement ;
- `set` : un ensemble trouvé par DFS borné (au plus 5 factures parmi les 25 mieux scorées), de cardinalité minimale ;
  son score est la moyenne de ses paires.

La meilleure l'emporte, avec une **marge** au second. Auto-validation si `score ≥ τ_high(type)` et `marge ≥ δ`,
revue dès `τ_low`, sinon manuel. `τ_high` est calibré par type sur le **replay de la validation** : premier jour où le
score franchit le seuil, au moins 20 propositions, précision ≥ 99,5 %."""),
    ("code", r"""display(pd.DataFrame({"τ_high": TH["kinds"]}).assign(**{"δ (marge)": TH["min_margin"], "τ_low (revue)": TH["tau_low"]})
        .style.format("{:.4f}"))
never = [k for k, t in TH["kinds"].items() if t > 1]
if never:
    display(Markdown(f"τ_high > 1 pour {', '.join(f'`{k}`' for k in never)} : jamais auto-validé, faute d'un seuil qui atteigne "
                     "la cible sur au moins 20 propositions de la validation. Ces propositions partent en revue."))
curve = pd.read_csv(REP / "evaluation_pipeline_test_curve.csv").replace([np.inf, -np.inf], np.nan)
line = alt.Chart(curve).mark_line(color="#245C99").encode(
    x=alt.X("taux_automatisation:Q", title="taux d'automatisation", axis=alt.Axis(format="%"), scale=alt.Scale(zero=False)),
    y=alt.Y("précision:Q", title="précision", axis=alt.Axis(format=".1%"), scale=alt.Scale(zero=False)))
target = alt.Chart(pd.DataFrame({"y": [settings.evaluation.target_precision]})).mark_rule(strokeDash=[4, 4], color="#A3520F").encode(y="y:Q")
point = alt.Chart(pd.DataFrame({"x": [S["taux_automatisation"]], "y": [S["précision"]]})).mark_point(
    size=140, filled=True, color="#14213D").encode(x="x:Q", y="y:Q")
(line + target + point).properties(width=560, height=300, title="Test : courbe a posteriori (ligne), point réel obtenu avec les seuils de validation, cible en pointillé")"""),

    ("md", r"""### 3.7 Calibration par la précision : le seuil choisi sur la validation tient-il sur le test ?

La décision ne lit pas une probabilité : elle compare un **score à un seuil** choisi pour une **précision** cible.
Le score est donc bien calibré « par la précision » si un même seuil donne la même précision d'une période à l'autre.

On le mesure **à protocole identique** sur la validation et sur le test : replay sans auto-validation ML, propositions
enregistrées chaque jour, et pour chaque seuil τ la règle de `calibrate_online`. Un paiement est auto-validé le
premier jour où son score franchit τ avec une marge ≥ δ, et l'auto-validation n'est juste que si l'ensemble proposé est
exactement l'ensemble imputé.

Le replay du test ne sert qu'à **mesurer** : les seuils restent ceux de la validation. Le seuil que le test aurait choisi
n'est affiché qu'en diagnostic de stabilité. En préparation, ce replay dure à peu près autant que la « calibration en
ligne » de `model.json`."""),
    ("code", r"""def _test_probe():
    print("  replay du test sans auto-validation ML (mesure seulement, les seuils restent ceux de la validation)")
    data, journal, _ = load_interim(INTERIM)
    out = REVIEW_DIR / "test_probe"
    out.mkdir(exist_ok=True)
    chosen = online_thresholds(data, journal, PairModel.load(MODEL_DIR), settings, project.rules, split.period("test"),
                               ground_truth(data.tables["imputation"]), print, out)
    return {"kinds": chosen.get("kinds", {}), "path": str(out / "validation_daily_proposals.parquet")}


def crossings(records: pd.DataFrame, delta: float) -> pd.DataFrame:
    # Jours où le score d'un paiement (marge ≥ δ) atteint un nouveau maximum : candidats au premier franchissement.
    k = records.assign(s=np.where(records["margin"] >= delta, records["score"], -np.inf))
    k = k.sort_values(["payment_id", "day"], kind="mergesort")
    cm = k.groupby("payment_id")["s"].cummax()
    k["prev"] = cm.groupby(k["payment_id"]).shift(fill_value=-np.inf).to_numpy()
    return k[k["s"] > k["prev"]]


def lower(n: int, k: int) -> float:
    return beta.ppf(0.05, k, n - k + 1) if k else 0.0


def _precision_calibration():
    delta = MODEL_META["thresholds"]["min_margin"]
    sets = {"validation": pd.read_parquet(INTERIM / "ml" / "validation_daily_proposals.parquet"),
            "test": pd.read_parquet(PROBE["path"])}
    curve, at_tau, bins = [], [], []
    edges = [0.3, 0.9, 0.99, 0.995, 0.997, 0.998, 0.999, 1.000001]
    labels = ["0,300–0,900", "0,900–0,990", "0,990–0,995", "0,995–0,997", "0,997–0,998", "0,998–0,999", "≥ 0,999"]
    for jeu, rec in sets.items():
        for kind, recs in rec.groupby("kind"):
            c = crossings(recs, delta)
            s, prev, ok = c["s"].to_numpy(), c["prev"].to_numpy(), c["correct"].to_numpy().astype(bool)
            for tau in np.linspace(0.99, 0.9995, 60):
                m = (prev < tau) & (tau <= s)
                if m.sum() >= 20:
                    curve.append({"kind": kind, "jeu": jeu, "seuil": float(tau), "auto-validations": int(m.sum()),
                                  "précision": float(ok[m].mean())})
            tau = MODEL_META["thresholds"]["kinds"].get(kind, 1.01)
            m = (prev < tau) & (tau <= s)
            n, k = int(m.sum()), int(ok[m].sum())
            at_tau.append({"type": kind, "jeu": jeu, "τ_high (validation)": tau, "auto-validations": n,
                           "précision": k / n if n else np.nan, "borne basse 95 %": lower(n, k) if n else np.nan})
            best = c[np.isfinite(c["s"])].drop_duplicates("payment_id", keep="last")      # meilleur score du paiement
            best = best.assign(tranche=pd.cut(best["s"], edges, labels=labels, right=False))
            g = best.groupby("tranche", observed=True)["correct"].agg(paiements="size", justes="sum").reset_index()
            bins.append(g.assign(kind=kind, jeu=jeu))
    bins = pd.concat(bins, ignore_index=True)
    bins["précision"] = bins["justes"] / bins["paiements"]
    bins["borne basse 95 %"] = [lower(n, k) for n, k in zip(bins["paiements"], bins["justes"])]
    at_tau = pd.DataFrame(at_tau)
    at_tau["τ que le test aurait choisi (diagnostic)"] = [PROBE["kinds"].get(k, np.nan) if j == "test" else np.nan
                                                        for k, j in zip(at_tau["type"], at_tau["jeu"])]
    return {"bins": bins, "curve": pd.DataFrame(curve), "at_tau": at_tau}


PROBE = cached("test_probe", _test_probe) if MODEL_META else None
PC = cached("precision_calibration", _precision_calibration) if PROBE else None
if PC is not None:
    display(Markdown("**Auto-validations au seuil choisi sur la validation : validation contre test**"),
            PC["at_tau"].style.format({"τ_high (validation)": "{:.4f}", "auto-validations": num, "précision": pct,
                                       "borne basse 95 %": pct, "τ que le test aurait choisi (diagnostic)": "{:.4f}"},
                                      na_rep="—").hide(axis="index"))
    for kind in ("single", "set"):
        b = PC["bins"][PC["bins"]["kind"] == kind]
        if b.empty:
            continue
        table = b.pivot(index="tranche", columns="jeu", values=["paiements", "précision", "borne basse 95 %"])
        table = table.reindex(columns=["validation", "test"], level=1)
        display(Markdown(f"**`{kind}` : précision par tranche du meilleur score du paiement (marge ≥ δ)**"),
                table.style.format({c: (num if c[0] == "paiements" else pct) for c in table.columns}, na_rep="—"))
    charts = []
    for kind in ("single", "set"):
        c = PC["curve"][PC["curve"]["kind"] == kind]
        if c.empty:
            continue
        line = alt.Chart(c).mark_line(point=True).encode(
            x=alt.X("seuil:Q", title="seuil τ sur le score", scale=alt.Scale(zero=False)),
            y=alt.Y("précision:Q", title="précision des auto-validations", axis=alt.Axis(format=".1%"), scale=alt.Scale(zero=False)),
            color=alt.Color("jeu:N", scale=alt.Scale(domain=["validation", "test"], range=["#245C99", "#A3520F"])),
            tooltip=["jeu", alt.Tooltip("seuil:Q", format=".4f"), "auto-validations", alt.Tooltip("précision:Q", format=".3%")])
        layers = [line, alt.Chart(pd.DataFrame({"y": [settings.evaluation.target_precision]}))
                  .mark_rule(strokeDash=[4, 4], color="gray").encode(y="y:Q")]
        tau = TH["kinds"].get(kind)
        if tau is not None and tau <= 1:
            layers.append(alt.Chart(pd.DataFrame({"x": [tau]})).mark_rule(color="#14213D").encode(x="x:Q"))
        charts.append(alt.layer(*layers).properties(width=380, height=280,
                                                    title=f"{kind} (trait plein : τ_high de la validation, pointillé : cible)"))
    if charts:
        display(alt.hconcat(*charts))"""),

    ("md", r"""### 3.8 Challenger : AutoGluon contre LightGBM

**La question.** LightGBM avec des hyperparamètres standard laisse-t-il de la performance sur la table ? AutoGluon
(`TabularPredictor`) essaie plusieurs familles de modèles (LightGBM, CatBoost, XGBoost, forêts aléatoires, extra
trees, k-NN, réseaux de neurones selon le preset et les extras installés) et les combine par ensembling.

**Protocole, identique pour tous les modèles.**
- Même échantillon de paiements (`TRAIN_PAYMENTS`, `VALID_PAYMENTS`) et mêmes features.
- La validation est coupée dans le temps en deux moitiés :
  - **V1** sert à l'early stopping de LightGBM et au tuning et à l'ensembling d'AutoGluon (`tuning_data`) ;
  - **V2** sert uniquement à l'évaluation.
- Quatre modèles :
  - LightGBM pass 1 ;
  - AutoGluon sur les mêmes 54 features ;
  - LightGBM pass 2 (la configuration de production) ;
  - AutoGluon avec les 3 features de compétition (son équivalent de la pass 2, nourri par les mêmes scores OOF).
- Deux niveaux de comparaison :
  - **par paire** : AUC, precision@1, MRR, log-loss ;
  - **par décision** : seuils choisis sur V1 avec la méthode offline, appliqués à V2 ; on compare le taux
    d'auto-validation et la précision.

**Comment lire le résultat.** AutoGluon doit battre la pass 2 de LightGBM **de plus que le bruit** pour justifier son
coût :
- des ensembles non déterministes ;
- un training et une inférence plus longs ;
- des dépendances lourdes, **incompatibles avec l'environnement du projet** ;
- une explicabilité réduite ;
- des seuils à recalibrer.

**Environnement séparé, obligatoire.** AutoGluon impose pandas < 2.4, alors que le projet utilise pandas 3 (chaînes
Arrow). L'installer dans l'environnement du projet casse la normalisation. Le notebook exporte donc les données en
parquet et lance AutoGluon dans **un autre Python**, par sous-processus. À préparer une fois :

```
python -m venv /chemin/venv-autogluon
/chemin/venv-autogluon/bin/pip install "autogluon.tabular[all]" pyarrow     # Windows : venv-autogluon\Scripts\pip
```

Puis, dans la cellule de paramètres : `AUTOGLUON_PYTHON = "/chemin/venv-autogluon/bin/python"`.

**Points pratiques.**
- Sans les extras `[all]`, AutoGluon se limite aux modèles disponibles : le leaderboard dit lesquels ont tourné.
- Les presets avec bagging (`good_quality` et au-dessus) tirent leurs folds au hasard, ligne par ligne : ils
  mélangeraient les candidats d'un même paiement entre folds. Ils sont donc refusés ici.
- Sur un petit échantillon, les seuils choisis sur V1 sont instables, et la précision par décision peut tomber sous la
  cible pour tous les modèles. Il faut alors comparer les modèles **entre eux**, pas à la cible."""),
    ("code", r"""AG_SCRIPT = '''
# Exécuté dans l'environnement AutoGluon (pandas 2) : ne dépend pas du code du projet.
import json, sys, time
from pathlib import Path
import pandas as pd
import autogluon.core
from autogluon.tabular import TabularPredictor

d = Path(sys.argv[1])
cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
result = {"version": autogluon.core.__version__, "timings": {}}
for name, cols in cfg["models"].items():
    def load(part, with_label):
        df = pd.read_parquet(d / f"{part}.parquet", columns=cols + (["label"] if with_label else []))
        for c in cfg["categorical"]:
            if c in df.columns:
                df[c] = df[c].map(lambda v: "manquant" if pd.isna(v) else str(int(v)))
        return df
    train, v1, v2 = load("train", True), load("v1", True), load("v2", False)
    t = time.perf_counter()
    predictor = TabularPredictor(label="label", problem_type="binary", eval_metric="log_loss",
                                 path=str(d / "models" / name), verbosity=1)
    predictor.fit(train, tuning_data=v1, presets=cfg["preset"], time_limit=cfg["time_limit"], num_cpus=cfg["num_cpus"])
    result["timings"][name] = time.perf_counter() - t
    for part, df in (("v1", v1.drop(columns="label")), ("v2", v2)):
        p = predictor.predict_proba(df, as_multiclass=False).to_numpy()
        pd.DataFrame({"p": p}).to_parquet(d / f"pred_{name}_{part}.parquet", index=False)
    board = predictor.leaderboard(silent=True)[["model", "score_val", "fit_time", "pred_time_val"]]
    board.to_csv(d / f"leaderboard_{name}.csv", index=False)
(d / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
'''


def run_autogluon(folder: Path) -> dict:
    import subprocess
    script = folder / "run_autogluon.py"
    script.write_text(AG_SCRIPT, encoding="utf-8")
    python = AUTOGLUON_PYTHON or sys.executable
    done = subprocess.run([str(python), str(script), str(folder)], capture_output=True, text=True)
    if done.returncode != 0:
        tail = (done.stderr or done.stdout)[-1500:]
        raise RuntimeError("AutoGluon a échoué. Vérifier AUTOGLUON_PYTHON (environnement séparé, voir ci-dessus).\n" + tail)
    return json.loads((folder / "result.json").read_text(encoding="utf-8"))


def _challenger():
    import time
    if AUTOGLUON_PRESET.split("_")[0] in ("good", "high", "best", "extreme"):
        raise ValueError(f"preset {AUTOGLUON_PRESET} : bagging par folds aléatoires, non groupés par paiement")
    train, valid = sample_pairs("train", TRAIN_PAYMENTS), sample_pairs("validation", VALID_PAYMENTS)
    vday = pd.to_datetime(valid["day"])
    first_half = (vday < vday.min() + (vday.max() - vday.min()) / 2).to_numpy()
    v1, v2 = valid[first_half].reset_index(drop=True), valid[~first_half].reset_index(drop=True)
    X, y, y1 = train[FEATS], train["label"].to_numpy(), v1["label"].to_numpy()
    timings, results = {}, {}

    print("  LightGBM : pass 1, folds OOF, pass 2")
    t = time.perf_counter()
    p1 = lgb_fit(X, y, Xval=v1[FEATS], yval=y1)
    rounds = max(p1.best_iteration, 10)
    fold = (pd.util.hash_array(train[GROUP].astype(str).to_numpy()) % 3).astype(int)
    oof = np.zeros(len(train))
    for k in range(3):
        oof[fold == k] = lgb_fit(X[fold != k], y[fold != k], rounds).predict(X[fold == k])
    timings["LightGBM pass 1 (54 features)"] = time.perf_counter() - t
    s1 = {"v1": p1.predict(v1[FEATS]), "v2": p1.predict(v2[FEATS])}

    def with_comp(df, s):
        return pd.concat([df[FEATS].reset_index(drop=True), competition_features(s, df[GROUP].to_numpy())], axis=1)

    X2, X2v1, X2v2 = with_comp(train, oof), with_comp(v1, s1["v1"]), with_comp(v2, s1["v2"])
    t = time.perf_counter()
    p2 = lgb_fit(X2, y, Xval=X2v1, yval=y1)
    timings["LightGBM pass 2 (production)"] = time.perf_counter() - t + timings["LightGBM pass 1 (54 features)"]
    results["LightGBM pass 1 (54 features)"] = (s1["v1"], s1["v2"])
    results["LightGBM pass 2 (production)"] = (p2.predict(X2v1), p2.predict(X2v2))

    print(f"  AutoGluon (environnement séparé) : preset {AUTOGLUON_PRESET}, {AUTOGLUON_TIME_LIMIT} s par modèle")
    folder = REVIEW_DIR / "autogluon"
    folder.mkdir(parents=True, exist_ok=True)
    X2.assign(label=y).to_parquet(folder / "train.parquet", index=False)
    X2v1.assign(label=y1).to_parquet(folder / "v1.parquet", index=False)
    X2v2.to_parquet(folder / "v2.parquet", index=False)
    comp = [c for c in X2.columns if c not in FEATS]
    models = {"base": ("AutoGluon (54 features)", FEATS), "comp": ("AutoGluon + compétition (57 features)", FEATS + comp)}
    (folder / "config.json").write_text(json.dumps({
        "categorical": CATS, "preset": AUTOGLUON_PRESET, "time_limit": AUTOGLUON_TIME_LIMIT, "num_cpus": available_cpus(),
        "models": {key: cols for key, (_, cols) in models.items()}}), encoding="utf-8")
    ag = run_autogluon(folder)
    boards = {}
    for key, (name, _) in models.items():
        results[name] = tuple(pd.read_parquet(folder / f"pred_{key}_{part}.parquet")["p"].to_numpy() for part in ("v1", "v2"))
        timings[name] = ag["timings"][key]
        boards[name] = pd.read_csv(folder / f"leaderboard_{key}.csv").head(8)

    truth = valid[valid["label"] == 1].groupby("payment_id")["invoice_id"].agg(lambda s: tuple(sorted(s)))

    def exact(props: pd.DataFrame) -> np.ndarray:
        return np.array([truth.get(pid) == inv for pid, inv in zip(props["payment_id"], props["invoice_ids"])])

    n_v2 = v2["payment_id"].nunique()
    metrics, decisions = {}, {}
    for name, (sa, sb) in results.items():
        metrics[name] = {**scores(v2, sb), "training (s)": timings[name]}
        props_a, _ = offline_proposals(v1, sa, settings)
        th = calibrate_thresholds(props_a, exact(props_a), settings.evaluation.target_precision, settings.reconcile_ml.decision)
        props_b, _ = offline_proposals(v2, sb, settings)
        auto = decide(props_b, th) == "auto"
        ok = exact(props_b)[auto]
        n, k = int(auto.sum()), int(ok.sum())
        decisions[name] = {"auto-validations": n, "taux d'auto-validation": n / n_v2, "précision": k / n if n else np.nan,
                           "borne basse 95 %": beta.ppf(0.05, k, n - k + 1) if k else np.nan}
    return {"metrics": pd.DataFrame(metrics).T, "decisions": pd.DataFrame(decisions).T, "boards": boards,
            "version": ag["version"], "preset": AUTOGLUON_PRESET, "time_limit": AUTOGLUON_TIME_LIMIT,
            "volumes": {"paires train": len(train), "paires V1": len(v1), "paires V2": len(v2), "paiements V2": n_v2}}


CH = cached("autogluon", _challenger) if DS is not None else None
if CH:
    vol = CH["volumes"]
    display(Markdown(f"AutoGluon {CH['version']}, preset `{CH['preset']}`, au plus {CH['time_limit']} s par modèle · "
                     f"{num(vol['paires train'])} paires de train, tuning sur {num(vol['paires V1'])} paires (V1), "
                     f"évaluation sur {num(vol['paires V2'])} paires (V2, {num(vol['paiements V2'])} paiements)."))
    m = CH["metrics"]
    display(Markdown("**Par paire, sur V2**"), m[["auc", "precision_at_1", "mrr", "logloss", "training (s)"]].style.format(
        {"auc": "{:.4f}", "precision_at_1": pct, "mrr": "{:.3f}", "logloss": "{:.4f}", "training (s)": "{:.0f}"}))
    display(Markdown("**Par décision** : seuils choisis sur V1 (méthode offline), appliqués à V2"),
            CH["decisions"].style.format({"auto-validations": num, "taux d'auto-validation": pct, "précision": pct,
                                          "borne basse 95 %": pct}, na_rep="—"))
    best = m["precision_at_1"].idxmax()
    ref = m.loc["LightGBM pass 2 (production)", "precision_at_1"]
    n_pos = m.loc["LightGBM pass 2 (production)", "groupes_avec_positif"]
    se = np.sqrt(ref * (1 - ref) / n_pos)
    gap = m.loc[best, "precision_at_1"] - ref
    if best == "LightGBM pass 2 (production)":
        verdict_ = "la configuration de production reste la meilleure en precision@1."
    else:
        verdict_ = (f"meilleur modèle `{best}`, {100 * gap:+.2f} pt par rapport à la production : "
                    + ("**écart significatif** (plus de 2 erreurs-types)." if gap > 2 * se else "**dans le bruit**."))
    display(Markdown(f"Erreur-type de la precision@1 sur V2 : **±{100 * se:.2f} pt** ; {verdict_}"))
    for name, board in CH["boards"].items():
        display(Markdown(f"**Leaderboard {name}** (`score_val` = −log-loss sur V1)"),
                board.style.format({"score_val": "{:.5f}", "fit_time": "{:.0f}", "pred_time_val": "{:.1f}"}).hide(axis="index"))"""),

    # --- 4. Leakage ------------------------------------------------------------------------------------------
    ("md", r"""## 4. Leakage : la théorie, puis la vérification

**Pourquoi l'impact est structurellement limité.** Le test est isolé par le split temporel, et les seuils sont figés
avant de l'ouvrir. Un leakage qui intervient pendant le training ou la sélection ne peut donc que :
- **dégrader le modèle**, ce qui se voit sur le test comme un taux plus bas, jamais plus haut ;
- **rendre optimistes les métriques de validation**, que nous ne communiquons pas comme résultat.

**Exception** : un leakage **de données**, présent dans le journal lui-même, touche aussi le replay du test. C'est le
cas de F2 (référentiel IBAN non daté) et de F5 (imputation datée avant son paiement), mesurés aux § 4.6 et 4.7.

**Pas de leakage grossier dans les folds : la théorie.**
- Le score OOF d'une paire vaut `s(p, j) = f₋ₖ₍ₚ₎(x_pj)`, et ses features de compétition `c_pj = g(s(p, ·))_j`.
- Le fold est tiré **par paiement** : le modèle `f₋ₖ₍ₚ₎` n'a vu aucun label de p, donc `c_pj` non plus.
- Deux conditions d'implémentation, remplies : l'unité du fold est l'unité d'agrégation de `g` (le paiement), et chaque
  paiement n'est enregistré qu'une fois.
- Ce que la théorie ne garantit pas : l'**échangeabilité dans le temps** des paiements. C'est F4, § 4.3.

Les expériences qui suivent sont faites sur un **échantillon** de paiements du dataset de training (paramètres
`TRAIN_PAYMENTS`, `VALID_PAYMENTS`), avec les hyperparamètres de production. Elles mesurent des mécanismes, pas le
modèle de production."""),
    ("code", r"""def _experiments():
    feats = FEATS
    cols = feats + ["payment_id", "invoice_id", "label", "group", "period", "day", "month"]
    train, valid = sample_pairs("train", TRAIN_PAYMENTS, cols), sample_pairs("validation", VALID_PAYMENTS, cols)
    X, y, Xv, yv = train[feats], train["label"].to_numpy(), valid[feats], valid["label"].to_numpy()
    fit = lgb_fit

    res = {"volumes": {"paiements train": train["payment_id"].nunique(), "paires train": len(train),
                       "paiements validation": valid["payment_id"].nunique(), "paires validation": len(valid)}}
    print("  pass 1 et folds")
    p1_ = fit(X, y, Xval=Xv, yval=yv)
    rounds = max(p1_.best_iteration, 10)
    s1_in, s1_v = p1_.predict(X), p1_.predict(Xv)
    fold = (pd.util.hash_array(train[GROUP].astype(str).to_numpy()) % 3).astype(int)
    oof, per_fold = np.zeros(len(train)), []
    for k in range(3):
        m = fit(X[fold != k], y[fold != k], rounds)
        oof[fold == k] = m.predict(X[fold == k])
        per_fold.append({"fold": k + 1, "in-sample": scores(train[fold != k], m.predict(X[fold != k]))["precision_at_1"],
                         "fold scoré (OOF)": scores(train[fold == k], oof[fold == k])["precision_at_1"],
                         "validation": scores(valid, m.predict(Xv))["precision_at_1"]})
    res["rounds"] = rounds
    res["pass1"] = pd.DataFrame({"train, in-sample": scores(train, s1_in), "train, OOF": scores(train, oof),
                                 "validation (pass 1 complète)": scores(valid, s1_v)}).T
    res["folds"] = pd.DataFrame(per_fold).set_index("fold")
    res["overlap"] = {"paiements à la fois en train et en validation": len(set(train["payment_id"]) & set(valid["payment_id"])),
                      "lignes vues par les 3 modèles de fold": int(sum((fold != k).sum() for k in range(3))),
                      "dont lignes de validation": 0, "lignes du train (×2 attendu)": len(train)}
    comp_tr = competition_features(oof, train[GROUP].to_numpy())
    comp_v = competition_features(s1_v, valid[GROUP].to_numpy())
    res["competition"] = pd.DataFrame({"train (OOF), moyenne": comp_tr.mean(), "validation, moyenne": comp_v.mean(),
                                       "train (OOF), médiane": comp_tr.median(), "validation, médiane": comp_v.median()})

    print("  walk-forward mensuel")
    imp_ = pd.read_parquet(INTERIM / "imputation.parquet", columns=["payment_id", "invoice_id", "updated_at"]).dropna()
    imp_ = imp_[imp_["payment_id"].isin(set(train["payment_id"]))]
    imp_["updated_at"] = utc(imp_["updated_at"])
    purge, retention = pd.Timedelta(days=settings.split.purge_days), pd.Timedelta(days=settings.split.retention_days)
    day = pd.to_datetime(train["day"])
    mstart = day.dt.to_period("M").dt.start_time
    wf, wf_log = np.full(len(train), np.nan), []
    for d_k in sorted(mstart.unique())[1:]:
        d_k = pd.Timestamp(d_k)
        seen = (day < d_k - purge).to_numpy()
        known = imp_[imp_["updated_at"] < d_k]
        lab = label_pairs(train.loc[seen, ["payment_id", "invoice_id"]], known)
        outcome = (train.loc[seen, "payment_id"].isin(known["payment_id"]) | (day[seen] < d_k - retention)).to_numpy()
        rows_fit = np.flatnonzero(seen)[outcome]
        n_pos = int(lab[outcome].sum())
        entry = {"mois scoré": d_k.strftime("%Y-%m"), "paires vues": len(rows_fit), "positifs": n_pos}
        if n_pos < 50:
            wf_log.append({**entry, "statut": "trop peu de positifs"})
            continue
        target = (mstart == d_k).to_numpy()
        wf[target] = fit(X.iloc[rows_fit], lab[outcome], rounds).predict(X[target])
        wf_log.append({**entry, "statut": "scoré"})
    has = ~np.isnan(wf)
    same = train[has].reset_index(drop=True)
    res["wf_log"] = pd.DataFrame(wf_log)
    res["oof_compare"] = pd.DataFrame({"in-sample": scores(same, s1_in[has]),
                                       "GroupKFold aléatoire, OOF (code actuel)": scores(same, oof[has]),
                                       "walk-forward (sans futur)": scores(same, wf[has])}).T

    print("  pass 2")
    Xv2 = pd.concat([Xv.reset_index(drop=True), comp_v], axis=1)

    def pass2(Xbase, comp, yy):
        X2 = pd.concat([Xbase.reset_index(drop=True), comp.reset_index(drop=True)], axis=1)
        return fit(X2, yy, Xval=Xv2, yval=yv).predict(Xv2)

    s2_all = pass2(X, comp_tr, y)
    s2_same = pass2(same[feats], comp_tr[has], same["label"].to_numpy())
    s2_wf = pass2(same[feats], competition_features(wf[has], same[GROUP].to_numpy()), same["label"].to_numpy())
    res["pass2"] = pd.DataFrame({"pass 1 seule": scores(valid, s1_v),
                                 "pass 2 · GroupKFold, tout le train (code actuel)": scores(valid, s2_all),
                                 "pass 2 · GroupKFold, mêmes paires que le walk-forward": scores(valid, s2_same),
                                 "pass 2 · walk-forward": scores(valid, s2_wf)}).T

    print("  canal du nombre d'itérations (T4)")
    p1_sh = fit(X, y, Xval=Xv, yval=np.random.default_rng(SEED).permutation(yv))
    rounds_sh = max(p1_sh.best_iteration, 10)

    def oof_scores(r):
        out = np.zeros(len(train))
        for k in range(3):
            out[fold == k] = fit(X[fold != k], y[fold != k], r).predict(X[fold == k])
        return out

    oof_again = oof_scores(rounds)
    oof_sh = oof_scores(rounds_sh) if rounds_sh != rounds else oof_again
    res["t4"] = pd.DataFrame({
        "itérations des folds": [rounds, rounds, rounds_sh],
        "écart moyen des scores OOF": [0.0, float(np.abs(oof_again - oof).mean()), float(np.abs(oof_sh - oof).mean())],
        "écart max des scores OOF": [0.0, float(np.abs(oof_again - oof).max()), float(np.abs(oof_sh - oof).max())],
        "precision@1 OOF": [scores(train, s)["precision_at_1"] for s in (oof, oof_again, oof_sh)]},
        index=["référence", "contrôle : mêmes itérations, folds refaits", "labels de validation shufflés"])

    print("  calibration isotonique")
    vday = pd.to_datetime(valid["day"])
    v1 = (vday < vday.min() + (vday.max() - vday.min()) / 2).to_numpy()
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(s2_all[v1], yv[v1])
    clip = lambda p: np.clip(p, 1e-7, 1 - 1e-7)
    top = s2_all >= np.quantile(s2_all, 0.95)
    res["isotonic"] = pd.Series({"log-loss sur V2, scores bruts": log_loss(yv[~v1], clip(s2_all[~v1])),
                                 "log-loss sur V2, isotonique fit sur V1": log_loss(yv[~v1], clip(iso.predict(s2_all[~v1]))),
                                 "valeurs distinctes des 5 % plus hauts scores, brut": float(len(np.unique(s2_all[top]))),
                                 "valeurs distinctes des 5 % plus hauts scores, isotonique": float(len(np.unique(iso.predict(s2_all[top]))))})
    return res


EXP = cached("experiments", _experiments) if DS is not None else None
if EXP:
    display(pd.Series(EXP["volumes"]).map(num).to_frame("échantillon").T)"""),
    ("md", r"""### 4.1 Les folds ne leakent pas le label

Si les scores OOF contenaient le label, ils ressembleraient aux scores **in-sample**. Ils doivent ressembler à ceux de
la **validation**."""),
    ("code", r"""if EXP:
    display(EXP["pass1"][["auc", "precision_at_1", "mrr", "logloss"]].style.format(
        {"auc": "{:.4f}", "precision_at_1": pct, "mrr": "{:.3f}", "logloss": "{:.4f}"}))
    display(Markdown("**precision@1 de chaque modèle de fold**"), EXP["folds"].style.format(pct))
    display(Markdown("**Features de compétition : train (OOF) contre validation**"), EXP["competition"].style.format("{:.3f}"))"""),
    ("md", r"""### 4.2 Aucun modèle ne voit la validation pendant son training

Les périodes sont disjointes par jour, séparées par une purge. Les modèles de fold sont fit sur des sous-ensembles du
train, sans early stopping."""),
    ("code", r"""gap = (pd.Timestamp(split.period("validation").start) - pd.Timestamp(split.period("train").end)).days
display(Markdown(f"Écart entre le dernier jour du train et le premier jour de la validation : **{gap} jours** "
                 f"(purge configurée : {settings.split.purge_days})."))
if EXP:
    display(pd.Series(EXP["overlap"]).map(num).to_frame("valeur"))"""),
    ("md", r"""Pas d'information future dans les **features** : toute lecture d'état passe par `LedgerState` avec un `as_of`, et
le test anti-fuite du dépôt vérifie qu'un journal tronqué après D ne change aucune sortie à D. Le notebook
`ml_pas_a_pas` (§ 8.1) le vérifie aussi sur l'apprentissage complet : mêmes paires et mêmes features après troncature,
seuls des labels changent (§ 4.5).

### 4.3 Leakage résiduel F4 : le CV des folds n'est pas temporel

**Théorie.** Un GroupKFold suppose les paiements **échangeables**. Ils ne le sont pas : les features du débiteur
forment une quasi-empreinte, et la relation features → label dérive dans le temps. Un paiement de janvier est scoré par
un modèle qui a vu février à août : c'est un **look-ahead leakage**, confiné au training de la pass 2. Il ne peut que
dégrader le modèle.

**Mesure.** Walk-forward mensuel en expanding window : pour scorer le mois k, un modèle entraîné uniquement sur les
paiements arrivés avant `Dₖ − purge`, avec les labels connus à `Dₖ`. Puis une pass 2 entraînée sur chaque source de
scores, **à paires identiques**, évaluée sur la validation."""),
    ("code", r"""if EXP:
    display(Markdown("**Scores des paires du train (mêmes paires, hors premier mois)**"),
            EXP["oof_compare"][["auc", "precision_at_1", "logloss"]].style.format({"auc": "{:.4f}", "precision_at_1": pct, "logloss": "{:.4f}"}))
    display(Markdown("**Pass 2 évaluée sur la validation : le test décisif**"),
            EXP["pass2"][["auc", "precision_at_1", "mrr", "logloss"]].style.format(
                {"auc": "{:.4f}", "precision_at_1": pct, "mrr": "{:.3f}", "logloss": "{:.4f}"}))
    d = EXP["pass2"]["precision_at_1"]
    delta = d["pass 2 · walk-forward"] - d["pass 2 · GroupKFold, mêmes paires que le walk-forward"]
    display(Markdown(f"Walk-forward − GroupKFold, à paires identiques : **{100 * delta:+.2f} pt** de precision@1. "
                     + ("Le walk-forward fait au moins aussi bien : la correction ne coûte rien." if delta >= 0 else
                        "Le walk-forward fait moins bien ici : l'écart est le prix du volume perdu le premier mois.")))"""),
    ("md", r"""### 4.4 Un canal validation → train : le nombre d'itérations des folds

**Théorie.** Les modèles de fold ne voient aucune ligne de validation, mais leur nombre d'itérations vient de l'early
stopping de la pass 1 **sur la validation**. Les features de compétition du train en dépendent donc.

**Mesure causale.** Contrôle : mêmes itérations, folds refaits → scores OOF identiques (seul canal). Labels de
validation shufflés → le nombre d'itérations change, et les scores OOF avec lui (canal actif)."""),
    ("code", r"""if EXP:
    display(EXP["t4"].style.format({"itérations des folds": "{:.0f}", "écart moyen des scores OOF": "{:.4f}",
                                    "écart max des scores OOF": "{:.4f}", "precision@1 OOF": pct}))"""),
    ("md", r"""### 4.5 F1 et F3 : la validation sert à choisir, les labels n'ont pas d'horizon

**F1, théorie.** La validation sert à cinq choix (early stopping des deux passes, itérations des folds, isotonique,
seuils), puis à mesurer : ses métriques sont optimistes. Le choix de τ_high est un **winner's curse** (seuil le plus
permissif atteignant la cible sur au moins 20 propositions). D'où l'intervalle de confiance du § 1, mesuré sur le
**test**.

**F3, théorie.** Les labels sont lus sur toute la table `imputation`, sans condition de date : un paiement de fin de
période reçoit son label final même s'il a été prononcé plus tard. Le chiffre qui décide est le délai entre l'arrivée
du paiement et son imputation, comparé à la purge."""),
    ("code", r"""if EXP:
    display(Markdown("**Calibration isotonique, out-of-sample (fit sur la 1re moitié de la validation, mesure sur la 2de)**"),
            EXP["isotonic"].map(lambda x: f"{x:.5f}" if x < 1 else f"{x:.0f}").to_frame("valeur"))


def _f3():
    imp_ = pd.read_parquet(INTERIM / "imputation.parquet", columns=["payment_id", "invoice_id", "updated_at"]).dropna()
    imp_["updated_at"] = utc(imp_["updated_at"])
    pay = pd.read_parquet(INTERIM / "payment.parquet", columns=["payment_id", "value_date", "booking_date"])
    arrival = pd.Series(utc(pay["booking_date"].fillna(pay["value_date"])).to_numpy(), index=pay["payment_id"])
    first = imp_.groupby("payment_id")["updated_at"].min()
    delay = (first - arrival.reindex(first.index)).dt.total_seconds() / 86400
    ends = {p.name: pd.Timestamp(p.end) + pd.Timedelta(days=1) for p in split.periods}
    pos = DS.to_table(columns=["payment_id", "invoice_id", "period"], filter=pads.field("label") == 1).to_pandas()
    pos = pos.drop_duplicates(["payment_id", "invoice_id"]).merge(
        imp_.groupby(["payment_id", "invoice_id"], as_index=False)["updated_at"].min(), on=["payment_id", "invoice_id"])
    late = {}
    for period in ("train", "validation"):
        p = pos[pos["period"] == period]
        flips = p["updated_at"] >= ends[period]
        late[period] = {"labels positifs": len(p), "imputés après la fin de la période": int(flips.sum()),
                        "part": float(flips.mean()) if len(p) else np.nan}
    return {"delay": delay.describe(percentiles=[0.5, 0.9, 0.95, 0.99]),
            "beyond_purge": float((delay > settings.split.purge_days).mean()),
            "beyond_retention": float((delay > settings.split.retention_days).mean()), "late": pd.DataFrame(late).T}


F3 = cached("f3", _f3) if DS is not None else None
if F3:
    display(Markdown("**Délai entre l'arrivée du paiement et sa première imputation (jours)**"),
            F3["delay"].round(1).to_frame("délai").T)
    display(Markdown(f"Imputés au-delà de la purge ({settings.split.purge_days} j) : **{pct(F3['beyond_purge'], 1)}** ; "
                     f"au-delà de la rétention ({settings.split.retention_days} j) : {pct(F3['beyond_retention'], 1)}."))
    display(Markdown("**Labels qui basculeraient avec un horizon à la fin de leur période**"),
            F3["late"].style.format({"labels positifs": num, "imputés après la fin de la période": num, "part": pct}))"""),
    ("md", r"""### 4.6 F2 : le référentiel IBAN n'est pas daté (leakage de données)

**Théorie.** `party_iban` est visible dès le premier jour du replay, dans son état de fin d'historique. Un IBAN
rattaché au débiteur **après** un rapprochement manuel est donc connu avant le paiement qui l'a révélé, y compris dans
le **test**.

**Mesure, pire cas.** Est compté comme « révélé » tout paiement qui est le premier (au jour près) à venir de cet IBAN
pour ce débiteur. Un IBAN connu dès l'entrée en relation est ainsi compté à tort : c'est un majorant."""),
    ("code", r"""def _f2():
    pay = pd.read_parquet(INTERIM / "payment.parquet", columns=["payment_id", "value_date", "booking_date", "iban_debtor"])
    pay = pd.DataFrame({"payment_id": pay["payment_id"], "day": utc(pay["booking_date"].fillna(pay["value_date"])).dt.normalize(),
                        "iban": pay["iban_debtor"]})
    imp_ = pd.read_parquet(INTERIM / "imputation.parquet", columns=["payment_id", "invoice_id"])
    inv = pd.read_parquet(INTERIM / "invoice.parquet", columns=["invoice_id", "debtor_id"])
    truth = imp_.merge(inv, on="invoice_id")[["payment_id", "debtor_id"]].drop_duplicates()
    pdx = pay[pay["iban"].notna()].merge(truth, on="payment_id")
    links_path = INTERIM / "party_iban.parquet"
    if links_path.exists():
        links = pd.read_parquet(links_path, columns=["role", "party_id", "iban"])
        links = links[links["role"] == "debtor"]
    else:
        links = pd.read_parquet(INTERIM / "debtor.parquet", columns=["party_id", "iban"]).dropna()
    links = links[["iban", "party_id"]].drop_duplicates().rename(columns={"party_id": "debtor_id"}).assign(connu=True)
    pdx = pdx.merge(links, on=["iban", "debtor_id"], how="left")
    pdx["connu"] = pdx["connu"].fillna(False).astype(bool)
    first = pdx.groupby(["iban", "debtor_id"])["day"].transform("min")
    pdx["revele"] = pdx["connu"] & (pdx["day"] == first)
    test = split.period("test")
    in_test = pdx[(pdx["day"] >= pd.Timestamp(test.start)) & (pdx["day"] <= pd.Timestamp(test.end))]
    per_pay = in_test.groupby("payment_id")[["connu", "revele"]].any()
    out = {"test": pd.Series({"paiements imputés avec IBAN": len(per_pay), "IBAN connu du référentiel": per_pay["connu"].mean(),
                              "IBAN révélé par ce paiement (pire cas)": per_pay["revele"].mean()})}
    if EXPORT.exists():
        d = pd.read_parquet(EXPORT, columns=["payment_id", "action", "step", "decision_retenue", "decision_retenue_exacte"])
        auto = d[d["decision_retenue"] & (d["action"] == "auto")].drop_duplicates("payment_id")
        auto = auto.assign(revele=auto["payment_id"].map(per_pay["revele"]).fillna(False).astype(bool),
                           exacte=auto["decision_retenue_exacte"].astype(float))
        out["auto"] = auto.groupby(["step", "revele"]).agg(auto_validations=("payment_id", "size"), précision=("exacte", "mean"))
        out["share"] = float(auto["revele"].mean())
    return out


F2 = cached("f2", _f2)
if F2:
    t = F2["test"]
    display(pd.Series({"paiements imputés avec IBAN (test)": num(t.iloc[0]), "IBAN connu du référentiel": pct(t.iloc[1], 1),
                       "IBAN révélé par ce paiement (pire cas)": pct(t.iloc[2], 1)}).to_frame("valeur"))
    if "auto" in F2:
        display(F2["auto"].rename(index={False: "IBAN non révélé", True: "IBAN révélé (pire cas)"}, level=1)
                .style.format({"auto_validations": num, "précision": pct}))
        display(Markdown(f"**{pct(F2['share'], 1)}** des auto-validations du test reposent peut-être sur un IBAN révélé par le "
                         f"paiement lui-même. Pire cas : taux d'automatisation × {1 - F2['share']:.3f} si on les retire toutes."))"""),
    ("md", r"""### 4.7 F5 et bilan

**F5** : une imputation datée avant l'arrivée de son paiement est déjà dans l'état figé du jour d'arrivée (leakage de
données). Le contrôle qualité du chargement les compte."""),
    ("code", r"""issues_path = REP / "load_issues.csv"
issues = pd.read_csv(issues_path) if issues_path.exists() else pd.DataFrame(columns=["kind", "count"])
F5 = int(issues.loc[issues["kind"] == "avant_connaissance_du_paiement", "count"].sum())
n_imp = pq.ParquetFile(INTERIM / "imputation.parquet").metadata.num_rows
display(Markdown(f"F5 : **{num(F5)}** imputations datées avant l'arrivée de leur paiement, sur {num(n_imp)}."))

rows = []
if EXP:
    p = EXP["pass1"]["precision_at_1"]
    rows.append(("Stacking : label des paires dans les folds", "aucun",
                 f"OOF {pct(p['train, OOF'], 1)} ≈ validation {pct(p['validation (pass 1 complète)'], 1)}, in-sample {pct(p['train, in-sample'], 1)}"))
    d = EXP["pass2"]["precision_at_1"]
    rows.append(("F4 · CV non temporel", "mécanisme réel, apprentissage seulement",
                 f"pass 2 walk-forward {pct(d['pass 2 · walk-forward'], 1)} contre {pct(d['pass 2 · GroupKFold, mêmes paires que le walk-forward'], 1)}"))
    t4 = EXP["t4"]
    rows.append(("Itérations des folds", "un entier, apprentissage seulement",
                 f"shuffle de la validation : {t4.iloc[0, 0]:.0f} → {t4.iloc[2, 0]:.0f} itérations"))
    iso = EXP["isotonic"]
    rows.append(("F1 · validation réutilisée", "métriques de validation optimistes",
                 f"isotonique : log-loss {iso.iloc[1]:.4f} contre {iso.iloc[0]:.4f} brut"))
if CI is not None:
    rows.append(("F1 · winner's curse sur τ_high", "mesuré sur le test", f"borne basse 95 % : {pct(CI.loc['total', 'borne basse 95 %'])}"))
if PC is not None:
    at = PC["at_tau"].set_index(["type", "jeu"])["précision"]
    if ("single", "test") in at.index:
        rows.append(("F1 · τ_high transposé au test", "calibration par la précision",
                     f"single : {pct(at[('single', 'validation')])} en validation, {pct(at[('single', 'test')])} en test"))
if F3:
    rows.append(("F3 · labels sans horizon", "apprentissage seulement",
                 f"délai P95 {F3['delay']['95%']:.1f} j ; {pct(F3['late'].loc['validation', 'part'], 2)} des labels de validation"))
if F2:
    rows.append(("F2 · IBAN non datés", "données : touche le test",
                 f"pire cas : {pct(F2.get('share', np.nan), 1)} des auto-validations"))
rows.append(("F5 · imputation avant le paiement", "données : touche le test", f"{num(F5)} cas"))
display(Markdown("### Bilan"), pd.DataFrame(rows, columns=["leakage", "portée", "mesure"]).style.hide(axis="index"))"""),

    # --- 5. Évolutions ---------------------------------------------------------------------------------------
    ("md", r"""## 5. Évolutions proposées

| # | Chantier | Mesure de succès |
|---|---|---|
| 1 | Labels à horizon (`updated_at < cutoff`), paiements non soldés écartés, purge ≥ P95 du délai (F3) | aucun label qui bascule au truncation test |
| 2 | Walk-forward mensuel à la place du GroupKFold pour les scores OOF (F4) | pass 2 au moins égale sur le test |
| 3 | Nombre d'itérations des folds fixé sans la validation (early stopping interne au train) | shuffle de la validation sans effet sur les scores OOF |
| 4 | Validation coupée en V1 (early stopping) / V2 (seuils), τ_high sur borne de Clopper-Pearson (F1) | borne basse ≥ 99,5 % sur le test |
| 5 | Dater le référentiel IBAN : historique, sinon date du premier paiement imputé (F2) | écart de taux sous un seuil convenu |
| 6 | F5 bloquant au chargement ; vocabulaires de catégories sur le train (F7) ; KPI oracle retiré de l'affichage (F6) | contrôles verts |
| 7 | Test anti-leakage du training en intégration continue | test vert |
| 8 | Recall des candidats : diagnostic des factures manquées par cause, élargissement ciblé | recall en hausse, précision tenue |
| 9 | Évaluation finale sur une période jamais vue, configuration figée | taux et précision stables |

**Décisions à prendre aujourd'hui**
- l'ordre du plan ;
- l'accès à l'historique du référentiel IBAN (F2) ;
- la période d'évaluation finale, jamais vue ;
- le format réel des client files, si l'on veut les exploiter davantage."""),
]


def build() -> dict:
    cells = []
    for i, (kind, text) in enumerate(CELLS):
        lines = text.splitlines(keepends=True)
        cell = {"id": f"cell-{i:02d}", "metadata": {}, "source": lines}
        if kind == "md":
            cells.append({"cell_type": "markdown", **cell})
        else:
            cells.append({"cell_type": "code", "execution_count": None, "outputs": [], **cell})
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}}}


def main() -> None:
    text = json.dumps(build(), indent=1, ensure_ascii=False)
    for out in (ROOT / "notebooks" / "review_ml.ipynb", ROOT / "single" / "review_ml.ipynb"):
        out.parent.mkdir(exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(out)


if __name__ == "__main__":
    main()
