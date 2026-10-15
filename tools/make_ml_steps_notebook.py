"""Génère notebooks/ml_pas_a_pas.ipynb : l'étape ML déroulée pas à pas (passe 1, plis hors échantillon,
features de compétition, passe 2, calibration), avec exploration des sorties à chaque étape et un jeu de
test synthétique généré à graine fixe.

    python tools/make_ml_steps_notebook.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CELLS: list[tuple[str, str]] = [
    ("md", """# L'étape ML pas à pas : passe 1, plis, features de compétition, passe 2

Ce notebook refait **à la main** ce que fait `PairModel.fit` (`src/reconcile_ml/model.py`), une étape à la
fois, et s'arrête après chaque étape pour regarder ce qu'elle produit et ce qu'elle vaut :

| § | Étape | Ce qu'on regarde |
|---|---|---|
| 1 | Jeu de test synthétique | tables, périodes |
| 2 | Jeu de paires (candidats + features) | volume, taux de positifs, rappel des candidats |
| 3 | **Passe 1** | courbes d'apprentissage, AUC, precision@1, MRR, importance, pourquoi la suite |
| 4 | **Plis hors échantillon** | performance de chaque pli, ce que chaque pli a vu (fuite F4), variante temporelle, **tests de fuite train / validation** |
| 5 | **Features de compétition** | construction sur un paiement, distributions train / validation |
| 6 | **Passe 2 et calibration** | gain sur la passe 1, effet de F4, calibration isotonique |
| 7 | Contrôle de fidélité | les étapes manuelles redonnent exactement `PairModel.fit` |
| 8 | **Preuves empiriques de (non-)fuite** | troncature de tout l'apprentissage après la validation, audit des features une à une |

Les étapes 4 à 6 mesurent aussi la fuite F4 décrite dans `docs/fuites_ml.md`.

**Ce que le notebook ne fait pas** : les ensembles (1↔n), la décision par seuils et le backtest. Ils viennent
après le modèle (`src/reconcile_ml/decision.py`, `project.backtest`).

**Données** : par défaut, un jeu **synthétique** généré à graine fixe (§ 1), conformément au brief : jamais de
tests sur les données réelles. `DATASET = "real"` permet de l'ouvrir sur les données réelles chargées par
`project.load()`.

**Temps** : environ 5 à 15 minutes au premier lancement avec 40 000 paiements (génération + rejeu
quotidien). Le jeu de paires est ensuite mis en cache : les lancements suivants partent du § 3."""),
    ("code", """import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

import altair as alt
import lightgbm as lgb
import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, roc_auc_score

from src import progress
from src.api import Project
from src.load.interim import load_interim
from src.reconcile_ml.features import BASE, CATEGORICAL, COMPETITION, FAMILIES, active_features
from src.reconcile_ml.model import GROUP, PairModel, _params, calibration_table, competition_features, pair_metrics
from src.reconcile_ml.pipeline import DatasetRecorder, candidate_recall, label_pairs
from src.evaluation.metrics import ground_truth
from src.timeline.loop import run_replay
from src.timeline.split import assign_period, compute_split
from src.timeline.state import LedgerState

pd.set_option("display.max_columns", 60)
alt.data_transformers.disable_max_rows()
progress.use_notebook()"""),
    ("md", """### Paramètres

Tout se règle ici. Les autres réglages (fenêtres de candidats, hyperparamètres LightGBM, périodes) viennent de
`config/settings.yaml`, comme pour `project.train()`."""),
    ("code", """DATASET = "synthetic"     # "real" : données chargées par project.load() depuis config/schema.yaml
N_PAYMENTS = 40_000       # volume du jeu de test synthétique
SEED = 42                 # graine du générateur : même graine → même jeu
PAYMENT_SAMPLE = 1.0      # part des paiements du résiduel gardés (config : 0,5) ; 1.0 sur un petit jeu
FOLDS = 3                 # plis hors échantillon, comme PairModel.fit
REBUILD = False           # True : régénère le jeu de test et le jeu de paires
CHECK_FIDELITY = True     # § 7 : réentraîne PairModel.fit pour vérifier l'identité (double le temps du § 3-6)"""),

    # --- 1. Jeu de test -------------------------------------------------------------------------------------
    ("md", """## 1. Le jeu de test

Le générateur `src/synthetic/generate.py` produit un an d'historique cohérent : cédants, débiteurs, contrats,
factures, paiements (libellés bruités, paiements groupés, partiels, escomptes, orphelins, doublons), client
files et **imputations** (la vérité). Les fichiers CSV sources sont écrits dans
`data/synthetic/<N_PAYMENTS>_seed<SEED>/`. On peut les ouvrir, les modifier, ou les remplacer par d'autres CSV
au même format (`config/schema.synthetic.yaml`).

`project.load()` les charge comme des données réelles : normalisation, journal d'événements, tables parquet
dans `data/interim/synthetic/`."""),
    ("code", """project = Project(dataset=DATASET, log=print)
meta = project.meta()
expected = f"synthetic/{N_PAYMENTS}_seed{SEED}"
if DATASET == "synthetic" and (REBUILD or meta is None or meta.get("source") != expected):
    project.load(n_payments=N_PAYMENTS, seed=SEED, regenerate=REBUILD)
    meta = project.meta()
print("source :", meta["source"], "· journal", meta["journal_sha256"][:12])

settings, rules = project.settings, project.rules
settings.reconcile_ml.training.payment_sample = PAYMENT_SAMPLE
data, journal, meta = load_interim(project.interim_dir)
display(pd.DataFrame({"lignes": {k: len(v) for k, v in data.tables.items()}}).T)"""),
    ("code", """def head(table: str, cols: list[str], n: int = 5) -> pd.DataFrame:
    df = data.tables[table]
    return df[[c for c in cols if c in df.columns]].head(n)


display(Markdown("**Paiements**"), head("payment", ["payment_id", "value_date", "amount", "label", "iban_debtor"]))
display(Markdown("**Factures**"), head("invoice", ["invoice_id", "client_reference", "debtor_id", "creation_date",
                                                     "due_date", "initial_amount"]))
display(Markdown("**Imputations (la vérité)**"), data.tables["imputation"].head(5))"""),
    ("md", """### Périodes

Découpage temporel du brief (§ 4.2) : train, validation, test, séparés par une purge. Le test n'est **jamais**
touché ici : le modèle n'apprend que sur le train, et la validation sert à l'arrêt précoce et à la
calibration."""),
    ("code", """state = LedgerState(data, journal, settings.reconcile_ml.features.behavioral_window_days)
first, last = state.payment_day_range()
split = compute_split(settings.split, first.date(), last.date())
train_p, valid_p = split.period("train"), split.period("validation")
periods = pd.DataFrame([{"période": p.name, "début": pd.Timestamp(p.start), "fin": pd.Timestamp(p.end) + pd.Timedelta(days=1),
                         "jours": p.days} for p in split.periods])
display(periods)
alt.Chart(periods).mark_bar(height=28).encode(
    x=alt.X("début:T", title=None), x2="fin:T", y=alt.Y("période:N", sort=None, title=None),
    color=alt.Color("période:N", legend=None), tooltip=["période", "début", "fin", "jours"]
).properties(width=700, height=110, title="Train → validation → test (les blancs sont les purges)")"""),

    # --- 2. Jeu de paires -----------------------------------------------------------------------------------
    ("md", """## 2. Le jeu de paires : un problème de classement déguisé en classification

Le ML ne voit pas des paiements, mais des **paires (paiement, facture candidate)**. `DatasetRecorder` rejoue la
boucle quotidienne du train et de la validation (`run_replay`, le même code que la production) :

```
jour D ─► état figé à la veille (LedgerState, as_of = D)
        ─► allocation → règles (étape 4) → résiduel
        ─► pour chaque paiement du résiduel arrivé le jour D :
              candidats (factures ouvertes à D : débiteur, référence, montant, client file)
              features de chaque paire, calculées sur l'état à D
```

Puis, **dans une seconde passe séparée**, chaque paire reçoit son étiquette : 1 si la facture a réellement été
imputée à ce paiement (`label_pairs`).

```
paiement P1 ─┬─ facture A   features…   label 1
             ├─ facture B   features…   label 0
             └─ facture C   features…   label 0      ← un « groupe » = un paiement
paiement P2 ─┬─ facture D   features…   label 0
             └─ facture E   features…   label 0      ← aucun positif : vraie facture absente, ou jamais imputé
```

Le modèle est un **classifieur binaire de paires** (pointwise), mais la question métier est un **classement par
paiement** : quelle facture (ou quel ensemble) est la bonne parmi les candidats ? D'où les métriques par groupe
utilisées partout : precision@1 (le meilleur candidat est-il un vrai ?) et MRR (rang moyen inverse du premier
vrai)."""),
    ("code", """cache = project.interim_dir / "ml_pas_a_pas" / f"pairs_{meta['journal_sha256'][:12]}_s{PAYMENT_SAMPLE}.parquet"
if cache.exists() and not REBUILD:
    ds = pd.read_parquet(cache)
    print(f"jeu de paires repris du cache : {cache}")
else:
    recorder = DatasetRecorder(state, settings, rules)
    run_replay(state, recorder, train_p.start, valid_p.end, settings.split.retention_days,
               label="jeu de paires (rejeu train + validation)")
    ds = recorder.dataset()
    ds["label"] = label_pairs(ds, data.tables["imputation"])       # seconde passe, séparée du rejeu
    ds[GROUP] = ds["payment_id"]
    ds["period"] = assign_period(ds["day"], split).to_numpy()
    ds["month"] = pd.to_datetime(ds["day"]).dt.strftime("%Y-%m")
    cache.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(cache, index=False)
    print(f"jeu de paires écrit : {cache}")

FEATURES = active_features(settings.reconcile_ml)
FAMILY_OF = {f: "base" for f in BASE} | {f: fam for fam, cols in FAMILIES.items() for f in cols}
train = ds[ds["period"] == "train"].reset_index(drop=True)
valid = ds[ds["period"] == "validation"].reset_index(drop=True)
truth = ground_truth(data.tables["imputation"])
print(f"{len(FEATURES)} features · train {len(train):,} paires · validation {len(valid):,} paires".replace(",", " "))"""),
    ("code", """def group_summary(df: pd.DataFrame) -> dict:
    g = df.groupby(GROUP)
    pos = g["label"].sum()
    return {"paiements": df[GROUP].nunique(), "paires": len(df), "candidats / paiement (médiane)": g.size().median(),
            "taux de positifs (paires)": df["label"].mean(), "paiements avec ≥ 1 positif": (pos > 0).mean(),
            "paiements avec ≥ 2 positifs (1↔n)": (pos > 1).mean()}

summary = pd.DataFrame({"train": group_summary(train), "validation": group_summary(valid)})
summary.loc["rappel des candidats"] = [candidate_recall(train, truth), candidate_recall(valid, truth)]
display(summary)

sizes = pd.concat([train, valid]).groupby([GROUP, "period"]).size().rename("n").reset_index()
sizes["tranche"] = pd.cut(sizes["n"], [0, 1, 2, 5, 10, 20, 50, 1_000], labels=["1", "2", "3-5", "6-10", "11-20", "21-50", "> 50"])
counts = sizes.groupby(["period", "tranche"], observed=True).size().rename("paiements").reset_index()
alt.Chart(counts).mark_bar().encode(
    x=alt.X("tranche:O", title="candidats par paiement"), y="paiements:Q", color="period:N", xOffset="period:N",
    tooltip=["period", "tranche", "paiements"]).properties(width=500, height=220, title="Taille des groupes")"""),
    ("md", """**À lire.**
- **Taux de positifs** : très déséquilibré (quelques % des paires). Pas de rééchantillonnage : le brief garde tous
  les négatifs, car c'est la concurrence entre candidats d'un même paiement qui compte.
- **Paiements sans positif** : la vraie facture n'est pas candidate (limite du rappel), ou le paiement n'a jamais
  été imputé (orphelin, doublon). Le modèle doit apprendre à n'y donner aucun score élevé.
- **Rappel des candidats** : plafond du ML. Un paiement dont la vraie facture manque ne peut pas être rapproché."""),
    ("code", """def pick_payment(df: pd.DataFrame, n_min: int, n_max: int) -> str:
    \"\"\"Un paiement à un seul positif et n_min à n_max candidats (sinon le plus proche).\"\"\"
    g = df.groupby(GROUP)
    size, pos = g[GROUP].transform("size"), g["label"].transform("sum")
    ok = df.loc[pos.eq(1) & size.between(n_min, n_max), GROUP]
    return ok.iloc[0] if len(ok) else df.loc[pos.ge(1), GROUP].iloc[0]


example_id = pick_payment(valid, 4, 8)
cols = ["invoice_id", "label", "payment_amount", "balance", "amount_diff", "amount_exact", "days_to_due",
        "ref_key_in_label", "src_reference", "src_amount", "alloc_score", "alloc_is_firm_debtor"]
display(Markdown(f"**Un paiement de la validation et ses candidats** (`{example_id}`)"),
        valid.loc[valid[GROUP] == example_id, [c for c in cols if c in valid.columns]])"""),

    # --- 3. Passe 1 -----------------------------------------------------------------------------------------
    ("md", """## 3. Passe 1 : un score par paire, sans regarder les concurrents

```
X_train (features de la paire)  ──► LightGBM binaire ──► s1 = P(label = 1 | paire)
                                      arrêt précoce sur la validation (30 itérations sans gain)
```

Mêmes hyperparamètres que `_train` (`_params(cfg)` : taux 0,05, 63 feuilles, 50 observations minimum par
feuille, graine fixe, mode déterministe). Les arbres n'ont besoin d'**aucune normalisation des features** : ils
coupent sur des seuils, et sont invariants à toute transformation monotone (échelle, log). Les variables
catégorielles (`iban_route`, `channel`, `bankroll_code`, `market`, `product`, `recourse`) sont données comme
telles à LightGBM.

On enregistre la perte sur le train et sur la validation à chaque itération pour voir l'apprentissage."""),
    ("code", """cfg = settings.reconcile_ml.training
PARAMS = _params(cfg)
CATS = [c for c in CATEGORICAL if c in FEATURES]
X, y = train[FEATURES], train["label"].to_numpy()
Xv, yv = valid[FEATURES], valid["label"].to_numpy()


def lgb_dataset(Xd, yd, reference=None):
    return lgb.Dataset(Xd, label=yd, categorical_feature=[c for c in CATS if c in Xd.columns], free_raw_data=False,
                       reference=reference)


def learning_curve(history: dict, best: int, title: str):
    rows = [{"itération": i + 1, "jeu": name, "logloss": v}
            for name, h in history.items() for i, v in enumerate(h["binary_logloss"])]
    curve = alt.Chart(pd.DataFrame(rows)).mark_line().encode(x="itération:Q", y=alt.Y("logloss:Q", scale=alt.Scale(zero=False)),
                                                             color="jeu:N")
    rule = alt.Chart(pd.DataFrame({"x": [best]})).mark_rule(strokeDash=[4, 4]).encode(x="x:Q")
    return (curve + rule).properties(width=600, height=250, title=f"{title} (pointillé : meilleure itération {best})")


d_train, history1 = lgb_dataset(X, y), {}
d_valid = lgb_dataset(Xv, yv, reference=d_train)
pass1 = lgb.train(PARAMS, d_train, num_boost_round=cfg.num_boost_round, valid_sets=[d_train, d_valid],
                  valid_names=["train", "validation"],
                  callbacks=[lgb.early_stopping(30, verbose=False), lgb.record_evaluation(history1)])
learning_curve(history1, pass1.best_iteration, "Passe 1 : perte par itération")"""),
    ("code", """def scores_table(named: dict[str, tuple[pd.DataFrame, np.ndarray]]) -> pd.DataFrame:
    rows = {}
    for name, (df, s) in named.items():
        m = pair_metrics(df, s)
        m["logloss"] = log_loss(df["label"], np.clip(s, 1e-7, 1 - 1e-7), labels=[0, 1])
        rows[name] = m
    return pd.DataFrame(rows).T[["auc", "precision_at_1", "mrr", "logloss", "groupes_avec_positif"]]


s1_train_in = pass1.predict(X)            # en échantillon : le modèle a appris sur ces paires
s1_valid = pass1.predict(Xv)
display(scores_table({"passe 1 · train (en échantillon)": (train, s1_train_in), "passe 1 · validation": (valid, s1_valid)}))"""),
    ("md", """**À lire.**
- `auc` mesure la séparation des paires vraies et fausses **toutes confondues** ; `precision_at_1` et `mrr` mesurent
  le **classement au sein de chaque paiement**, ce qui compte pour la décision.
- L'écart train / validation est l'**optimisme en échantillon** : sur le train, le modèle a vu les étiquettes.
  Cet écart sera le cœur du § 4."""),
    ("code", """def binned(df_scores: pd.DataFrame, value: str, by: str, bins) -> pd.DataFrame:
    out = df_scores.assign(tranche=pd.cut(df_scores[value], bins, include_lowest=True))
    out = out.groupby([by, "tranche"], observed=True).size().rename("n").reset_index()
    out["part"] = out["n"] / out.groupby(by)["n"].transform("sum")
    out["tranche"] = out["tranche"].map(lambda iv: round(iv.mid, 3)).astype(float)
    return out


dist = binned(pd.DataFrame({"s1": s1_valid, "étiquette": np.where(yv == 1, "vraie paire", "fausse paire")}),
              "s1", "étiquette", np.linspace(0, 1, 26))
hist = alt.Chart(dist).mark_line(point=True).encode(
    x=alt.X("tranche:Q", title="score de passe 1"),
    y=alt.Y("part:Q", title="part des paires de l'étiquette", scale=alt.Scale(type="symlog")), color="étiquette:N")

imp = pd.DataFrame({"feature": pass1.feature_name(), "gain": pass1.feature_importance("gain")})
imp["famille"] = imp["feature"].map(FAMILY_OF)
imp = imp.sort_values("gain", ascending=False).head(20)
bars = alt.Chart(imp).mark_bar().encode(x=alt.X("gain:Q"), y=alt.Y("feature:N", sort="-x", title=None), color="famille:N",
                                        tooltip=["feature", "famille", "gain"])
(hist.properties(width=380, height=260, title="Scores de passe 1 en validation")
 | bars.properties(width=320, height=380, title="Importance (gain), 20 premières"))"""),
    ("code", """ex = valid.loc[valid[GROUP] == example_id, ["invoice_id", "label", "payment_amount", "balance"]].assign(
    s1=s1_valid[valid[GROUP].to_numpy() == example_id]).sort_values("s1", ascending=False)
display(Markdown(f"**Le paiement exemple, noté par la passe 1**"), ex)"""),
    ("md", """### Pourquoi une passe 2 ? Le score isolé ignore la concurrence

La passe 1 note chaque paire **seule**. Or 0,90 n'a pas le même sens si le deuxième candidat est à 0,10 ou à
0,88 :

```
Paiement P :  facture A 0.90   facture B 0.10     → A est très probablement la bonne
Paiement Q :  facture C 0.90   facture D 0.88     → C ou D : on ne sait pas
```

Le graphique ci-dessous le vérifie sur la validation : à **score du meilleur candidat comparable**, le meilleur
candidat est bien plus souvent juste quand la **marge au deuxième** est grande. La marge porte une information que
le score seul n'a pas. La passe 2 la donne au modèle sous forme de features (rang, marge, meilleur concurrent) :
c'est un empilement (« stacking »), un pas vers un modèle de classement (« learning to rank »)."""),
    ("code", """def top1_frame(df: pd.DataFrame, s: np.ndarray) -> pd.DataFrame:
    \"\"\"Par paiement ayant ≥ 2 candidats et ≥ 1 positif : score du meilleur, marge au deuxième, meilleur juste ?\"\"\"
    d = pd.DataFrame({"g": df[GROUP].to_numpy(), "y": df["label"].to_numpy(), "s": s})
    d = d[d.groupby("g")["y"].transform("sum") > 0]
    d = d.sort_values(["g", "s"], ascending=[True, False], kind="mergesort")
    d["r"] = d.groupby("g").cumcount()
    top = d[d["r"] == 0].set_index("g")
    second = d[d["r"] == 1].set_index("g")["s"]
    top = top.loc[top.index.intersection(second.index)]
    return pd.DataFrame({"best": top["s"], "margin": top["s"] - second.loc[top.index], "juste": top["y"] == 1})


t1 = top1_frame(valid, s1_valid)
BEST = ["< 0,5", "0,5 – 0,8", "0,8 – 0,95", "> 0,95"]
MARGIN = ["< 0,05", "0,05 – 0,2", "0,2 – 0,5", "> 0,5"]
t1["score du meilleur"] = pd.cut(t1["best"], [0, 0.5, 0.8, 0.95, 1.0], labels=BEST, include_lowest=True).astype(str)
t1["marge au 2e"] = pd.cut(t1["margin"], [0, 0.05, 0.2, 0.5, 1.0], labels=MARGIN, include_lowest=True).astype(str)
grid = t1.groupby(["score du meilleur", "marge au 2e"], observed=True).agg(paiements=("juste", "size"),
                                                                           precision_at_1=("juste", "mean")).reset_index()
alt.Chart(grid).mark_rect().encode(
    x=alt.X("marge au 2e:O", sort=MARGIN), y=alt.Y("score du meilleur:O", sort=BEST[::-1]),
    color=alt.Color("precision_at_1:Q", scale=alt.Scale(domain=[0, 1], scheme="blues")),
    tooltip=["score du meilleur", "marge au 2e", "paiements", alt.Tooltip("precision_at_1:Q", format=".1%")]
).properties(width=380, height=220, title="Validation : precision@1 selon le score du meilleur et sa marge")"""),
    ("md", """### Et pourquoi pas simplement les scores de la passe 1 sur le train ?

Pour entraîner la passe 2, il faut ces features de compétition **sur chaque paire du train**, donc des scores de
passe 1 sur le train. Si on prend les scores du modèle de passe 1 lui-même (« en échantillon »), il note des paires
dont il a appris l'étiquette : les marges sont trop nettes, et le meilleur candidat presque toujours juste. La
passe 2 apprendrait « marge moyenne = sûr », ce qui est faux sur des paiements nouveaux.

Le graphique compare la distribution de la marge du meilleur candidat : train en échantillon contre validation."""),
    ("code", """def margin_dist(named: dict[str, tuple[pd.DataFrame, np.ndarray]]) -> pd.DataFrame:
    parts = []
    for name, (df, s) in named.items():
        t = top1_frame(df, s)
        parts.append(pd.DataFrame({"source": name, "margin": t["margin"].to_numpy(), "juste": t["juste"].to_numpy()}))
    return pd.concat(parts, ignore_index=True)


def margin_charts(md: pd.DataFrame, title: str):
    bins = np.linspace(0, 1, 21)
    d = binned(md, "margin", "source", bins)
    hist = alt.Chart(d).mark_line(point=True).encode(x=alt.X("tranche:Q", title="marge du meilleur au 2e"),
                                                     y=alt.Y("part:Q", title="part des paiements"), color="source:N")
    acc = md.assign(tranche=pd.cut(md["margin"], bins, include_lowest=True)).groupby(["source", "tranche"], observed=True)
    acc = acc.agg(n=("juste", "size"), precision_at_1=("juste", "mean")).reset_index()
    acc = acc[acc["n"] >= 20].assign(tranche=lambda a: a["tranche"].map(lambda iv: round(iv.mid, 3)).astype(float))
    line = alt.Chart(acc).mark_line(point=True).encode(x=alt.X("tranche:Q", title="marge du meilleur au 2e"),
                                                       y=alt.Y("precision_at_1:Q", scale=alt.Scale(zero=False)),
                                                       color="source:N", tooltip=["source", "tranche", "n", "precision_at_1"])
    return (hist.properties(width=360, height=240, title="Distribution de la marge")
            | line.properties(width=360, height=240, title="precision@1 selon la marge")).properties(title=title)


margin_charts(margin_dist({"train · en échantillon": (train, s1_train_in), "validation": (valid, s1_valid)}),
              "En échantillon, les marges du train ne ressemblent pas à celles de la validation")"""),

    # --- 4. Plis --------------------------------------------------------------------------------------------
    ("md", """## 4. Les plis hors échantillon

Solution du code (`PairModel.fit`, `model.py:118-125`) : couper le train en 3 plis **par paiement**, et noter
chaque pli avec un modèle entraîné sur les deux autres. Chaque paire du train reçoit ainsi un score d'un modèle qui
n'a **jamais vu son paiement**.

```
pli d'un paiement = hash(payment_id) mod 3

            entraîné sur        note
modèle 0 :  plis 1 + 2    ──►   pli 0
modèle 1 :  plis 0 + 2    ──►   pli 1        →  oof = score hors échantillon de chaque paire du train
modèle 2 :  plis 0 + 1    ──►   pli 2
```

Chaque modèle de pli a le **même nombre d'itérations** que la meilleure itération de la passe 1
(`max(best_iteration, 10)`) et pas d'arrêt précoce."""),
    ("code", """fold = (pd.util.hash_array(train[GROUP].astype(str).to_numpy()) % FOLDS).astype(int)
ROUNDS = max(pass1.best_iteration, 10)
oof = np.zeros(len(train))
fold_models, rows = [], []
for k in range(FOLDS):
    fit_rows, held = fold != k, fold == k
    m = lgb.train({**PARAMS}, lgb_dataset(X[fit_rows], y[fit_rows]), num_boost_round=ROUNDS)
    fold_models.append(m)
    oof[held] = m.predict(X[held])
    for name, df, s in (("appris (en échantillon)", train[fit_rows], m.predict(X[fit_rows])),
                        ("pli noté (hors échantillon)", train[held], oof[held]),
                        ("validation", valid, m.predict(Xv))):
        rows.append({"modèle": f"pli {k}", "jeu": name, "paiements": df[GROUP].nunique(), **pair_metrics(df, s)})
folds_perf = pd.DataFrame(rows)
display(Markdown("**precision@1 de chaque modèle de pli**"),
        folds_perf.pivot(index="modèle", columns="jeu", values="precision_at_1").round(3))
display(scores_table({"passe 1 · train en échantillon": (train, s1_train_in), "plis · train hors échantillon (oof)": (train, oof),
                      "passe 1 · validation": (valid, s1_valid)}))"""),
    ("md", """**À lire.**
- Chaque modèle de pli est **meilleur sur ce qu'il a appris** que sur le pli qu'il note : c'est l'optimisme que les
  plis retirent.
- Les scores `oof` du train devraient ressembler aux scores de la passe 1 sur la validation. S'ils restent
  meilleurs, il reste une fuite : c'est la question du § 4.1."""),
    ("md", """### 4.1 Ce que chaque pli a vu : la fuite F4

Le pli dépend du hachage du `payment_id`, **pas de la date**. Chaque pli contient donc des paiements de tous les
mois, et le modèle qui note un paiement de février a appris sur des paiements de mars à août : il a vu **le futur**,
étiquettes comprises (imputations prononcées plus tard), et souvent les mêmes débiteurs.

Le premier graphique montre la répartition des plis par mois. Le second montre, pour chaque mois noté, de quels
mois venaient les paiements d'apprentissage du modèle qui l'a noté. Tout ce qui est **à droite de la diagonale**
est du futur."""),
    ("code", """months = sorted(train["month"].unique())
pay = train.drop_duplicates(GROUP)[[GROUP, "month"]].assign(pli=fold[~train.duplicated(GROUP).to_numpy()].astype(str))
by_month = pay.groupby(["month", "pli"]).size().rename("paiements").reset_index()
stack = alt.Chart(by_month).mark_bar().encode(x=alt.X("month:O", title=None), y="paiements:Q", color="pli:N",
                                              tooltip=["month", "pli", "paiements"])


def seen_matrix(shares) -> pd.DataFrame:
    \"\"\"Pour chaque mois noté : part des paiements d'apprentissage venant de chaque mois.\"\"\"
    return pd.DataFrame([{"mois noté": scored, "mois appris": m, "part": float(share.get(m, 0.0))}
                         for scored, share in shares for m in months])


learnt_by_fold = {k: pay.loc[pay["pli"] != k, "month"].value_counts(normalize=True) for k in pay["pli"].unique()}
random_seen = []
for m_scored in months:
    weights = pay.loc[pay["month"] == m_scored, "pli"].value_counts(normalize=True)
    random_seen.append((m_scored, sum(w * learnt_by_fold[k] for k, w in weights.items())))
seen = seen_matrix(random_seen)
seen["futur"] = seen["mois appris"] > seen["mois noté"]


def heat(df, title):
    base = alt.Chart(df).encode(x=alt.X("mois appris:O"), y=alt.Y("mois noté:O"))
    rect = base.mark_rect().encode(color=alt.Color("part:Q", scale=alt.Scale(scheme="oranges"), title="part"),
                                   tooltip=["mois noté", "mois appris", alt.Tooltip("part:Q", format=".0%")])
    mark = base.transform_filter("datum.futur && datum.part > 0").mark_text(text="F", color="black")
    return (rect + mark).properties(width=300, height=300, title=title)


(stack.properties(width=300, height=300, title="Plis par mois : chaque pli couvre tout le train")
 | heat(seen, "Plis aléatoires : mois appris par mois noté (F = futur)"))"""),
    ("md", """### 4.2 La variante sans fuite : avancer dans le temps (walk-forward)

Pour noter le mois *m*, on entraîne un modèle **uniquement sur le passé**, comme la boucle quotidienne le ferait en
production :

```
         jan   fév   mar   avr   mai   juin  juil  août
M_fév :  ███                                              → note fév
M_mar :  █████████                                        → note mar
M_avr :  ███████████████                                  → note avr
  …
```

Deux règles, pour qu'aucune information postérieure au début du mois Dₖ n'entre :
- paires d'apprentissage : paiements arrivés avant `Dₖ − purge` ;
- étiquettes : seules les imputations d'horodatage **antérieur à Dₖ** comptent (F3). Un paiement sans imputation
  connue à Dₖ n'est gardé que s'il est assez ancien (`retention_days`) pour être considéré comme jamais imputé ;
  sinon son issue est inconnue à Dₖ, et il est écarté.

Le premier mois n'a pas de passé : il n'a pas de score, et il est exclu de la passe 2 de cette variante. Même nombre
d'itérations que les plis, pour comparer à armes égales."""),
    ("code", """imputations = data.tables["imputation"][["payment_id", "invoice_id", "updated_at"]].dropna()
purge = pd.Timedelta(days=settings.split.purge_days)
retention = pd.Timedelta(days=settings.split.retention_days)
day = pd.to_datetime(train["day"])
month_start = day.dt.to_period("M").dt.start_time
wf = np.full(len(train), np.nan)
wf_log, wf_seen = [], []
for d_k in sorted(month_start.unique())[1:]:
    seen_rows = (day < d_k - purge).to_numpy()
    known = imputations[imputations["updated_at"] < d_k]
    lab = label_pairs(train.loc[seen_rows, ["payment_id", "invoice_id"]], known)
    outcome_known = (train.loc[seen_rows, "payment_id"].isin(known["payment_id"])
                     | (day[seen_rows] < d_k - retention)).to_numpy()
    rows_fit = np.flatnonzero(seen_rows)[outcome_known]
    target = (month_start == d_k).to_numpy()
    n_pos = int(lab[outcome_known].sum())
    if n_pos < 50:
        wf_log.append({"mois noté": d_k.strftime("%Y-%m"), "paires apprises": len(rows_fit), "positifs": n_pos, "statut": "trop peu de positifs"})
        continue
    m = lgb.train({**PARAMS}, lgb_dataset(X.iloc[rows_fit], lab[outcome_known]), num_boost_round=ROUNDS)
    wf[target] = m.predict(X[target])
    wf_log.append({"mois noté": d_k.strftime("%Y-%m"), "paires apprises": len(rows_fit), "positifs": n_pos, "statut": "noté",
                   "étiquettes positives perdues (imputées après Dₖ)": int(train.loc[rows_fit, "label"].sum()) - n_pos})
    wf_seen.append((d_k.strftime("%Y-%m"), train.loc[rows_fit].drop_duplicates(GROUP)["month"].value_counts(normalize=True)))
display(pd.DataFrame(wf_log))
seen_wf = seen_matrix(wf_seen)
seen_wf["futur"] = seen_wf["mois appris"] > seen_wf["mois noté"]
heat(seen, "Plis aléatoires (F = futur)") | heat(seen_wf, "Walk-forward : passé seulement")"""),
    ("code", """has_wf = ~np.isnan(wf)
same_rows = train[has_wf]
display(Markdown("**Mêmes paires du train (hors premier mois), trois façons de les noter, et la validation pour référence**"))
display(scores_table({"passe 1 · en échantillon": (same_rows, s1_train_in[has_wf]),
                      "plis aléatoires (code actuel)": (same_rows, oof[has_wf]),
                      "walk-forward (sans futur)": (same_rows, wf[has_wf]),
                      "passe 1 · validation": (valid, s1_valid)}))
margin_charts(margin_dist({"plis aléatoires": (same_rows, oof[has_wf]), "walk-forward": (same_rows, wf[has_wf]),
                           "validation": (valid, s1_valid)}),
              "Quelle notation du train ressemble le plus à ce que la passe 2 verra en validation ?")"""),
    ("md", """**À lire : c'est la mesure de F4.**
- Si les plis aléatoires donnent sur le train une precision@1 et des marges **plus élevées** que le walk-forward, la
  différence est ce qu'ils doivent au futur.
- La notation qui ressemble le plus à la validation est celle que la passe 2 devrait apprendre : c'est ce qu'elle
  recevra en validation, en test et en production.
- Sur le synthétique, les débiteurs ont des habitudes stables et le générateur ne dérive pas dans le temps : l'écart
  y est **sous-estimé**. Il faut refaire la mesure sur les données réelles (`DATASET = "real"`)."""),

    # --- 4.3 Tests de fuite train / validation ---------------------------------------------------------------
    ("md", """### 4.3 Tests de fuite entre train et validation autour des plis

Une fuite train / validation, c'est une information de l'un qui entre dans l'autre alors qu'elle ne le devrait
pas. Autour des plis, il y a cinq chemins à tester :

```
                ┌──────────────── train ────────────────┐      ┌──── validation ────┐
lignes          │ paires des paiements du train         │  ?   │ paires de la valid │   T1 : mêmes lignes ?
entités         │ factures, débiteurs                   │  ?   │ idem               │   T2 : recouvrements
étiquettes      │ imputations des paiements du train    │ ◄─?─ │ période validation │   T3 : étiquettes venues de la validation
scores oof      │ modèles de pli : nb d'itérations      │ ◄─?─ │ arrêt précoce      │   T4 : les oof dépendent-ils de la validation ?
distribution    │ features de compétition (oof)         │  ≠?  │ (passe 1)          │   T5 : validation adverse
                └───────────────────────────────────────┘      └────────────────────┘
```

Chaque test rend un verdict : **OK** (pas de fuite), **FUITE** (une information passe), ou **À SURVEILLER**
(recouvrement normal, mais à connaître). Les verdicts sont regroupés à la fin du § 4.3."""),
    ("code", """verdicts = []


def verdict(test: str, constat: str, statut: str, mesure) -> None:
    verdicts.append({"test": test, "constat": constat, "statut": statut, "mesure": mesure})
    display(Markdown(f"**{test} — {statut}** : {constat} (mesure : `{mesure}`)"))


# T1 — mêmes lignes de part et d'autre ?
common_payments = np.intersect1d(train[GROUP].unique(), valid[GROUP].unique())
gap_days = (pd.to_datetime(valid["day"]).min() - pd.to_datetime(train["day"]).max()).days
fold_rows = sum(int((fold != k).sum()) for k in range(FOLDS))
verdict("T1a paiements communs", f"{len(common_payments)} paiement(s) à la fois dans le train et la validation",
        "OK" if len(common_payments) == 0 else "FUITE", len(common_payments))
verdict("T1b écart entre périodes", f"{gap_days} jours entre le dernier jour du train et le premier de la validation "
        f"(purge configurée : {settings.split.purge_days} jours)",
        "OK" if gap_days > settings.split.purge_days else "FUITE", gap_days)
verdict("T1c lignes des plis", f"les {FOLDS} modèles de pli ont appris sur {fold_rows:,} lignes, toutes du train "
        f"({FOLDS - 1} × {len(train):,})".replace(",", " "),
        "OK" if fold_rows == (FOLDS - 1) * len(train) else "FUITE", fold_rows)"""),
    ("code", """# T2 — mêmes entités de part et d'autre (normal en affacturage, mais à connaître)
inv_both = np.intersect1d(train["invoice_id"].unique(), valid["invoice_id"].unique())
pos_both = set(train.loc[train["label"] == 1, "invoice_id"]) & set(valid.loc[valid["label"] == 1, "invoice_id"])
deb_of = data.tables["invoice"].set_index("invoice_id")["debtor_id"]
deb_train = set(deb_of.reindex(train["invoice_id"].unique()).dropna())
deb_valid = set(deb_of.reindex(valid["invoice_id"].unique()).dropna())
share_deb = len(deb_train & deb_valid) / max(len(deb_valid), 1)
verdict("T2a factures candidates des deux côtés", f"{len(inv_both) / max(valid['invoice_id'].nunique(), 1):.1%} des factures "
        "candidates en validation l'étaient déjà au train (factures restées ouvertes)", "À SURVEILLER", len(inv_both))
verdict("T2b factures vraies des deux côtés", f"{len(pos_both)} facture(s) imputée(s) à un paiement du train et à un "
        "paiement de la validation (paiements partiels successifs)", "À SURVEILLER", len(pos_both))
verdict("T2c débiteurs communs", f"{share_deb:.1%} des débiteurs de la validation figurent au train : le modèle peut "
        "apprendre des habitudes propres à un débiteur", "À SURVEILLER", round(share_deb, 3))"""),
    ("md", """**T2 n'est pas une fuite en soi** : en production aussi, le modèle connaît les débiteurs et les factures ouvertes du
passé. Cela le devient si une information *postérieure* les accompagne : c'est ce que mesurent T3 et T4."""),
    ("code", """# T3 — des étiquettes du train ne sont-elles connues que pendant (ou après) la validation ? (F3)
pos_pairs = train.loc[train["label"] == 1, ["payment_id", "invoice_id", "day"]]
first_known = (pos_pairs.merge(imputations, on=["payment_id", "invoice_id"], how="left")
               .groupby(["payment_id", "invoice_id", "day"], as_index=False)["updated_at"].min())
valid_start = pd.Timestamp(valid_p.start)
late = first_known["updated_at"] >= valid_start
verdict("T3 étiquettes du train venues de la validation",
        f"{late.mean():.2%} des paires positives du train ({int(late.sum())}) ne sont imputées qu'à partir du "
        f"{valid_start.date()} : leur étiquette vient de la période de validation, voire du test",
        "OK" if late.sum() == 0 else "FUITE", int(late.sum()))
delay_days = (first_known["updated_at"] - pd.to_datetime(first_known["day"])).dt.days
display(delay_days.describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_frame("délai arrivée → imputation (jours)").T.round(1))"""),
    ("md", """### T4 — Les scores oof du train dépendent-ils de la validation ?

Les modèles de pli n'apprennent que sur le train (T1c), mais leur **nombre d'itérations** vient de l'arrêt précoce
de la passe 1 **sur la validation** (`max(pass1.best_iteration, 10)`, `model.py:121`). La validation peut donc
modifier des features du train.

Test causal : on **mélange les étiquettes de la validation** (elle ne contient alors plus aucune information) et on
refait passe 1 + plis. Si les oof du train changent, c'est que la validation passait dans le train.
Contrôle : refaire les plis au même nombre d'itérations doit redonner exactement les mêmes oof (déterminisme). Cela
prouve que le **seul** canal est ce nombre d'itérations."""),
    ("code", """def oof_scores(rounds: int) -> np.ndarray:
    out = np.zeros(len(train))
    for k in range(FOLDS):
        m = lgb.train({**PARAMS}, lgb_dataset(X[fold != k], y[fold != k]), num_boost_round=rounds)
        out[fold == k] = m.predict(X[fold == k])
    return out


yv_shuffled = np.random.default_rng(SEED).permutation(yv)
d_tr = lgb_dataset(X, y)
pass1_shuffled = lgb.train(PARAMS, d_tr, num_boost_round=cfg.num_boost_round,
                           valid_sets=[lgb_dataset(Xv, yv_shuffled, reference=d_tr)],
                           callbacks=[lgb.early_stopping(30, verbose=False)])
rounds_shuffled = max(pass1_shuffled.best_iteration, 10)
oof_again = oof_scores(ROUNDS)                                    # contrôle : mêmes itérations
oof_shuffled = oof_scores(rounds_shuffled) if rounds_shuffled != ROUNDS else oof_again.copy()

t4 = pd.DataFrame({
    "itérations des plis": [ROUNDS, ROUNDS, rounds_shuffled],
    "écart max des oof": [0.0, float(np.abs(oof_again - oof).max()), float(np.abs(oof_shuffled - oof).max())],
    "écart moyen des oof": [0.0, float(np.abs(oof_again - oof).mean()), float(np.abs(oof_shuffled - oof).mean())],
    "precision@1 des oof (train)": [pair_metrics(train, s)["precision_at_1"] for s in (oof, oof_again, oof_shuffled)],
}, index=["référence (validation réelle)", "contrôle : mêmes itérations", "validation aux étiquettes mélangées"])
display(t4.round(4))
deterministic = np.allclose(oof_again, oof)
moved = not np.allclose(oof_shuffled, oof)
verdict("T4a canal unique", "à nombre d'itérations égal, les oof sont identiques : les plis ne lisent pas la validation"
        if deterministic else "les oof changent à itérations égales : un autre canal est à chercher",
        "OK" if deterministic else "FUITE", round(float(np.abs(oof_again - oof).max()), 6))
verdict("T4b dépendance à la validation", f"mélanger les étiquettes de validation fait passer les plis de {ROUNDS} à "
        f"{rounds_shuffled} itérations et modifie les oof du train" if moved else
        "mélanger les étiquettes de validation ne change pas les oof", "FUITE" if moved else "OK",
        round(float(np.abs(oof_shuffled - oof).mean()), 4))"""),
    ("md", """**À lire.** T4b est une fuite **faible** : un seul entier, le nombre d'itérations, passe de la validation au train.
Elle a deux conséquences :
- les features de compétition du train ne sont pas indépendantes de la validation, qui sert ensuite à arrêter la passe
  2 et à calibrer : la validation est réutilisée (F1) ;
- le correctif est simple : fixer ce nombre d'itérations **sans la validation** (arrêt précoce à l'intérieur du
  train, sur un pli temporel), ou sur une sous-période de validation qui ne sert à rien d'autre."""),
    ("md", """### T5 — Validation adverse : les features de compétition du train ressemblent-elles à celles de la validation ?

Méthode classique pour détecter un décalage train / validation : entraîner un classifieur à **deviner si une ligne vient
du train ou de la validation**. AUC ≈ 0,5 : impossible de les distinguer. AUC proche de 1 : distributions différentes.

On l'applique au **meilleur candidat de chaque paiement** (score et marge au deuxième), pour chaque façon de noter le
train. Deux repères :
- **dérive naturelle** : première moitié de la validation contre seconde moitié, notées par la même passe 1. C'est
  l'écart dû au seul passage du temps ;
- **en échantillon** : l'écart d'une notation qui fuit franchement.

Une notation du train sans fuite doit se rapprocher de la dérive naturelle.

**Contrôle du test lui-même** : la notation en échantillon fuit à coup sûr. Si le classifieur ne la distingue pas mieux
que les autres, le test n'a pas la puissance de conclure, et son verdict est « NON CONCLUANT »."""),
    ("code", """def top1_features(df: pd.DataFrame, s: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    \"\"\"Par paiement, sans regarder les étiquettes : meilleur score et marge au 2e ; et le jour du paiement.\"\"\"
    d = pd.DataFrame({"g": df[GROUP].to_numpy(), "s": s, "day": pd.to_datetime(df["day"]).to_numpy()})
    d = d.sort_values(["g", "s"], ascending=[True, False], kind="mergesort")
    r = d.groupby("g").cumcount().to_numpy()
    first = d[r == 0].set_index("g")
    second = d[r == 1].set_index("g")["s"].reindex(first.index).fillna(0.0)
    out = pd.DataFrame({"best": first["s"], "margin": first["s"] - second})
    return out.reset_index(drop=True), first["day"].to_numpy()


def adversarial_auc(a: pd.DataFrame, b: pd.DataFrame) -> float:
    \"\"\"AUC hors échantillon (2 plis) d'un classifieur qui distingue les lignes de `a` de celles de `b`.\"\"\"
    Xa = pd.concat([a, b], ignore_index=True)
    ya = np.r_[np.zeros(len(a)), np.ones(len(b))]
    part = np.random.default_rng(SEED).integers(0, 2, len(Xa))
    p = np.zeros(len(Xa))
    adv_params = {"objective": "binary", "learning_rate": 0.1, "num_leaves": 15, "min_data_in_leaf": 50, "seed": SEED,
                  "deterministic": True, "force_row_wise": True, "verbosity": -1}
    for k in (0, 1):
        m = lgb.train(adv_params, lgb.Dataset(Xa[part != k], label=ya[part != k]), num_boost_round=100)
        p[part == k] = m.predict(Xa[part == k])
    return float(roc_auc_score(ya, p))


valid_top, valid_top_day = top1_features(valid, s1_valid)
first_half = valid_top_day < valid_top_day.min() + (valid_top_day.max() - valid_top_day.min()) / 2
adv = pd.Series({
    "repère : dérive naturelle (validation 1re moitié vs 2e moitié)": adversarial_auc(valid_top[first_half], valid_top[~first_half]),
    "train noté en échantillon (fuite franche)": adversarial_auc(top1_features(train, s1_train_in)[0], valid_top),
    "train noté par plis aléatoires (code actuel)": adversarial_auc(top1_features(train, oof)[0], valid_top),
    "train noté en walk-forward (sans futur)": adversarial_auc(top1_features(same_rows, wf[has_wf])[0], valid_top),
}, name="AUC adverse")
display(adv.round(3).to_frame())
ref_auc, in_auc, cur_auc, wf_auc = adv.iloc[0], adv.iloc[1], adv.iloc[2], adv.iloc[3]
powerful = in_auc > max(cur_auc, wf_auc) + 0.02          # le témoin positif doit être détecté
statut = "NON CONCLUANT" if not powerful else ("OK" if cur_auc <= wf_auc + 0.02 else "FUITE")
verdict("T5 validation adverse", f"en échantillon : AUC {in_auc:.3f} ; plis aléatoires : {cur_auc:.3f} ; walk-forward : "
        f"{wf_auc:.3f} ; dérive naturelle : {ref_auc:.3f}"
        + ("" if powerful else " — le témoin en échantillon n'est pas détecté : le test ne peut pas conclure"),
        statut, round(cur_auc, 3))
alt.Chart(adv.reset_index().rename(columns={"index": "comparaison"})).mark_bar().encode(
    x=alt.X("AUC adverse:Q", scale=alt.Scale(domain=[0.4, 1])), y=alt.Y("comparaison:N", sort=None, title=None),
    tooltip=["comparaison", alt.Tooltip("AUC adverse:Q", format=".3f")]
).properties(width=420, height=150, title="Train ou validation ? 0,5 = indiscernables")"""),
    ("md", """**À lire.**
- L'AUC « en échantillon » montre à quoi ressemble une fuite franche : le classifieur reconnaît le train à ses marges
  trop nettes.
- Si les plis aléatoires dépassent nettement le walk-forward et la dérive naturelle, les features de compétition du
  train portent une information que la validation n'a pas : c'est F4.
- La dérive dans le temps fait aussi monter l'AUC : le walk-forward note les premiers mois avec des modèles appris
  sur peu de données, donc des scores plus bas. D'où la comparaison **entre notations**, et le contrôle par le témoin
  en échantillon.
- Les tests T3 et T4 sont causaux et plus fiables ; T5 est un indicateur de distribution, à lire avec eux."""),
    ("code", """display(Markdown("### Bilan des tests de fuite train / validation"))
summary_leaks = pd.DataFrame(verdicts)
display(summary_leaks)
n_leaks = int((summary_leaks["statut"] == "FUITE").sum())
display(Markdown(f"**{n_leaks} fuite(s) détectée(s)** sur {len(summary_leaks)} contrôles. Correctifs : `docs/fuites_ml.md` "
                 "(F1 pour T4, F3 pour T3, F4 pour T5)."))"""),

    # --- 5. Features de compétition ---------------------------------------------------------------------------
    ("md", """## 5. Construction des features de compétition

`competition_features(scores, group)` (`model.py:37-54`) prend un score par paire et calcule, **au sein de chaque
paiement** :

| feature | définition |
|---|---|
| `rank_in_payment` | rang du score dans le paiement (1 = meilleur) |
| `score_best_other` | meilleur score **des autres** candidats : le 2e pour le 1er, le 1er pour tous les autres |
| `score_margin` | score − `score_best_other` : positif pour le 1er seulement |

D'où viennent les scores selon le jeu :

```
train      : oof (plis)                 ─┐
validation : passe 1 complète            ├─► competition_features ─► 3 colonnes ajoutées aux features
production : passe 1 complète            ─┘      (même code partout ; seule la source des scores change)
```"""),
    ("code", """train_ex = pick_payment(train, 3, 6)
rows_ex = train[GROUP].to_numpy() == train_ex
comp_ex = competition_features(oof[rows_ex], train.loc[rows_ex, GROUP].to_numpy())
display(Markdown(f"**Un paiement du train** (`{train_ex}`) : score oof → features de compétition"),
        pd.concat([train.loc[rows_ex, ["invoice_id", "label", "balance"]].reset_index(drop=True),
                   pd.DataFrame({"s1 (oof)": oof[rows_ex]}), comp_ex], axis=1).sort_values("rank_in_payment"))

COMP_TRAIN = competition_features(oof, train[GROUP].to_numpy())
COMP_TRAIN_WF = competition_features(wf[has_wf], same_rows[GROUP].to_numpy())
COMP_VALID = competition_features(s1_valid, valid[GROUP].to_numpy())
desc = pd.concat({"train (oof aléatoire)": COMP_TRAIN.describe().T, "train (walk-forward)": COMP_TRAIN_WF.describe().T,
                  "validation (passe 1)": COMP_VALID.describe().T})[["mean", "50%", "std"]]
display(desc.unstack(0).round(3))"""),
    ("code", """def label_rate_by(comp: pd.DataFrame, labels: np.ndarray, name: str) -> pd.DataFrame:
    d = comp.assign(label=labels, src=name)
    d["tranche"] = pd.cut(d["score_margin"], np.linspace(-1, 1, 21), include_lowest=True)
    out = d.groupby(["src", "tranche"], observed=True).agg(n=("label", "size"), taux_vraies=("label", "mean")).reset_index()
    out = out[out["n"] >= 30]
    out["tranche"] = out["tranche"].map(lambda iv: round(iv.mid, 3)).astype(float)
    return out


rates = pd.concat([label_rate_by(COMP_TRAIN, y, "train (oof aléatoire)"),
                   label_rate_by(COMP_TRAIN_WF, same_rows["label"].to_numpy(), "train (walk-forward)"),
                   label_rate_by(COMP_VALID, yv, "validation (passe 1)")])
alt.Chart(rates).mark_line(point=True).encode(
    x=alt.X("tranche:Q", title="score_margin"), y=alt.Y("taux_vraies:Q", title="part de vraies paires"), color="src:N",
    tooltip=["src", "tranche", "n", alt.Tooltip("taux_vraies:Q", format=".1%")]
).properties(width=600, height=280, title="Ce que la passe 2 va apprendre : P(vraie paire | marge), selon la source des scores")"""),
    ("md", """**À lire.** La passe 2 apprend la courbe de **sa source d'entraînement** et l'applique à des marges calculées
comme en validation. Plus la courbe « train » s'écarte de la courbe « validation », plus la passe 2 se trompe
d'interprétation sur les marges."""),

    # --- 6. Passe 2 -----------------------------------------------------------------------------------------
    ("md", """## 6. Passe 2, puis calibration

```
train      : [features de la paire | rang, marge, meilleur concurrent (oof)]      ─► LightGBM ─► s2
validation : [features de la paire | rang, marge, meilleur concurrent (passe 1)]  ─► arrêt précoce
```

Mêmes hyperparamètres, arrêt précoce sur la validation. Toujours aucune normalisation des features : les trois
colonnes de compétition sont dans [0, 1] ou [−1, 1], ce qui est indifférent aux arbres."""),
    ("code", """X2 = pd.concat([X.reset_index(drop=True), COMP_TRAIN], axis=1)
Xv2 = pd.concat([Xv.reset_index(drop=True), COMP_VALID], axis=1)
d2_train, history2 = lgb_dataset(X2, y), {}
d2_valid = lgb_dataset(Xv2, yv, reference=d2_train)
pass2 = lgb.train(PARAMS, d2_train, num_boost_round=cfg.num_boost_round, valid_sets=[d2_train, d2_valid],
                  valid_names=["train", "validation"],
                  callbacks=[lgb.early_stopping(30, verbose=False), lgb.record_evaluation(history2)])
s2_valid = pass2.predict(Xv2)
display(learning_curve(history2, pass2.best_iteration, "Passe 2 : perte par itération"))
display(scores_table({"passe 1 · validation": (valid, s1_valid), "passe 2 · validation": (valid, s2_valid)}))

imp2 = pd.DataFrame({"feature": pass2.feature_name(), "gain": pass2.feature_importance("gain")})
imp2["famille"] = imp2["feature"].map(FAMILY_OF).fillna("compétition")
imp2["part du gain"] = imp2["gain"] / imp2["gain"].sum()
display(imp2.groupby("famille")["part du gain"].sum().sort_values(ascending=False).round(3).to_frame())"""),
    ("md", """### 6.1 Effet de F4 sur la passe 2

On entraîne une seconde passe 2, identique, mais dont les features de compétition du train viennent du
walk-forward (sans futur). Elle apprend sur moins de paires (le premier mois n'a pas de score). Les deux sont
évaluées sur la **même** validation."""),
    ("code", """X2_wf = pd.concat([same_rows[FEATURES].reset_index(drop=True), COMP_TRAIN_WF], axis=1)
d2_wf = lgb_dataset(X2_wf, same_rows["label"].to_numpy())
pass2_wf = lgb.train(PARAMS, d2_wf, num_boost_round=cfg.num_boost_round, valid_sets=[lgb_dataset(Xv2, yv, reference=d2_wf)],
                     callbacks=[lgb.early_stopping(30, verbose=False)])
X2_same = pd.concat([same_rows[FEATURES].reset_index(drop=True), COMP_TRAIN[has_wf].reset_index(drop=True)], axis=1)
d2_same = lgb_dataset(X2_same, same_rows["label"].to_numpy())
pass2_same = lgb.train(PARAMS, d2_same, num_boost_round=cfg.num_boost_round,
                       valid_sets=[lgb_dataset(Xv2, yv, reference=d2_same)], callbacks=[lgb.early_stopping(30, verbose=False)])
display(scores_table({"passe 1": (valid, s1_valid),
                      "passe 2 · plis aléatoires, tout le train (code actuel)": (valid, s2_valid),
                      "passe 2 · plis aléatoires, mêmes paires que walk-forward": (valid, pass2_same.predict(Xv2)),
                      "passe 2 · walk-forward": (valid, pass2_wf.predict(Xv2))}))"""),
    ("md", """**À lire.** La ligne à comparer à « walk-forward » est « mêmes paires que walk-forward » : même volume
d'apprentissage, seule la source des scores change. L'écart entre les deux est le coût (ou le gain) de F4 **sur des
paiements futurs**, ce qui est la seule chose qui compte. Rappel : la validation ne contient pas de futur, donc ces
chiffres ne sont pas gonflés par F4 ; F4 change le modèle, pas la mesure.

Attention : la validation sert ici aussi à l'arrêt précoce (F1). Pour une conclusion ferme, refaire la comparaison
sur le test avec `project.backtest`."""),
    ("md", """### 6.2 La « normalisation » des scores : calibration isotonique

Le score brut d'un LightGBM n'est pas une probabilité fiable (surtout sans rééchantillonnage, avec beaucoup de
négatifs). La **régression isotonique** apprend une fonction croissante, en escalier, qui transforme le score brut en
probabilité : parmi les paires notées autour de 0,9, environ 90 % doivent être vraies.

Le code l'ajuste **sur la validation** puis rapporte sa qualité **sur la validation** (F1) : c'est presque parfait
par construction. Pour une lecture honnête, on l'ajuste aussi sur la première moitié de la validation (V1) et on la
mesure sur la seconde (V2)."""),
    ("code", """calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(s2_valid, yv)   # comme PairModel.fit
p_valid = calibrator.predict(s2_valid)

vday = pd.to_datetime(valid["day"])
v1 = (vday < vday.min() + (vday.max() - vday.min()) / 2).to_numpy()
honest = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(s2_valid[v1], yv[v1])

frames = []
for name, yy, pp in (("brut · validation", yv, s2_valid), ("isotonique · ajusté et mesuré sur la validation (code)", yv, p_valid),
                     ("isotonique · ajusté sur V1, mesuré sur V2", yv[~v1], honest.predict(s2_valid[~v1]))):
    t = calibration_table(yy, pp)
    frames.append(t.assign(version=name))
cal = pd.concat(frames, ignore_index=True)
diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(color="gray", strokeDash=[4, 4]).encode(x="x:Q", y="y:Q")
pts = alt.Chart(cal).mark_line(point=True).encode(
    x=alt.X("score_moyen:Q", title="probabilité annoncée"), y=alt.Y("taux_observé:Q", title="part de vraies paires"),
    color=alt.Color("version:N", legend=alt.Legend(orient="bottom", direction="vertical")),
    tooltip=["version", "tranche", "paires", "score_moyen", "taux_observé"])
display((diag + pts).properties(width=420, height=380, title="Courbe de fiabilité (diagonale = parfaitement calibré)"))
display(pd.DataFrame({"logloss": {"brut · validation": log_loss(yv, np.clip(s2_valid, 1e-7, 1 - 1e-7)),
                                  "isotonique · V1 → V2": log_loss(yv[~v1], np.clip(honest.predict(s2_valid[~v1]), 1e-7, 1 - 1e-7)),
                                  "brut · V2": log_loss(yv[~v1], np.clip(s2_valid[~v1], 1e-7, 1 - 1e-7))}}))"""),
    ("code", """top = s2_valid >= np.quantile(s2_valid, 0.95)
plateaus = pd.DataFrame({"scores distincts (5 % les plus hauts)": {"brut": np.unique(s2_valid[top]).size,
                                                                     "calibré": np.unique(p_valid[top]).size}})
display(plateaus)
steps = pd.DataFrame({"brut": np.sort(s2_valid)[::max(len(s2_valid) // 400, 1)]})
steps["calibré"] = calibrator.predict(steps["brut"])
alt.Chart(steps).mark_line(interpolate="step-after").encode(x=alt.X("brut:Q", title="score brut de passe 2"),
                                                            y=alt.Y("calibré:Q", title="probabilité calibrée")
).properties(width=420, height=260, title="La fonction isotonique : un escalier")"""),
    ("md", """**À lire.**
- La calibration est **monotone** : elle ne change pas l'ordre des candidats, donc ni la precision@1 ni le MRR.
- Elle crée des **paliers** : beaucoup de paires partagent la même probabilité près de 1. C'est pourquoi la décision
  (`decision.py`) place ses seuils sur le **score brut**, de même ordre mais sans paliers, et garde la probabilité
  calibrée pour la lecture et les diagnostics (`docs/choix_etape_ml.md`, décision 5)."""),

    # --- 7. Fidélité ----------------------------------------------------------------------------------------
    ("md", """## 7. Contrôle de fidélité : les étapes manuelles sont-elles bien `PairModel.fit` ?

On entraîne `PairModel.fit` sur les mêmes données et on compare les scores bruts et calibrés sur la validation.
LightGBM est en mode déterministe : on attend une égalité à la précision machine près."""),
    ("code", """if CHECK_FIDELITY:
    model = PairModel.fit(train, valid, FEATURES, cfg, settings.reconcile_ml.second_pass, settings.reconcile_ml.calibration,
                          folds=FOLDS)
    raw_ref = model.raw(valid, valid[GROUP].to_numpy())
    checks = {"passe 1 : même meilleure itération": model.pass1.best_iteration == pass1.best_iteration,
              "passe 1 : mêmes scores": np.allclose(model.pass1.predict(Xv), s1_valid),
              "passe 2 : mêmes scores bruts": np.allclose(raw_ref, s2_valid),
              "calibration : mêmes probabilités": np.allclose(model.calibrate(raw_ref), p_valid)}
    display(pd.Series(checks, name="identique"))
else:
    print("CHECK_FIDELITY = False : contrôle sauté")"""),

    # --- 8. Preuves empiriques ------------------------------------------------------------------------------
    ("md", """## 8. Preuves empiriques : ne pas croire, mesurer

Les sections précédentes raisonnent sur le code. Ici, deux tests **empiriques**, qui ne dépendent d'aucun raisonnement :

| Test | Question | Méthode |
|---|---|---|
| **8.1 Troncature de l'apprentissage** | Une information **postérieure à la validation** (période de test et après) entre-t-elle dans le jeu d'entraînement ou le modèle ? | effacer physiquement de toutes les tables ce qui date d'après la validation, **refaire tout l'apprentissage**, comparer |
| **8.2 Audit des features une à une** | Une feature contient-elle l'étiquette déguisée ? | mesurer ce que chaque feature prédit **seule** ; une feature quasi parfaite seule est le symptôme classique d'une fuite |

Ce que ces tests **ne couvrent pas**, et pourquoi :
- les fuites **à l'intérieur** du train (F4 : un pli de février a vu mars) : elles sont mesurées au § 4.2 ;
- la dépendance des plis à la validation (T4b) : mesurée au § 4.3 ;
- les attributs non datés (F2 : IBAN, noms) : la troncature ne peut pas les dater, puisque les données n'ont pas de
  date. Le générateur synthétique les crée dès le départ : F2 ne peut se mesurer que sur les données réelles, avec des
  dates de rattachement."""),
    ("md", """### 8.1 Troncature de l'apprentissage complet

```
journal complet  ──► rejeu train + validation ──► jeu d'entraînement A ──► modèle A
journal tronqué  ──► rejeu train + validation ──► jeu d'entraînement B ──► modèle B
(tout ce qui date d'après la fin de la validation est supprimé de toutes les tables :
 paiements, factures, imputations, client files, contrats, parties)
```

Si A = B (mêmes paires, mêmes features, mêmes étiquettes), alors **rien de postérieur à la validation n'a servi à
l'apprentissage**, et comme LightGBM est déterministe, les modèles sont identiques. Toute différence est une fuite,
et le test dit où : dans quelles colonnes, ou dans les étiquettes.

Même troncature que le test anti-fuite du projet (`tests/test_timeline.py:199-214`), recopiée ici pour ne pas
dépendre de pytest. Durée : un rejeu de plus (≈ 1 à 2 minutes sur le jeu de test)."""),
    ("code", """from src.load.events import build_journal, payment_event_time
from src.load.loader import LoadedData


def truncate(loaded: LoadedData, cutoff: pd.Timestamp) -> LoadedData:
    \"\"\"Données telles qu'elles existaient avant `cutoff` : tout événement ≥ cutoff supprimé.\"\"\"
    t = {k: v.copy() for k, v in loaded.tables.items()}
    t["payment"] = t["payment"][payment_event_time(t["payment"]) < cutoff]
    t["invoice"] = t["invoice"][t["invoice"]["creation_date"] < cutoff]
    t["imputation"] = t["imputation"][t["imputation"]["updated_at"] < cutoff]
    if "client_file" in t:
        t["client_file"] = t["client_file"][t["client_file"]["received_at"] < cutoff]
        t["client_file_line"] = t["client_file_line"][t["client_file_line"]["file_id"].isin(t["client_file"]["file_id"])]
    t["agreement"] = t["agreement"][t["agreement"]["created_at"] < cutoff]
    t["agreement"]["disabled_at"] = t["agreement"]["disabled_at"].where(t["agreement"]["disabled_at"] < cutoff)
    for role in ("assignor", "debtor"):
        p = t[role][~(t[role]["opened_at"] >= cutoff)].copy()
        p["closed_at"] = p["closed_at"].where(p["closed_at"] < cutoff)
        t[role] = p
    return LoadedData(tables={k: v.reset_index(drop=True) for k, v in t.items()}, mapped_fields=loaded.mapped_fields)


cutoff = pd.Timestamp(valid_p.end) + pd.Timedelta(days=1)
data_cut = truncate(data, cutoff)
journal_cut, _ = build_journal(data_cut)
removed = pd.DataFrame({"lignes complètes": {k: len(v) for k, v in data.tables.items()},
                        "après troncature": {k: len(data_cut.tables[k]) for k in data.tables}})
removed["supprimées"] = removed["lignes complètes"] - removed["après troncature"]
display(Markdown(f"**Tout ce qui date du {cutoff.date()} ou après est supprimé**"), removed)

state_cut = LedgerState(data_cut, journal_cut, settings.reconcile_ml.features.behavioral_window_days)
recorder_cut = DatasetRecorder(state_cut, settings, rules)
run_replay(state_cut, recorder_cut, train_p.start, valid_p.end, settings.split.retention_days,
           label="jeu d'entraînement sur journal tronqué")
ds_cut = recorder_cut.dataset()
ds_cut["label"] = label_pairs(ds_cut, data_cut.tables["imputation"])
print(f"jeu complet : {len(ds):,} paires · jeu tronqué : {len(ds_cut):,} paires".replace(",", " "))"""),
    ("code", """KEY = ["payment_id", "invoice_id"]
a = ds.set_index(KEY).sort_index()
b = ds_cut.set_index(KEY).sort_index()
only_a, only_b = a.index.difference(b.index), b.index.difference(a.index)
common = a.index.intersection(b.index)
a, b = a.loc[common], b.loc[common]


def differs(col: str) -> np.ndarray:
    x, z = a[col].to_numpy(), b[col].to_numpy()
    if np.issubdtype(np.asarray(x).dtype, np.number):
        return ~(np.isclose(x.astype(float), z.astype(float), equal_nan=True))
    return ~((x == z) | (pd.isna(x) & pd.isna(z)))


diff_cols = {c: int(differs(c).sum()) for c in FEATURES + ["label", "balance", "payment_amount"]}
diff_cols = {c: n for c, n in diff_cols.items() if n}
t81 = pd.DataFrame({"mesure": {
    "paires seulement dans le jeu complet": len(only_a),
    "paires seulement dans le jeu tronqué": len(only_b),
    "paires communes": len(common),
    "colonnes qui diffèrent sur les paires communes": len(diff_cols),
    "étiquettes qui diffèrent": diff_cols.get("label", 0)}})
display(t81)
if diff_cols:
    display(Markdown("**Colonnes qui diffèrent** (nombre de paires) : la fuite est là"),
            pd.Series(diff_cols, name="paires").sort_values(ascending=False).to_frame())
identical = not len(only_a) and not len(only_b) and not diff_cols
verdict("8.1 troncature de l'apprentissage",
        "jeu d'entraînement identique avec ou sans les données postérieures à la validation : rien de la période de "
        "test n'entre dans l'apprentissage" if identical else
        f"le jeu d'entraînement change quand on retire l'après-validation ({len(only_a) + len(only_b)} paires, "
        f"{len(diff_cols)} colonnes) : information postérieure utilisée",
        "OK" if identical else "FUITE", len(only_a) + len(only_b) + sum(diff_cols.values()))"""),
    ("code", """# Si les jeux diffèrent, on mesure ce que la différence change au modèle (sinon, modèles identiques par déterminisme).
if not identical:
    ds_cut[GROUP] = ds_cut["payment_id"]
    ds_cut["period"] = assign_period(ds_cut["day"], split).to_numpy()
    train_cut = ds_cut[ds_cut["period"] == "train"].reset_index(drop=True)
    valid_cut = ds_cut[ds_cut["period"] == "validation"].reset_index(drop=True)
    model_cut = PairModel.fit(train_cut, valid_cut, FEATURES, cfg, settings.reconcile_ml.second_pass,
                              settings.reconcile_ml.calibration, folds=FOLDS)
    model_full = model if CHECK_FIDELITY else PairModel.fit(train, valid, FEATURES, cfg, settings.reconcile_ml.second_pass,
                                                            settings.reconcile_ml.calibration, folds=FOLDS)
    display(scores_table({"modèle appris avec l'après-validation": (valid, model_full.raw(valid, valid[GROUP].to_numpy())),
                          "modèle appris sans": (valid, model_cut.raw(valid, valid[GROUP].to_numpy()))}))
else:
    print("Jeux identiques : les deux apprentissages donnent exactement le même modèle (LightGBM déterministe).")"""),
    ("md", """**À lire.**
- **OK** : la troncature est la définition même de « pas d'information future » : l'apprentissage n'a utilisé que ce qui
  existait à la fin de la validation. Les étiquettes comprises (F3 au-delà de la validation).
- **FUITE** : les colonnes listées disent où. Typiquement : des **étiquettes** qui changent (imputations prononcées après
  la validation, F3), ou des **colonnes catégorielles** dont le codage change (vocabulaires construits sur toutes les
  données, F7).
- Sur le jeu de test synthétique, les imputations arrivent en 3 jours au plus : F3 y est quasi invisible. **Le test
  décisif est sur les données réelles** (`DATASET = "real"`)."""),
    ("md", """### 8.2 Audit des features une à une

Une fuite se trahit souvent par une feature **trop bonne seule** : elle classe presque parfaitement sans aucune autre
aide. Pour chaque feature, on mesure sur la validation :
- l'**AUC seule** (dans le meilleur sens : la feature ou son opposé) ;
- la **precision@1 seule** : choisir, dans chaque paiement, le candidat qui a la plus forte valeur de la feature.

Repère : la precision@1 du modèle complet. Une feature qui s'en approche seule mérite d'être inspectée : soit elle est
réellement décisive (une référence exacte dans le libellé, par exemple), soit elle porte l'étiquette. C'est une alerte,
pas une preuve : l'inspection dit lequel des deux."""),
    ("code", """rows = []
for f in FEATURES:
    v = valid[f].to_numpy(dtype=float)
    if np.isfinite(v).sum() < 2 or np.nanstd(v) == 0:
        continue                                                     # feature vide ou constante
    filled = np.where(np.isnan(v), np.nanmin(v) - 1, v)
    auc = roc_auc_score(yv, filled)
    s = filled if auc >= 0.5 else -filled
    s = s + 1e-9 * np.random.default_rng(SEED).random(len(s))     # départage aléatoire des ex æquo
    rows.append({"feature": f, "famille": FAMILY_OF.get(f, "?"), "AUC seule": max(auc, 1 - auc),
                 "precision@1 seule": pair_metrics(valid, s)["precision_at_1"]})
audit = pd.DataFrame(rows).sort_values("precision@1 seule", ascending=False).reset_index(drop=True)
full_p1 = pair_metrics(valid, s2_valid)["precision_at_1"]
audit["alerte"] = np.where((audit["AUC seule"] > 0.99) | (audit["precision@1 seule"] > full_p1 - 0.02), "à inspecter", "")
display(Markdown(f"precision@1 du modèle complet (passe 2) : **{full_p1:.3f}**"), audit.head(15).round(3))
suspects = audit[audit["alerte"] != ""]
verdict("8.2 audit des features", f"{len(suspects)} feature(s) presque aussi bonne(s) seule(s) que le modèle complet"
        + (f" : {', '.join(suspects['feature'])}" if len(suspects) else ""),
        "À SURVEILLER" if len(suspects) else "OK", len(suspects))
top = audit.head(20)
bar = alt.Chart(top).mark_bar().encode(x=alt.X("precision@1 seule:Q", scale=alt.Scale(domain=[0, 1])),
                                       y=alt.Y("feature:N", sort="-x", title=None), color="famille:N",
                                       tooltip=["feature", "famille", alt.Tooltip("AUC seule:Q", format=".3f"),
                                                alt.Tooltip("precision@1 seule:Q", format=".3f")])
rule = alt.Chart(pd.DataFrame({"x": [full_p1]})).mark_rule(strokeDash=[4, 4], color="red").encode(x="x:Q")
(bar + rule).properties(width=420, height=420, title="Chaque feature seule (rouge : modèle complet)")"""),
    ("code", """display(Markdown("### Bilan de tous les contrôles de fuite"))
all_checks = pd.DataFrame(verdicts)
display(all_checks[["test", "statut", "constat"]])
display(Markdown(f"**{int((all_checks['statut'] == 'FUITE').sum())} fuite(s)**, "
                 f"{int((all_checks['statut'] == 'À SURVEILLER').sum())} point(s) à surveiller, "
                 f"{int((all_checks['statut'] == 'NON CONCLUANT').sum())} test(s) non concluant(s), sur {len(all_checks)} contrôles."))"""),

    # --- Récapitulatif --------------------------------------------------------------------------------------
    ("md", """## Récapitulatif

| Étape | Entrée | Sortie | Ce qu'il faut retenir |
|---|---|---|---|
| Jeu de paires | rejeu quotidien train + validation | une ligne par (paiement, facture candidate), features à D, étiquette | rappel des candidats = plafond du ML |
| Passe 1 | features de la paire | `s1` | bon classement, mais ignore la concurrence entre candidats |
| Plis | train coupé en 3 par paiement | `oof` : `s1` hors échantillon sur le train | retirent l'optimisme en échantillon, mais **pas le futur** (F4) |
| Compétition | scores + paiement | rang, marge, meilleur concurrent | `oof` sur le train, passe 1 en validation et en production |
| Passe 2 | features + compétition | `s2` (score brut, utilisé par la décision) | gain sur la passe 1 ; sensible à la source des scores du train |
| Calibration | `s2` en validation | probabilité | monotone, en escalier ; mesurée en échantillon par le code (F1) |

Suite de la chaîne : ensembles et décision (`src/reconcile_ml/decision.py`), puis backtest sur le test
(`project.backtest("test", matcher="pipeline")`). Les fuites identifiées et leurs correctifs : `docs/fuites_ml.md`."""),
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
    out = ROOT / "notebooks" / "ml_pas_a_pas.ipynb"
    out.write_text(json.dumps(build(), indent=1, ensure_ascii=False), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
