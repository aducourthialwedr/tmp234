# Diagnostic de performance : ce qui grossit avec la longueur de l'historique, sans rejouer la pipeline.
# Dans le notebook, après `project = Project(dataset="real", ...)` : `%run -i ../tools/diag_performance.py`.
#
# Le coût d'une journée de replay croît surtout avec deux volumes :
# - le stock de factures OUVERTES à D (balayé par les règles R3/R4, le signal montant de l'allocation, les
#   candidats ML « montant exact ») : une facture jamais imputée reste ouverte pour toujours, le restant dû étant
#   reconstruit par les seules imputations ;
# - le LOT du jour : paiements du jour + reliquat des 60 derniers jours non encore imputés.
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.memory import read_history
except ModuleNotFoundError:                      # version portable
    from reconciliation import memory as _memory
    read_history = _memory.read_history

# --- Où part le temps : le suivi mémoire est aussi un profil (un relevé par seconde, étiqueté) --------------
candidates = [globals().get("MEMORY_LOG"), Path("reports/memory.csv"), Path("../reports/memory.csv")]
log = next((Path(p) for p in candidates if p is not None and Path(p).exists()), None)
if log is None:
    print("Pas de memory.csv : lancer les étapes avec le MemoryMonitor du notebook pour obtenir le profil de temps.")
else:
    h = read_history(log)
    h["dt"] = (h.groupby("session")["elapsed_s"].shift(-1) - h["elapsed_s"]).clip(upper=60).fillna(0)
    prof = (h.groupby(["session", "step", "sub"], sort=False)["dt"].sum().reset_index()
            .groupby(["step", "sub"], sort=False)["dt"].sum().reset_index())
    prof["minutes"] = prof["dt"] / 60
    prof["part de l'étape"] = prof["minutes"] / prof.groupby("step")["minutes"].transform("sum")
    display(Markdown(f"**Temps passé par sous-étape** (`{log}`, toutes sessions)"),
            prof.sort_values("minutes", ascending=False).head(25)[["step", "sub", "minutes", "part de l'étape"]]
            .style.format({"minutes": "{:.1f}", "part de l'étape": "{:.0%}"}).hide(axis="index"))

d = project.interim_dir
split = project.split()
utc = lambda v: pd.to_datetime(v, utc=True).dt.tz_localize(None)
DAY = np.timedelta64(1, "D")

# --- Factures ouvertes au fil du temps ---------------------------------------------------------------------
inv = pd.read_parquet(d / "invoice.parquet", columns=["invoice_id", "creation_date", "due_date", "initial_amount", "debtor_id"])
inv["creation_date"], inv["due_date"] = utc(inv["creation_date"]), utc(inv["due_date"])
journal = pd.read_parquet(d / "journal.parquet", columns=["ts", "event_type", "related_id", "entity_id", "amount"],
                          filters=[("event_type", "==", "IMPUTATION_APPLIED")])
imp = journal.rename(columns={"related_id": "invoice_id", "entity_id": "payment_id"})
imp["ts"] = utc(imp["ts"])
imp = imp.merge(inv[["invoice_id", "initial_amount"]], on="invoice_id").sort_values(["invoice_id", "ts"], kind="mergesort")
imp["cum"] = imp.groupby("invoice_id")["amount"].cumsum()
closed_at = imp[imp["cum"] >= imp["initial_amount"]].groupby("invoice_id")["ts"].min()
inv["closed_at"] = inv["invoice_id"].map(closed_at)
inv["ever_imputed"] = inv["invoice_id"].isin(imp["invoice_id"].unique())

months = pd.date_range(inv["creation_date"].min().normalize(), max(p.end for p in split.periods), freq="MS")
created_sorted = np.sort(inv["creation_date"].to_numpy())
closed_sorted = np.sort(inv["closed_at"].dropna().to_numpy())
open_curve = pd.DataFrame({"mois": months.strftime("%Y-%m"),
                           "factures créées": np.searchsorted(created_sorted, months.to_numpy(), side="left"),
                           "factures soldées": np.searchsorted(closed_sorted, months.to_numpy(), side="left")})
open_curve["factures ouvertes"] = open_curve["factures créées"] - open_curve["factures soldées"]
display(Markdown("**Stock de factures ouvertes en début de mois** (doit rester stable ; s'il croît sans cesse, chaque "
                 "journée de replay devient plus lourde)"), open_curve.set_index("mois").T)

t0 = pd.Timestamp(split.period("test").start)
is_open = (inv["creation_date"] < t0) & ~(inv["closed_at"] < t0)
o = inv[is_open]
past_due = (t0 - o["due_date"]).dt.days
buckets = pd.cut(past_due, [-10**6, 0, 90, 180, 365, 730, 10**6],
                 labels=["pas encore échue", "0–90 j de retard", "90–180 j", "180–365 j", "1–2 ans", "> 2 ans"])
age = pd.DataFrame({"factures": buckets.value_counts().sort_index()})
age["part"] = age["factures"] / age["factures"].sum()
age["dont jamais imputées"] = o.groupby(buckets, observed=False)["ever_imputed"].apply(lambda s: (~s).sum())
display(Markdown(f"**Factures ouvertes au début du test ({t0.date()}) : {len(o):,}** sur {len(inv):,}, par retard"
                 .replace(",", " ")), age.style.format({"part": "{:.1%}"}))
per_debtor = o.groupby("debtor_id").size()
display(Markdown(f"Factures ouvertes par débiteur : médiane {per_debtor.median():.0f}, P99 {per_debtor.quantile(0.99):.0f}, "
                 f"max {per_debtor.max():,} ; les 10 plus gros débiteurs en portent "
                 f"{per_debtor.nlargest(10).sum() / max(len(o), 1):.1%}.".replace(",", " ")))

# --- Taille du lot quotidien (paiements du jour + reliquat) ------------------------------------------------
pay = pd.read_parquet(d / "payment.parquet", columns=["payment_id", "value_date", "booking_date"])
pay["arrival"] = utc(pay["booking_date"].fillna(pay["value_date"])).dt.normalize()
first_imp = imp.groupby("payment_id")["ts"].min()
pay["first_imputation"] = pay["payment_id"].map(first_imp)
delay = (pay["first_imputation"] - pay["arrival"]) / DAY
display(Markdown(f"**Délai arrivée → première imputation** : médiane {delay.median():.1f} j, P90 {delay.quantile(0.9):.1f} j, "
                 f"P95 {delay.quantile(0.95):.1f} j ; **jamais imputés : {delay.isna().mean():.1%}** des paiements "
                 "(ils restent 60 jours dans le reliquat)."))

retention = project.settings.split.retention_days
a = pay["arrival"].to_numpy()
f = pay["first_imputation"].fillna(pd.Timestamp.max.normalize()).to_numpy()
rows = []
for p in split.periods:
    if p.name == "train":
        continue
    for day in pd.date_range(p.start, p.end, freq="7D"):
        D = np.datetime64(day)
        window = (a <= D) & (a >= D - retention * DAY)
        rows.append({"période": p.name, "jour": day.date(), "paiements du jour": int((a == D).sum()),
                     "lot (majorant)": int((window & (f >= D)).sum())})
batch = pd.DataFrame(rows)
display(Markdown("**Taille du lot quotidien, un jour par semaine** (majorant : sans les auto-validations du moteur, qui "
                 "sortent aussi du reliquat)"),
        batch.groupby("période")[["paiements du jour", "lot (majorant)"]].agg(["mean", "max"]).round(0))
