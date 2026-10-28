# Leakage F2 (IBAN non datés) : quelle part des paiements, et des auto-validations, peut en dépendre ?
# Dans le notebook, après `project = Project(dataset="real", ...)` : `%run -i ../tools/diag_iban_leakage.py`.
#
# Le référentiel `party_iban` n'a pas de date : il est visible dès le premier jour du replay. Un IBAN rattaché au
# débiteur APRÈS coup, à partir d'un rapprochement manuel, est donc connu avant le paiement qui l'a révélé.
# Majorant : on considère comme « révélé » tout paiement qui est le premier (au jour près) à venir de cet IBAN pour
# ce débiteur. Un IBAN connu dès l'entrée en relation est ainsi compté à tort : la mesure est un pire cas.
import numpy as np
import pandas as pd

from src.load.interim import load_interim
from src.timeline.split import assign_period, compute_split

data, _, _ = load_interim(project.interim_dir)
t = data.tables
pay = t["payment"]
arrival = (pay["booking_date"].fillna(pay["value_date"]) if "booking_date" in pay else pay["value_date"])
pay = pd.DataFrame({"payment_id": pay["payment_id"].astype(object),
                    "day": pd.to_datetime(arrival, utc=True).dt.tz_localize(None).dt.normalize(),
                    "iban": pay["iban_debtor"].astype(object)})
split = compute_split(project.settings.split, pay["day"].min().date(), pay["day"].max().date())
pay["period"] = assign_period(pay["day"], split).to_numpy()

# Débiteur réel de chaque paiement (via ses imputations), et lien (IBAN, débiteur) connu du référentiel.
truth = (t["imputation"][["payment_id", "invoice_id"]]
         .merge(t["invoice"][["invoice_id", "debtor_id"]], on="invoice_id")[["payment_id", "debtor_id"]]
         .drop_duplicates())
pd_ = pay[pay["iban"].notna()].merge(truth, on="payment_id")
links = t.get("party_iban")
if links is None:
    links = t["debtor"].loc[t["debtor"]["iban"].notna(), ["party_id", "iban"]].assign(role="debtor")
links = links[links["role"] == "debtor"][["party_id", "iban"]].drop_duplicates()
links = set(zip(links["iban"].astype(object), links["party_id"].astype(object)))
pd_["iban_connu"] = [(i, d) in links for i, d in zip(pd_["iban"], pd_["debtor_id"])]
first = pd_.groupby(["iban", "debtor_id"])["day"].transform("min")
pd_["revele"] = pd_["iban_connu"] & (pd_["day"] == first)
train_end = pd.Timestamp(split.period("train").end)
pd_["apparu_apres_train"] = pd_["iban_connu"] & (first > train_end)
per_pay = pd_.groupby("payment_id").agg(period=("period", "first"), iban_connu=("iban_connu", "any"),
                                        revele=("revele", "any"), apparu_apres_train=("apparu_apres_train", "any"))

print("Paiements imputés avec IBAN, par période :")
display(per_pay.groupby("period")[["iban_connu", "revele", "apparu_apres_train"]].mean()
        .rename(columns={"iban_connu": "IBAN connu du référentiel",
                         "revele": "IBAN révélé par ce paiement (pire cas)",
                         "apparu_apres_train": "IBAN apparu après la fin du train"})
        .map(lambda v: f"{100 * v:.1f} %"))

# Croisement avec les décisions du backtest de test, si l'export existe.
export = project.reports_dir / "decisions_pipeline_test.parquet"
if not export.exists():
    print(f"\nPas d'export {export.name} : lancer project.export_decisions('test', 'pipeline') pour le croisement.")
else:
    d = pd.read_parquet(export).drop_duplicates("payment_id")
    d = d.join(per_pay[["revele", "apparu_apres_train"]], on="payment_id")
    d[["revele", "apparu_apres_train"]] = d[["revele", "apparu_apres_train"]].fillna(False).astype(bool)
    auto = d[d["action_retenue_paiement"] == "auto"].copy()
    auto["exacte"] = auto["decision_retenue_exacte"].astype(float)
    print(f"\nAuto-validations du test : {len(auto)}")
    display(auto.groupby("revele").agg(paiements=("payment_id", "size"), precision=("exacte", "mean"))
            .rename(index={False: "IBAN non révélé", True: "IBAN révélé par le paiement (pire cas)"}))
    share = float(auto["revele"].mean()) if len(auto) else 0.0
    print(f"{share:.1%} des auto-validations reposent peut-être sur un IBAN révélé par le paiement lui-même.\n"
          f"Pire cas : taux d'automatisation officiel × {1 - share:.3f} si on les retire toutes, "
          f"avec une precision de {auto.loc[~auto['revele'], 'exacte'].mean():.2%} sur les autres.")
