# Chiffres à préparer pour la review ML, sur les données chargées (réelles ou synthétiques).
# Dans le notebook, après `project = Project(dataset="real", ...)` : `%run -i ../tools/review_chiffres.py`.
# Le croisement avec les décisions demande `project.export_decisions("test", "pipeline")` au préalable.
import numpy as np
import pandas as pd
from scipy.stats import beta

from src.load.interim import load_interim

data, _, meta = load_interim(project.interim_dir)
t = data.tables
s = project.settings
print("source :", meta.get("source"))

# F5 : imputations antérieures à l'arrivée de leur paiement (contrôle qualité du chargement).
issues_path = project.reports_dir / "load_issues.csv"
issues = pd.read_csv(issues_path) if issues_path.exists() else pd.DataFrame(columns=["kind", "count"])
f5 = int(issues.loc[issues["kind"] == "avant_connaissance_du_paiement", "count"].sum())
print(f"\nF5 · imputations datées avant l'arrivée de leur paiement : {f5} sur {len(t['imputation'])}")

# F3 : délai entre l'arrivée du paiement et sa première imputation, à comparer à la purge.
pay = t["payment"]
arrival = pay["booking_date"].fillna(pay["value_date"]) if "booking_date" in pay else pay["value_date"]
utc = lambda v: pd.to_datetime(v, utc=True).dt.tz_localize(None)
arrival = pd.Series(utc(arrival).to_numpy(), index=pay["payment_id"].astype(object))
first_imp = utc(t["imputation"].groupby("payment_id")["updated_at"].min())
delay = (first_imp - arrival.reindex(first_imp.index)).dt.total_seconds() / 86400
purge = s.split.purge_days
print(f"\nF3 · délai arrivée → première imputation (jours), purge configurée : {purge} j")
display(delay.describe(percentiles=[.5, .9, .95, .99]).round(1).to_frame("délai (j)").T)
print(f"part des paiements imputés au-delà de la purge : {(delay > purge).mean():.1%} ; "
      f"au-delà de la rétention ({s.split.retention_days} j) : {(delay > s.split.retention_days).mean():.1%}")

# Précision des auto-validations du test, avec borne basse de Clopper-Pearson unilatérale à 95 %.
export = project.reports_dir / "decisions_pipeline_test.parquet"
if not export.exists():
    print(f"\nPas d'export {export.name} : lancer project.export_decisions('test', 'pipeline').")
else:
    d = pd.read_parquet(export)
    auto = d[d["decision_retenue"] & (d["action"] == "auto")].drop_duplicates("payment_id")

    def bound(g: pd.DataFrame) -> pd.Series:
        n, k = len(g), int(g["decision_retenue_exacte"].astype(float).sum())
        lb = beta.ppf(0.05, k, n - k + 1) if k > 0 else 0.0
        return pd.Series({"auto-validations": n, "erreurs": n - k, "precision": k / n if n else np.nan,
                          "borne basse 95 %": lb})

    table = pd.concat([auto.groupby("step").apply(bound), bound(auto).to_frame("total").T])
    print(f"\nAuto-validations du test, par étape (cible {s.evaluation.target_precision:.1%}) :")
    display(table.style.format({"precision": "{:.2%}", "borne basse 95 %": "{:.2%}",
                                "auto-validations": "{:.0f}", "erreurs": "{:.0f}"}))

rate = s.evaluation.current_automation_rate
print("\nTaux d'automatisation actuel (référence du tableau en cascade) :",
      f"{rate:.1%}" if rate is not None else "NON RENSEIGNÉ : evaluation.current_automation_rate dans config/settings.yaml")
