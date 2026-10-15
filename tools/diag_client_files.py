# Diagnostic des client files : à quelle condition chacun échoue-t-il ?
# Dans le notebook, après `project = Project(dataset="real", ...)` : `%run -i ../tools/diag_client_files.py` (ou coller dans une cellule).
import numpy as np
import pandas as pd

from src.load.interim import load_interim

data, _, _ = load_interim(project.interim_dir)
t = data.tables
cf, cfl = t.get("client_file"), t.get("client_file_line")
assert cf is not None and cfl is not None, "pas de client files chargés : remplir client_file / client_file_line dans config/schema.yaml"
pay, inv, imp = t["payment"], t["invoice"], t["imputation"]
day = lambda s: pd.to_datetime(s, utc=True).dt.tz_localize(None).dt.normalize()
pct = lambda x: f"{100 * float(np.mean(x)):.1f} %"

# 1. Remplissage des champs utilisés par le moteur
print(f"{len(cf)} client files, {len(cfl)} lignes ({len(cfl) / max(len(cf), 1):.1f} par fichier)")
fill = {f"client_file.{c}": cf[c].notna().mean() for c in
        ("received_at", "total_amount", "payment_date", "iban", "payment_reference")}
fill |= {f"client_file_line.{c}": cfl[c].notna().mean() for c in ("invoice_reference", "amount")}
display(pd.Series(fill, name="remplis").map(lambda v: f"{100 * v:.1f} %").to_frame())

# 2. Les références des lignes retrouvent-elles une facture ? (égalité exacte après normalisation, sans
#    les variantes sans zéros / sans préfixe : légèrement pessimiste par rapport au moteur)
compact = lambda s: s.fillna("").astype(str).str.replace(" ", "", regex=False)
ref_inv = pd.concat([pd.DataFrame({"ref": compact(inv[c]), "invoice_id": inv["invoice_id"]})
                     for c in ("client_reference_norm", "internal_reference_norm") if c in inv])
ref_inv = ref_inv[ref_inv["ref"].str.len() >= 4].drop_duplicates()
n_inv_per_ref = ref_inv.groupby("ref").size()
lines = cfl[["file_id", "invoice_reference", "amount"]].assign(ref=compact(cfl["invoice_reference_norm"]))
lines["n_factures"] = lines["ref"].map(n_inv_per_ref).fillna(0).astype(int)
print("\nlignes dont la référence désigne 0 / 1 / plusieurs factures :")
display(lines["n_factures"].clip(upper=2).map({0: "0 (introuvable)", 1: "1", 2: "≥ 2 (ambiguë)"})
        .value_counts(normalize=True).map(lambda v: f"{100 * v:.1f} %").to_frame())
display(lines[lines["n_factures"] == 0][["invoice_reference"]].head(10))

# 3. Le vrai paiement de chaque fichier (via les factures citées et les imputations réelles)
hit = lines[lines["n_factures"] == 1].merge(ref_inv, on="ref").merge(imp[["invoice_id", "payment_id"]], on="invoice_id")
true_pay = (hit.groupby(["file_id", "payment_id"]).size().rename("lignes").reset_index()
            .sort_values("lignes", ascending=False).drop_duplicates("file_id"))
print(f"\nfichiers dont le vrai paiement est retrouvé : {len(true_pay)} / {len(cf)} ({len(true_pay) / max(len(cf), 1):.1%})")

f = cf.merge(true_pay, on="file_id").merge(
    pay[["payment_id", "amount", "value_date", "iban_debtor", "label_norm"]], on="payment_id")
line_sum = cfl.groupby("file_id")["amount"].sum(min_count=1).rename("somme_lignes")
f = f.join(line_sum, on="file_id")
f["delai_reception_j"] = (day(f["received_at"]) - day(f["value_date"])).dt.days
f["montant_total_egal"] = f["total_amount"] == f["amount"]
f["somme_lignes_egale"] = f["somme_lignes"] == f["amount"]
f["date_ok"] = (day(f["payment_date"]) - day(f["value_date"])).dt.days.abs() <= project.settings.allocation.signals.client_file.date_tolerance_days
f["iban_egal"] = f["iban"].notna() & (f["iban"].astype(object) == f["iban_debtor"].astype(object))
f["ref_dans_libelle"] = [bool(r) and isinstance(r, str) and r in (l or "") for r, l in
                         zip(f["payment_reference_norm"], f["label_norm"])]
f["corrobore"] = f["date_ok"] | f["iban_egal"] | f["ref_dans_libelle"]
same = pay["amount"].value_counts()
f["paiements_meme_montant"] = f["total_amount"].map(same).fillna(0).astype(int)

print("\nconditions du rattachement, sur les fichiers dont on connaît le vrai paiement :")
display(pd.Series({
    "montant total = montant du paiement (exact, centimes)": pct(f["montant_total_egal"]),
    "  dont manquant (total_amount vide)": pct(f["total_amount"].isna()),
    "somme des lignes = montant du paiement": pct(f["somme_lignes_egale"]),
    "date du fichier à ± tolérance de la date de valeur": pct(f["date_ok"]),
    "IBAN du fichier = IBAN du paiement": pct(f["iban_egal"]),
    "référence de virement dans le libellé": pct(f["ref_dans_libelle"]),
    "au moins une corroboration": pct(f["corrobore"]),
    "montant exact ET corroboré": pct(f["montant_total_egal"] & f["corrobore"]),
    "  dont montant partagé par > 1 paiement (risque d'ambiguïté)":
        pct(f["montant_total_egal"] & f["corrobore"] & (f["paiements_meme_montant"] > 1)),
}, name="part des fichiers").to_frame())

print("délai réception du fichier − date de valeur (jours) ; > 0 : le fichier arrive après le paiement")
display(f["delai_reception_j"].describe(percentiles=[.05, .25, .5, .75, .95]).round(1).to_frame().T)
print("écart montant total − paiement, quand il n'est pas nul (euros) :")
gap = (f["total_amount"] - f["amount"])[~f["montant_total_egal"] & f["total_amount"].notna()] / 100
display(gap.describe(percentiles=[.05, .25, .5, .75, .95]).round(2).to_frame().T)
