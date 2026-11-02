"""Évaluation approfondie : une ligne par paiement du périmètre, pour comprendre les erreurs et le résiduel.

Même périmètre, même vérité et même décision retenue que `evaluate` (src/evaluation/metrics.py) : paiements
arrivés dans la période, imputations réelles toutes dates confondues, première auto-validation sinon première
mise en revue. Colonnes ajoutées :

- `issue` : auto juste, auto fausse, revue, aucune décision ;
- `type_erreur` (auto fausse) : paiement jamais imputé, mauvais débiteur, bon débiteur mais autre facture,
  ensemble incomplet, factures en trop, ensemble en partie juste ;
- `raison` (paiement non auto-validé, avec imputation réelle) : premier blocage rencontré, du débiteur au seuil
  (voir `REASONS`) ; le nombre de paiements par raison borne le gain possible du levier correspondant ;
- segments : tranche de montant, canal, type, bankroll, IBAN, client file, statut d'allocation, type de groupe,
  marché, taille du débiteur, délai d'imputation, mois ;
- diagnostic ML (rapprocheur règles + ML) : candidats au premier passage en ML, rang et score de la vraie facture.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.diagnostics import clopper_pearson_lower
from src.evaluation.metrics import _proposals, _tuples_by_key
from src.timeline.loop import AUTO, REVIEW

AUTO_OK, AUTO_KO, IN_REVIEW, NO_DECISION = "auto juste", "auto fausse", "revue", "aucune décision"
OUTCOMES = (AUTO_OK, AUTO_KO, IN_REVIEW, NO_DECISION)
ERROR_TYPES = ("paiement jamais imputé", "mauvais débiteur", "bon débiteur, autre facture", "ensemble incomplet",
               "factures en trop", "ensemble en partie juste")
NO_TRUTH = "sans imputation réelle"
REASONS = (
    "proposition juste, score sous τ_high",
    "proposition juste, non auto-validable (marge au second, ou type sans seuil)",
    "aucun débiteur alloué",
    "vrai débiteur absent de l'allocation",
    "aucune règle applicable sans ambiguïté",
    "aucun candidat ML",
    "vraie facture absente des candidats (débiteur non alloué)",
    "vraie facture absente des candidats",
    "ensemble : factures manquantes dans les candidats",
    "ensemble candidat, non reconstitué",
    "vraie facture candidate, pas en tête",
    "proposition fausse malgré la vraie facture en tête",
    "vraie facture en tête, score sous τ_low",
    NO_TRUTH,
)
SEGMENTS = {
    "Tranche de montant": "tranche_montant", "Canal": "canal", "Type de paiement": "type_paiement",
    "Bankroll": "bankroll", "IBAN": "iban", "Route IBAN": "route_iban", "Client file": "client_file",
    "Allocation": "allocation", "Type de groupe réel": "type_groupe", "Marché": "marché", "Produit": "produit",
    "Taille du débiteur (factures)": "taille_débiteur", "Délai d'imputation": "délai_imputation",
    "Mois d'arrivée": "mois", "Étape de décision": "étape", "Règle": "règle",
}
AMOUNT_EDGES = [-np.inf, 100, 1_000, 10_000, 100_000, np.inf]
AMOUNT_LABELS = ["< 100 €", "100 – 1 000 €", "1 000 – 10 000 €", "10 000 – 100 000 €", "≥ 100 000 €"]
SIZE_EDGES = [0, 10, 100, 1_000, np.inf]
SIZE_LABELS = ["1 – 10", "11 – 100", "101 – 1 000", "> 1 000"]
DELAY_EDGES = [-np.inf, 1, 7, 30, 60, np.inf]
DELAY_LABELS = ["≤ 1 j", "2 – 7 j", "8 – 30 j", "31 – 60 j", "> 60 j"]
MISSING = "non renseigné"


def _pairs(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """(payment_id, invoice_id) depuis une colonne de tuples de factures."""
    f = frame[frame[column].map(lambda v: isinstance(v, (tuple, list, np.ndarray)) and len(v) > 0)]
    lengths = f[column].map(len).to_numpy(dtype=np.int64)
    return pd.DataFrame({"payment_id": np.repeat(f["payment_id"].to_numpy(dtype=object), lengths),
                         "invoice_id": [i for t in f[column] for i in t]})


def _joined(pairs: pd.DataFrame, value: str) -> pd.Series:
    """Valeurs distinctes de `value` par paiement, jointes par « · » (ordre d'apparition)."""
    d = pairs[["payment_id", value]].dropna().drop_duplicates().sort_values("payment_id", kind="mergesort")
    keys, groups = _tuples_by_key(d["payment_id"].to_numpy(dtype=object), d[value].astype(str).to_numpy(dtype=object))
    return pd.Series([" · ".join(g) for g in groups], index=keys, dtype=object)


def _error_type(pred: tuple, truth, pred_debtors: set, truth_debtors: set) -> str:
    if not isinstance(truth, tuple):
        return ERROR_TYPES[0]
    p, t = set(pred), set(truth)
    if p < t:
        return "ensemble incomplet"
    if t < p:
        return "factures en trop"
    if p & t:
        return "ensemble en partie juste"
    return "bon débiteur, autre facture" if set(pred_debtors) & set(truth_debtors) else "mauvais débiteur"


def _candidate_stats(candidates: pd.DataFrame, true_pairs: pd.DataFrame) -> pd.DataFrame:
    """Par paiement passé en ML : candidats, score du premier, rang et score de la meilleure vraie facture."""
    c = candidates[["payment_id", "invoice_id", "p"]].merge(true_pairs.assign(vraie=True), how="left",
                                                           on=["payment_id", "invoice_id"])
    c["vraie"] = c["vraie"].fillna(False).astype(bool)
    c = c.sort_values(["payment_id", "p"], ascending=[True, False], kind="mergesort")
    c["rang"] = c.groupby("payment_id", sort=False).cumcount() + 1
    g = c.groupby("payment_id", sort=False)
    out = pd.DataFrame({"candidats": g.size(), "score_premier": g["p"].first(), "premier_vrai": g["vraie"].first(),
                        "vraies_candidates": g["vraie"].sum()})
    true = c[c["vraie"]].groupby("payment_id", sort=False)
    out["rang_vraie"] = true["rang"].min()
    out["score_vraie"] = true["p"].max()
    return out


def analyze_payments(decisions: pd.DataFrame, truth: pd.DataFrame, scope: pd.DataFrame, payment: pd.DataFrame,
                     invoice: pd.DataFrame, agreement: pd.DataFrame, imputation: pd.DataFrame,
                     candidates: pd.DataFrame | None = None, info: pd.DataFrame | None = None,
                     thresholds: dict | None = None) -> pd.DataFrame:
    """Une ligne par paiement du périmètre (voir la docstring du module).

    `payment` : payment_id, amount, label, channel, payment_type, bankroll_code, iban_debtor ;
    `invoice` : invoice_id, debtor_id, agreement_id, client_reference ; `agreement` : agreement_id, market,
    product ; `imputation` : payment_id, updated_at ; `candidates` : premier passage en ML (payment_id,
    invoice_id, p) ; `info` : allocation au jour d'arrivée (client_file_id, allocation_status, iban_route,
    allocation_debtors) ; `thresholds` : seuils du modèle (kinds, tau_high)."""
    props = _proposals(decisions)
    p = (scope[["payment_id", "arrival_day"]].merge(truth, on="payment_id", how="left")
         .merge(props, on="payment_id", how="left"))
    has_truth = p["truth_invoices"].notna()
    same = pd.Series([isinstance(a, tuple) and isinstance(b, tuple) and a == b
                      for a, b in zip(p["invoices"], p["truth_invoices"])], index=p.index)
    auto, review = p["action"] == AUTO, p["action"] == REVIEW
    p["issue"] = np.select([auto & same, auto, review], [AUTO_OK, AUTO_KO, IN_REVIEW], NO_DECISION)
    p["proposition_juste"] = same & p["action"].notna()

    # Attributs du paiement.
    ids = set(p["payment_id"])
    pay = payment[payment["payment_id"].isin(ids)].drop_duplicates("payment_id").set_index("payment_id")
    p["montant"] = p["payment_id"].map(pay["amount"]).astype("Float64") / 100
    p["tranche_montant"] = pd.cut(p["montant"].astype(float), AMOUNT_EDGES, labels=AMOUNT_LABELS, right=False) \
        .astype(object).fillna(MISSING)
    p["libellé"] = p["payment_id"].map(pay["label"])
    for col, src in (("canal", "channel"), ("type_paiement", "payment_type"), ("bankroll", "bankroll_code")):
        p[col] = p["payment_id"].map(pay[src]).astype(object).fillna(MISSING) if src in pay else MISSING
    p["iban"] = np.where(p["payment_id"].map(pay["iban_debtor"]).notna(), "IBAN présent", "IBAN absent") \
        if "iban_debtor" in pay else MISSING
    p["mois"] = pd.to_datetime(p["arrival_day"]).dt.strftime("%Y-%m")
    p["type_groupe"] = p["group_type"].fillna(NO_TRUTH)
    p["étape"] = p["step"].fillna("—")
    p["règle"] = p["rule_id"].fillna("—")

    first_imp = imputation[imputation["payment_id"].isin(ids)].groupby("payment_id")["updated_at"].min()
    delay = (p["payment_id"].map(first_imp) - pd.to_datetime(p["arrival_day"])).dt.total_seconds() / 86_400
    p["délai_imputation"] = pd.cut(delay, DELAY_EDGES, labels=DELAY_LABELS).astype(object).fillna("jamais imputé")

    # Factures réelles et proposées : débiteurs, références, contrat.
    inv_cols = [c for c in ("invoice_id", "debtor_id", "agreement_id", "client_reference") if c in invoice]
    true_pairs = _pairs(p, "truth_invoices")
    look = invoice[inv_cols][invoice["invoice_id"].isin(set(true_pairs["invoice_id"]))]
    tp = true_pairs.merge(look, on="invoice_id", how="left")
    p["débiteur_réel"] = p["payment_id"].map(_joined(tp, "debtor_id"))
    p["factures_réelles"] = p["payment_id"].map(_joined(tp, "client_reference"))
    agr = agreement.drop_duplicates("agreement_id").set_index("agreement_id")
    first = tp.drop_duplicates("payment_id").set_index("payment_id")
    for col, src in (("marché", "market"), ("produit", "product")):
        p[col] = p["payment_id"].map(first["agreement_id"].map(agr[src])).astype(object).fillna(MISSING) \
            if src in agr else MISSING
    sizes = invoice["debtor_id"].value_counts()
    n_inv = p["payment_id"].map(first["debtor_id"].map(sizes))
    p["taille_débiteur"] = pd.cut(n_inv.astype(float), SIZE_EDGES, labels=SIZE_LABELS).astype(object).fillna(MISSING)

    shown = p[p["issue"] != AUTO_OK]
    pred_pairs = _pairs(shown, "invoices")
    pp = pred_pairs.merge(invoice[inv_cols][invoice["invoice_id"].isin(set(pred_pairs["invoice_id"]))],
                          on="invoice_id", how="left")
    p["factures_proposées"] = p["payment_id"].map(_joined(pp, "client_reference"))
    p["débiteur_proposé"] = p["payment_id"].map(_joined(pp, "debtor_id"))

    # Type d'erreur des auto-validations fausses.
    wrong = p["issue"] == AUTO_KO
    wrong_ids = set(p.loc[wrong, "payment_id"])
    pred_deb = pp[pp["payment_id"].isin(wrong_ids)].groupby("payment_id")["debtor_id"].agg(set)
    true_deb = tp[tp["payment_id"].isin(wrong_ids)].groupby("payment_id")["debtor_id"].agg(set)
    p["type_erreur"] = ""
    p.loc[wrong, "type_erreur"] = [
        _error_type(pred, t, pred_deb.get(pid, set()), true_deb.get(pid, set()))
        for pid, pred, t in zip(p.loc[wrong, "payment_id"], p.loc[wrong, "invoices"], p.loc[wrong, "truth_invoices"])]

    # Allocation au jour d'arrivée.
    allocated = pd.Series(np.nan, index=p.index, dtype=object)
    if info is not None and len(info):
        inf = info.drop_duplicates("payment_id").set_index("payment_id")
        p["client_file"] = np.where(p["payment_id"].map(inf["client_file_id"]).notna(), "avec client file",
                                    "sans client file")
        p["allocation"] = p["payment_id"].map(inf["allocation_status"]).astype(object).fillna(MISSING)
        p["route_iban"] = p["payment_id"].map(inf["iban_route"]).astype(object).fillna(MISSING) \
            if "iban_route" in inf else MISSING
        if "allocation_debtors" in inf:
            alloc_pairs = _pairs(inf.reset_index()[["payment_id", "allocation_debtors"]], "allocation_debtors") \
                .rename(columns={"invoice_id": "debtor_id"})
            found = tp[["payment_id", "debtor_id"]].merge(alloc_pairs, on=["payment_id", "debtor_id"])
            known = p["payment_id"].isin(set(inf.index))
            allocated = pd.Series(np.where(known, p["payment_id"].isin(set(found["payment_id"])), np.nan),
                                  index=p.index, dtype=object)
            n_alloc = p["payment_id"].map(inf["allocation_debtors"].map(len))
            p["débiteurs_alloués"] = n_alloc
    else:
        p["client_file"] = p["allocation"] = p["route_iban"] = MISSING
    p["vrai_débiteur_alloué"] = allocated

    # Candidats ML au premier passage.
    ml = candidates is not None
    if ml:
        stats = _candidate_stats(candidates[candidates["payment_id"].isin(ids)], true_pairs)
        for col in stats.columns:
            p[col] = p["payment_id"].map(stats[col])

    p["raison"] = _reasons(p, ml, thresholds)
    p["invoices"] = p["invoices"].map(lambda v: list(v) if isinstance(v, tuple) else [])
    p["truth_invoices"] = p["truth_invoices"].map(lambda v: list(v) if isinstance(v, tuple) else [])
    return p.drop(columns=["group_type", "step", "rule_id"]).rename(
        columns={"invoices": "factures_proposées_id", "truth_invoices": "factures_réelles_id", "n_invoices":
                 "nb_factures_réelles", "day": "jour_décision"})


def _reasons(p: pd.DataFrame, ml: bool, thresholds: dict | None) -> pd.Series:
    """Premier blocage rencontré par chaque paiement non auto-validé (vide sinon)."""
    out = pd.Series("", index=p.index, dtype=object)
    residual = p["issue"].isin([IN_REVIEW, NO_DECISION])
    no_truth = p["truth_invoices"].isna()
    out[residual & no_truth] = NO_TRUTH
    todo = residual & ~no_truth
    kinds = (thresholds or {}).get("kinds", {})
    default_tau = (thresholds or {}).get("tau_high", 1.01)
    right = todo & p["proposition_juste"]
    if right.any():
        # Type de proposition ML dans l'identifiant de règle (ML_SINGLE, ML_SET, ML_PARTIAL, ML_NN).
        kind = p.loc[right, "rule_id"].fillna("").str.removeprefix("ML_").str.lower()
        tau = kind.map(lambda k: kinds.get(k, default_tau)).astype(float)
        blocked = (kind == "nn") | (tau > 1) | (p.loc[right, "score"].fillna(0) >= tau)
        out[right] = np.where(blocked, REASONS[1], REASONS[0])
    todo &= ~right
    status = p["allocation"] if "allocation" in p else pd.Series(MISSING, index=p.index)
    allocated = p["vrai_débiteur_alloué"]
    none_alloc = status.isin(["aucun", "aucune", "none"]) | (p.get("débiteurs_alloués", pd.Series(np.nan, index=p.index)) == 0)
    if not ml:
        out[todo & none_alloc] = REASONS[2]
        out[todo & ~none_alloc & (allocated == False)] = REASONS[3]  # noqa: E712
        out[todo & (out == "")] = REASONS[4]
        return out
    in_ml = p["candidats"].notna()
    out[todo & ~in_ml & none_alloc] = REASONS[2]
    out[todo & ~in_ml & ~none_alloc] = REASONS[5]
    todo &= in_ml
    n_inv = p["n_invoices"].fillna(0)
    complete = p["vraies_candidates"].fillna(0) >= n_inv
    some = p["vraies_candidates"].fillna(0) > 0
    out[todo & ~complete & some] = REASONS[8]
    out[todo & ~complete & ~some & (allocated == False)] = REASONS[6]  # noqa: E712
    out[todo & ~complete & ~some & (allocated != False)] = REASONS[7]  # noqa: E712
    todo &= complete
    out[todo & (n_inv > 1)] = REASONS[9]
    todo &= n_inv <= 1
    top = p["premier_vrai"].fillna(False).astype(bool)
    out[todo & ~top] = REASONS[10]
    out[todo & top & (p["issue"] == IN_REVIEW)] = REASONS[11]
    out[todo & top & (p["issue"] == NO_DECISION)] = REASONS[12]
    return out


# --- Tableaux ------------------------------------------------------------------------------------------------

def rate_by(p: pd.DataFrame, column: str) -> pd.DataFrame:
    """Par valeur d'un segment : paiements, auto-validés, justes, précision et sa borne basse, automatisation."""
    g = p.assign(_auto=p["issue"].isin([AUTO_OK, AUTO_KO]), _ok=p["issue"] == AUTO_OK,
                 _review=p["issue"] == IN_REVIEW).groupby(column, dropna=False, sort=True)
    out = pd.DataFrame({"paiements": g.size(), "auto": g["_auto"].sum(), "justes": g["_ok"].sum(),
                        "revue": g["_review"].sum()}).reset_index()
    out["part des paiements"] = out["paiements"] / max(len(p), 1)
    out["taux d'automatisation"] = out["auto"] / out["paiements"]
    out["précision"] = out["justes"] / out["auto"].where(out["auto"] > 0)
    out["précision min. 95 %"] = [clopper_pearson_lower(int(a), int(c)) for a, c in zip(out["auto"], out["justes"])]
    return out


def reason_table(p: pd.DataFrame) -> pd.DataFrame:
    """Résiduel : paiements par raison, et ce que chaque levier rapporterait au plus (points d'automatisation)."""
    r = p[p["raison"] != ""]
    out = r["raison"].value_counts().rename("paiements").rename_axis("raison").reset_index()
    out["part du résiduel"] = out["paiements"] / max(len(r), 1)
    out["gain maximal (points)"] = np.where(out["raison"] == NO_TRUTH, np.nan, 100 * out["paiements"] / max(len(p), 1))
    order = {r: i for i, r in enumerate(REASONS)}
    return out.assign(_o=out["raison"].map(order)).sort_values("_o").drop(columns="_o").reset_index(drop=True)


def error_table(p: pd.DataFrame, by: str = "règle") -> pd.DataFrame:
    """Auto-validations fausses par type d'erreur et par règle (ou étape)."""
    wrong = p[p["issue"] == AUTO_KO]
    if wrong.empty:
        return pd.DataFrame(columns=["type d'erreur", "total"])
    table = pd.crosstab(wrong["type_erreur"], wrong[by])
    table["total"] = table.sum(axis=1)
    order = [e for e in ERROR_TYPES if e in table.index]
    return table.loc[order].rename_axis("type d'erreur").reset_index()


def compare_outcomes(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    """Issue de chaque paiement sous deux rapprocheurs (ex. règles seules → règles + ML), mêmes paiements."""
    m = before[["payment_id", "issue"]].merge(after[["payment_id", "issue"]], on="payment_id", suffixes=(" avant", " après"))
    table = pd.crosstab(m["issue avant"], m["issue après"])
    table = table.reindex(index=[o for o in OUTCOMES if o in table.index],
                          columns=[o for o in OUTCOMES if o in table.columns], fill_value=0)
    return table


def summary(p: pd.DataFrame) -> dict:
    n = len(p)
    counts = p["issue"].value_counts()
    auto = int(counts.get(AUTO_OK, 0) + counts.get(AUTO_KO, 0))
    return {"paiements": n, **{k: int(counts.get(k, 0)) for k in OUTCOMES},
            "taux_automatisation": auto / n if n else 0.0,
            "précision": counts.get(AUTO_OK, 0) / auto if auto else None,
            "précision_min_95": clopper_pearson_lower(auto, int(counts.get(AUTO_OK, 0))) if auto else None,
            "erreurs_par_type": p.loc[p["issue"] == AUTO_KO, "type_erreur"].value_counts().to_dict(),
            "résiduel_par_raison": p.loc[p["raison"] != "", "raison"].value_counts().to_dict()}
