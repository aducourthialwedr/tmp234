"""Diagnostics des données et des résultats pour l'interface, repris des outils `tools/diag_*.py` du notebook.

Chaque fonction lit les sorties de l'étape 1 (`data/interim/...`) ou un export de décisions, colonne par colonne,
et renvoie des tableaux prêts à afficher. Rien n'est écrit. Les volumes réels (millions de lignes) restent
supportables : seules les colonnes utiles sont lues.

- `open_invoice_stock` : stock de factures ouvertes mois par mois (une facture jamais imputée reste ouverte) ;
- `pending_batches` : taille du lot quotidien (paiements du jour + reliquat) ;
- `imputation_delays` : délai arrivée → imputation, comparé à la purge et à la rétention (fuite F3) ;
- `client_file_conditions` : à quelle condition le rattachement d'un client file échoue ;
- `iban_worst_case` : pire cas de la fuite F2 (IBAN connus avant le paiement qui les a révélés) ;
- `clopper_pearson_lower`, `with_precision_bounds` : borne basse de la précision des auto-validations ;
- `threshold_curve` : volume et précision des auto-validations ML selon τ_high, sur la boucle de validation ;
- `importance_by_family` : gain LightGBM par famille de features.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

from src.timeline.split import Split, assign_period, compute_split

DAY = np.timedelta64(1, "D")


def _utc(values) -> pd.Series:
    return pd.to_datetime(values, utc=True).dt.tz_localize(None)


def _read(interim: Path, table: str, columns: list[str], filters=None) -> pd.DataFrame:
    import pyarrow.parquet as pq
    path = Path(interim) / f"{table}.parquet"
    present = set(pq.read_schema(path).names)
    return pd.read_parquet(path, columns=[c for c in columns if c in present], filters=filters)


def arrivals(interim: Path) -> pd.DataFrame:
    """payment_id et jour d'arrivée (booking_date, sinon value_date), comme le journal."""
    pay = _read(interim, "payment", ["payment_id", "value_date", "booking_date"])
    day = pay["booking_date"].fillna(pay["value_date"]) if "booking_date" in pay else pay["value_date"]
    return pd.DataFrame({"payment_id": pay["payment_id"], "day": _utc(day).dt.normalize()})


def data_split(interim: Path, split_settings) -> Split:
    """Découpage temporel sur la plage des paiements chargés (comme `Project.split`)."""
    days = arrivals(interim)["day"]
    return compute_split(split_settings, days.min().date(), days.max().date())


def _imputations(interim: Path) -> pd.DataFrame:
    """Lignes d'imputation datées avec leur montant imputé (tel que le journal les applique)."""
    j = _read(interim, "journal", ["ts", "event_type", "entity_id", "related_id", "amount"],
              filters=[("event_type", "==", "IMPUTATION_APPLIED")])
    return pd.DataFrame({"payment_id": j["entity_id"], "invoice_id": j["related_id"], "ts": _utc(j["ts"]),
                         "amount": j["amount"].astype("float64")})


# --- Volumétrie : ce qui grossit avec la longueur de l'historique ---------------------------------------------

def open_invoice_stock(interim: Path, split: Split) -> dict[str, pd.DataFrame]:
    """Stock de factures ouvertes en début de mois, et âge du stock au début du test.

    Le restant dû n'est reconstruit que par les imputations : une facture jamais (totalement) imputée reste
    ouverte pour toujours et alourdit chaque jour de replay (règle R3, allocation par montant, candidats ML)."""
    inv = _read(interim, "invoice", ["invoice_id", "creation_date", "due_date", "initial_amount", "debtor_id"])
    inv["creation_date"], inv["due_date"] = _utc(inv["creation_date"]), _utc(inv["due_date"])
    imp = _imputations(interim).merge(inv[["invoice_id", "initial_amount"]], on="invoice_id")
    imp = imp.sort_values(["invoice_id", "ts"], kind="mergesort")
    imp["cum"] = imp.groupby("invoice_id")["amount"].cumsum()
    closed_at = imp[imp["cum"] >= imp["initial_amount"]].groupby("invoice_id")["ts"].min()
    inv["closed_at"] = inv["invoice_id"].map(closed_at)
    inv["ever_imputed"] = inv["invoice_id"].map(imp.groupby("invoice_id")["ts"].min()).notna()

    months = pd.date_range(inv["creation_date"].min().normalize(), pd.Timestamp(split.periods[-1].end), freq="MS")
    created = np.sort(inv["creation_date"].to_numpy())
    closed = np.sort(inv["closed_at"].dropna().to_numpy())
    curve = pd.DataFrame({"mois": months, "créées": np.searchsorted(created, months.to_numpy(), side="left"),
                          "soldées": np.searchsorted(closed, months.to_numpy(), side="left")})
    curve["ouvertes"] = curve["créées"] - curve["soldées"]

    t0 = pd.Timestamp(split.period("test").start)
    stock = inv[(inv["creation_date"] < t0) & ~(inv["closed_at"] < t0)]
    late = (t0 - stock["due_date"]).dt.days
    labels = ["pas encore échue", "0–90 j de retard", "90–180 j", "180–365 j", "1–2 ans", "> 2 ans"]
    bucket = pd.cut(late, [-10**7, 0, 90, 180, 365, 730, 10**7], labels=labels)
    age = pd.DataFrame({"factures": bucket.value_counts().reindex(labels).fillna(0).astype(int)})
    age["part"] = age["factures"] / max(len(stock), 1)
    age["jamais imputées"] = stock.groupby(bucket, observed=False)["ever_imputed"].apply(lambda s: int((~s).sum())) \
        .reindex(labels).fillna(0).astype(int)
    per_debtor = stock.groupby("debtor_id").size()
    debtors = pd.Series({"débiteurs avec factures ouvertes": len(per_debtor),
                         "factures ouvertes par débiteur (médiane)": float(per_debtor.median()) if len(per_debtor) else 0,
                         "P99": float(per_debtor.quantile(0.99)) if len(per_debtor) else 0,
                         "maximum": int(per_debtor.max()) if len(per_debtor) else 0,
                         "part portée par les 10 plus gros": float(per_debtor.nlargest(10).sum() / max(len(stock), 1))})
    return {"curve": curve, "age": age.rename_axis("retard au début du test").reset_index(), "debtors": debtors,
            "total": pd.Series({"factures": len(inv), "ouvertes au début du test": len(stock), "début du test": t0})}


def imputation_delays(interim: Path, purge_days: int, retention_days: int) -> dict:
    """Délai entre l'arrivée d'un paiement et sa première imputation (jours)."""
    pay = arrivals(interim)
    first = _imputations(interim).groupby("payment_id")["ts"].min()
    delay = (pay["payment_id"].map(first) - pay["day"]) / DAY
    known = delay.dropna()
    return {"describe": known.describe(percentiles=[0.5, 0.9, 0.95, 0.99]),
            "jamais imputés": float(delay.isna().mean()),
            "au-delà de la purge": float((known > purge_days).mean()) if len(known) else 0.0,
            "au-delà de la rétention": float((known > retention_days).mean()) if len(known) else 0.0,
            "purge": purge_days, "rétention": retention_days}


def pending_batches(interim: Path, split: Split, retention_days: int, every_days: int = 7) -> pd.DataFrame:
    """Taille du lot quotidien (majorant : sans les auto-validations du moteur, qui sortent aussi du reliquat),
    un jour tous les `every_days` sur la validation et le test."""
    pay = arrivals(interim)
    first = _imputations(interim).groupby("payment_id")["ts"].min()
    a = pay["day"].to_numpy()
    f = pay["payment_id"].map(first).fillna(pd.Timestamp.max.normalize()).to_numpy()
    rows = []
    for p in split.periods:
        if p.name == "train":
            continue
        for day in pd.date_range(p.start, p.end, freq=f"{every_days}D"):
            d = np.datetime64(day)
            window = (a <= d) & (a >= d - retention_days * DAY)
            rows.append({"période": p.name, "jour": day, "paiements du jour": int((a == d).sum()),
                         "lot (majorant)": int((window & (f >= d)).sum())})
    return pd.DataFrame(rows)


# --- Client files ------------------------------------------------------------------------------------------

def client_file_conditions(interim: Path, date_tolerance_days: int) -> dict | None:
    """Pour chaque client file dont le vrai paiement est retrouvé (via les factures citées et les imputations
    réelles), quelles conditions du rattachement sont remplies. None : pas de client files chargés."""
    base = Path(interim)
    if not (base / "client_file.parquet").exists() or not (base / "client_file_line.parquet").exists():
        return None
    cf = _read(base, "client_file", ["file_id", "received_at", "total_amount", "payment_date", "iban",
                                     "payment_reference", "payment_reference_norm"])
    cfl = _read(base, "client_file_line", ["file_id", "invoice_reference", "invoice_reference_norm", "amount"])
    inv = _read(base, "invoice", ["invoice_id", "client_reference_norm", "internal_reference_norm"])
    imp = _read(base, "imputation", ["payment_id", "invoice_id"])
    pay = _read(base, "payment", ["payment_id", "amount", "value_date", "iban_debtor", "label_norm"])
    day = lambda s: _utc(s).dt.normalize()  # noqa: E731

    fill = {f"client_file.{c}": cf[c].notna().mean() for c in
            ("received_at", "total_amount", "payment_date", "iban", "payment_reference") if c in cf}
    fill |= {f"client_file_line.{c}": cfl[c].notna().mean() for c in ("invoice_reference", "amount") if c in cfl}

    compact = lambda s: s.fillna("").astype(str).str.replace(" ", "", regex=False)  # noqa: E731
    ref_inv = pd.concat([pd.DataFrame({"ref": compact(inv[c]), "invoice_id": inv["invoice_id"]})
                         for c in ("client_reference_norm", "internal_reference_norm") if c in inv])
    ref_inv = ref_inv[ref_inv["ref"].str.len() >= 4].drop_duplicates()
    lines = cfl.assign(ref=compact(cfl["invoice_reference_norm"]))
    lines["n"] = lines["ref"].map(ref_inv.groupby("ref").size()).fillna(0).astype(int)
    refs = lines["n"].clip(upper=2).map({0: "introuvable", 1: "une facture", 2: "plusieurs (ambiguë)"}) \
        .value_counts(normalize=True)

    hit = lines[lines["n"] == 1].merge(ref_inv, on="ref").merge(imp, on="invoice_id")
    true_pay = (hit.groupby(["file_id", "payment_id"]).size().rename("lignes").reset_index()
                .sort_values("lignes", ascending=False).drop_duplicates("file_id"))
    f = cf.merge(true_pay, on="file_id").merge(pay, on="payment_id")
    f = f.join(cfl.groupby("file_id")["amount"].sum(min_count=1).rename("somme_lignes"), on="file_id")
    f = f.rename(columns={"amount": "montant_paiement"})
    exact = f["total_amount"] == f["montant_paiement"]
    date_ok = (day(f["payment_date"]) - day(f["value_date"])).dt.days.abs() <= date_tolerance_days
    iban_ok = f["iban"].notna() & (f["iban"].astype(object) == f["iban_debtor"].astype(object))
    ref_ok = pd.Series([isinstance(r, str) and bool(r) and r in (lbl or "")
                        for r, lbl in zip(f.get("payment_reference_norm", pd.Series(dtype=object)), f["label_norm"])],
                       index=f.index, dtype=bool)
    corroborated = date_ok | iban_ok | ref_ok
    shared = f["total_amount"].map(pay["amount"].value_counts()).fillna(0) > 1
    conditions = pd.Series({
        "montant total = montant du paiement (exact)": exact.mean(),
        "total_amount vide": f["total_amount"].isna().mean(),
        "somme des lignes = montant du paiement": (f["somme_lignes"] == f["montant_paiement"]).mean(),
        f"date du fichier à ± {date_tolerance_days} j de la date de valeur": date_ok.mean(),
        "IBAN du fichier = IBAN du paiement": iban_ok.mean(),
        "référence de virement dans le libellé": ref_ok.mean(),
        "au moins une corroboration": corroborated.mean(),
        "montant exact et corroboré : rattachable": (exact & corroborated).mean(),
        "dont montant partagé par plusieurs paiements (ambiguïté)": (exact & corroborated & shared).mean(),
    }) if len(f) else pd.Series(dtype=float)
    gap = (f["total_amount"] - f["montant_paiement"])[~exact & f["total_amount"].notna()] / 100
    return {"fichiers": len(cf), "lignes": len(cfl), "vrai paiement retrouvé": len(true_pay),
            "remplissage": pd.Series(fill), "références des lignes": refs, "conditions": conditions,
            "délai de réception (j)": (day(f["received_at"]) - day(f["value_date"])).dt.days.describe(
                percentiles=[0.05, 0.5, 0.95]),
            "écart total − paiement (€)": gap.describe(percentiles=[0.05, 0.5, 0.95])}


# --- Fuite F2 : IBAN non datés -----------------------------------------------------------------------------

def iban_worst_case(interim: Path, split: Split, decisions: Path | None = None) -> dict:
    """Part des paiements, et des auto-validations si l'export de décisions est donné, qui reposent peut-être sur
    un IBAN révélé par le paiement lui-même (premier paiement imputé de cet IBAN pour ce débiteur). Majorant."""
    base = Path(interim)
    pay = arrivals(base).merge(_read(base, "payment", ["payment_id", "iban_debtor"]), on="payment_id")
    pay["period"] = assign_period(pay["day"], split).to_numpy()
    truth = (_read(base, "imputation", ["payment_id", "invoice_id"])
             .merge(_read(base, "invoice", ["invoice_id", "debtor_id"]), on="invoice_id")[["payment_id", "debtor_id"]]
             .drop_duplicates())
    pdx = pay[pay["iban_debtor"].notna()].rename(columns={"iban_debtor": "iban"}).merge(truth, on="payment_id")
    if (base / "party_iban.parquet").exists():
        links = _read(base, "party_iban", ["role", "party_id", "iban"])
        links = links[links["role"] == "debtor"]
    else:
        links = _read(base, "debtor", ["party_id", "iban"]).dropna()
    links = links[["iban", "party_id"]].drop_duplicates().rename(columns={"party_id": "debtor_id"}).assign(known=True)
    pdx = pdx.merge(links, on=["iban", "debtor_id"], how="left")
    pdx["known"] = pdx["known"].fillna(False).astype(bool)
    pdx["revealed"] = pdx["known"] & (pdx["day"] == pdx.groupby(["iban", "debtor_id"])["day"].transform("min"))
    per_pay = pdx.groupby("payment_id").agg(period=("period", "first"), known=("known", "any"),
                                            revealed=("revealed", "any"))
    by_period = per_pay.groupby("period")[["known", "revealed"]].mean().rename(
        columns={"known": "IBAN connu du référentiel", "revealed": "IBAN révélé par ce paiement (pire cas)"})
    out = {"par période": by_period}
    if decisions is not None and Path(decisions).exists():
        d = pd.read_parquet(decisions, columns=["payment_id", "action", "step", "decision_retenue",
                                                "decision_retenue_exacte"])
        auto = d[d["decision_retenue"] & (d["action"] == "auto")].drop_duplicates("payment_id")
        auto = auto.assign(revealed=auto["payment_id"].map(per_pay["revealed"]).fillna(False).astype(bool),
                           ok=auto["decision_retenue_exacte"].astype(float))
        out["auto-validations"] = auto.groupby(["step", "revealed"]).agg(auto_validations=("payment_id", "size"),
                                                                         précision=("ok", "mean")).reset_index()
        out["part révélée"] = float(auto["revealed"].mean()) if len(auto) else 0.0
    return out


# --- Précision des auto-validations ------------------------------------------------------------------------

def clopper_pearson_lower(n: int, k: int, confidence: float = 0.95) -> float:
    """Borne basse unilatérale de la précision k / n : la vraie précision est au-dessus avec cette confiance."""
    if n <= 0:
        return float("nan")
    return float(beta.ppf(1 - confidence, k, n - k + 1)) if k > 0 else 0.0


def with_precision_bounds(table: pd.DataFrame, n: str = "auto", k: str = "corrects") -> pd.DataFrame:
    """Ajoute la borne basse à 95 % (« précision min. 95 % ») à un tableau de comptes (auto, corrects)."""
    out = table.copy()
    out["précision min. 95 %"] = [clopper_pearson_lower(int(a), int(c)) for a, c in zip(out[n], out[k])]
    return out


def threshold_curve(proposals: Path, min_margin: float, points: int = 200,
                    applied: dict[str, float] | None = None) -> pd.DataFrame:
    """Volume et précision des auto-validations ML selon τ_high, par type de proposition, sur la boucle de validation
    enregistrée à la calibration (`<interim>/ml/validation_daily_proposals.parquet`).

    Même règle que `calibrate_online` : un paiement rescoré chaque jour est auto-validé le premier jour où son score
    (marge ≥ δ) franchit τ ; c'est la proposition de ce jour-là qui compte. `applied` (type → τ) : seuils ajoutés à
    la grille, marqués dans la colonne `appliqué`."""
    applied = applied or {}
    r = pd.read_parquet(proposals, columns=["day", "payment_id", "kind", "score", "margin", "correct"])
    rows = []
    for kind, k in r.groupby("kind", sort=True):
        k = k.assign(s=np.where(k["margin"] >= min_margin, k["score"], -np.inf))
        k = k.sort_values(["payment_id", "day"], kind="mergesort")
        best = k.groupby("payment_id")["s"].cummax()
        prev = best.groupby(k["payment_id"]).shift(1).fillna(-np.inf).to_numpy()
        s, ok = k["s"].to_numpy(), k["correct"].to_numpy()
        setter = s > prev                                   # jours qui portent un nouveau maximum
        s, prev, ok = s[setter], prev[setter], ok[setter]
        finite = np.unique(s[np.isfinite(s)])
        if not len(finite):
            continue
        taus = np.quantile(finite, np.linspace(0, 1, points), method="nearest")
        chosen = applied.get(kind)
        if chosen is not None and chosen <= 1:
            taus = np.append(taus, chosen)
        for tau in np.unique(taus):
            m = (prev < tau) & (tau <= s)
            n, c = int(m.sum()), int(ok[m].sum())
            rows.append({"type": kind, "τ_high": float(tau), "auto-validations": n, "corrects": c,
                         "précision": c / n if n else np.nan, "précision min. 95 %": clopper_pearson_lower(n, c),
                         "appliqué": bool(chosen is not None and tau == chosen)})
    return pd.DataFrame(rows, columns=["type", "τ_high", "auto-validations", "corrects", "précision",
                                       "précision min. 95 %", "appliqué"])


def importance_by_family(importance: pd.DataFrame) -> pd.DataFrame:
    """Gain LightGBM (feature, gain) agrégé par famille de features (`FAMILIES`, plus base et compétition)."""
    from src.reconcile_ml.features import BASE, COMPETITION, FAMILIES
    family = {f: name for name, cols in FAMILIES.items() for f in cols}
    family |= {f: "base (sources, rang du montant)" for f in BASE} | {f: "compétition (pass 2)" for f in COMPETITION}
    out = importance.assign(famille=importance["feature"].map(family).fillna("autre"))
    out = out.groupby("famille").agg(features=("feature", "size"), gain=("gain", "sum")).reset_index()
    out["part du gain"] = out["gain"] / out["gain"].sum()
    return out.sort_values("gain", ascending=False, ignore_index=True)
