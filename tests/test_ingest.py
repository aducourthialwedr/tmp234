"""Ingestion assistée (src/load/ingest.py) : lecture, détection du format, contrôle du mapping, validation complète."""

import copy

import pandas as pd

from src.config import REPO_ROOT, read_yaml
from src.load import ingest as I

SCHEMA = read_yaml(REPO_ROOT / "config" / "schema.synthetic.yaml")


def _payments_csv(path, sep=";", encoding="latin-1"):
    df = pd.DataFrame({"ref": ["P1", "P2", "P3", "P3"], "date_valeur": ["01/03/2024", "02/03/2024", "pas une date",
                                                                        "03/03/2024"],
                       "montant": ["12,50", "1 000,00", "abc", "7,1"], "devise": "EUR",
                       "libellé": ["VIR Société Générale", "VIR Été", "", "VIR X"]})
    df.to_csv(path, sep=sep, index=False, encoding=encoding)
    return path


def test_sniff_and_read_sample(tmp_path):
    path = _payments_csv(tmp_path / "pay.csv")
    options = I.sniff_csv(path)
    assert options == {"sep": ";", "encoding": "latin-1"}
    sample = I.read_sample(path, options)
    assert list(sample.columns) == ["ref", "date_valeur", "montant", "devise", "libellé"]
    assert sample.loc[1, "libellé"] == "VIR Été" and I.count_rows(path) == 4
    profile = I.profile_columns(sample).set_index("colonne")
    assert profile.loc["libellé", "remplie"] == 0.75 and profile.loc["ref", "distinctes"] == 3


def test_check_fields_statuses(tmp_path):
    sample = I.read_sample(_payments_csv(tmp_path / "pay.csv"), {"sep": ";", "encoding": "latin-1"})
    tcfg = {"source": "pay.csv", "read_options": {}, "columns": {"payment_id": "ref", "value_date": "date_valeur",
                                                                 "amount": "montant", "currency": "devise",
                                                                 "label": "inexistante"}}
    schema = {"amount_unit": None, "decimal_separator": ",", "date_format": "%d/%m/%Y"}
    fields = I.check_fields("payment", sample, tcfg, schema).set_index("champ")
    assert fields.loc["amount", "statut"] == I.ERROR                     # unité des montants à renseigner
    assert fields.loc["label", "statut"] == I.ERROR                      # colonne absente du fichier
    assert fields.loc["payment_id", "statut"] == I.ERROR                 # identifiant en double
    assert fields.loc["value_date", "statut"] == I.WARNING
    assert fields.loc["value_date", "invalides"] == 1 and "pas une date" in fields.loc["value_date", "exemples invalides"]
    assert fields.loc["channel", "statut"] == I.INFO                     # optionnel non mappé
    assert I.table_status(fields.reset_index()) == I.ERROR

    ok = I.check_fields("payment", sample, tcfg, {**schema, "amount_unit": "units"}).set_index("champ")
    assert ok.loc["amount", "invalides"] == 1 and ok.loc["amount", "statut"] == I.WARNING
    assert "1 000.00" in ok.loc["amount", "aperçu"] or "1 000,00" in ok.loc["amount", "aperçu"]


def test_status_values_need_translation(synthetic_dir):
    tcfg = copy.deepcopy(SCHEMA["tables"]["imputation"])
    sample = I.read_sample(synthetic_dir / tcfg["source"], tcfg.get("read_options"))
    assert I.check_fields("imputation", sample, tcfg, SCHEMA).set_index("champ").loc["status", "statut"] == I.OK
    tcfg["value_maps"] = {"status": {}}
    fields = I.check_fields("imputation", sample, tcfg, SCHEMA).set_index("champ")
    assert fields.loc["status", "statut"] == I.WARNING and "value_maps" in fields.loc["status", "détail"]


def test_list_sources_skips_hidden_and_other_files(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / ".cache").mkdir()
    for name in ("a.csv", "sub/b.parquet", ".cache/c.csv", "notes.txt.bak"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    assert I.list_sources(tmp_path) == ["a.csv", "sub/b.parquet"]


def test_validate_sources_on_synthetic(synthetic_dir):
    result = I.validate_sources(SCHEMA, synthetic_dir, log=lambda m: None)
    assert result["problems"] == []
    volumes = result["volumes"].set_index("table")
    assert volumes.loc["payment", "lignes"] > 0 and "client_file_line" in volumes.index
    assert set(result["fields"]["statut"]) <= {I.OK, I.INFO}

    broken = copy.deepcopy(SCHEMA)
    broken["tables"]["invoice"]["columns"]["due_date"] = "colonne_inconnue"
    broken["tables"]["agreement"]["source"] = None
    problems = I.validate_sources(broken, synthetic_dir, log=lambda m: None)["problems"]
    assert any("colonne_inconnue" in p for p in problems)
    assert any("agreement" in p for p in problems)
