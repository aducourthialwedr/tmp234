import pandas as pd
import streamlit as st
import yaml
from pydantic import ValidationError

from src.settings import RulesConfig, Settings, load_rules
from src.ui.common import (
    current_settings, render_model, rules_path, save_section, schema_path, settings_path, step_header,
    validation_errors,
)
from src.yaml_io import scalar

step_header("config")
st.caption("Le mapping des sources réelles (schema.yaml) se fait dans la page « Sources et ingestion ».")
tab_general, tab_files, tab_env = st.tabs(["Paramètres généraux", "Fichiers de configuration", "Environnement"])

# --- Paramètres généraux -----------------------------------------------------------------------
with tab_general:
    settings = current_settings()
    st.subheader("Chemins")
    paths = render_model(settings.paths, "paths")
    save_section(settings, "paths", paths, "paths")
    st.divider()
    st.subheader("Chargement")
    load = render_model(settings.load, "load")
    save_section(settings, "load", load, "load")


# --- Fichiers de configuration ------------------------------------------------------------------
def _flatten(value, prefix: str = "") -> dict:
    if isinstance(value, dict):
        return {k: v for key, item in value.items() for k, v in _flatten(item, f"{prefix}{key}.").items()}
    return {prefix.rstrip("."): value}


with tab_files:
    st.caption("Toute la configuration de la pipeline tient dans trois fichiers, édités par les pages de l'interface, "
               "le notebook ou à la main. Les pages Découpage, Allocation, Règles, ML et Évaluation éditent chacune "
               "leur section.")
    current = current_settings()
    rules = load_rules(rules_path())
    diff = [{"paramètre": k, "valeur": scalar(v), "défaut": scalar(d)}
            for (k, v), d in zip(_flatten(current.model_dump()).items(), _flatten(Settings().model_dump()).values())
            if v != d]
    defaults = {r.id: r for r in RulesConfig().rules}
    for r in rules.rules:
        base_rule = defaults.get(r.id)
        for k, v in _flatten({"enabled": r.enabled, "priority": r.priority, **r.params}).items():
            d = None if base_rule is None else _flatten(
                {"enabled": base_rule.enabled, "priority": base_rule.priority, **base_rule.params}).get(k)
            if v != d:
                diff.append({"paramètre": f"rules.{r.id}.{k}", "valeur": scalar(v), "défaut": scalar(d)})
    if rules.min_precision != RulesConfig().min_precision:
        diff.append({"paramètre": "rules.min_precision", "valeur": scalar(rules.min_precision),
                     "défaut": scalar(RulesConfig().min_precision)})
    st.markdown(f"**Réglages modifiés par rapport aux valeurs par défaut** ({len(diff)}) · règles v{rules.version}")
    if diff:
        st.dataframe(pd.DataFrame(diff), hide_index=True, width="stretch")
    else:
        st.caption("Aucun : tous les réglages sont à leur valeur par défaut.")

    for label, path, model in [("settings.yaml", settings_path(), Settings), ("rules.yaml", rules_path(), RulesConfig),
                               ("schema.yaml", schema_path(), None)]:
        with st.expander(f"{label} — `{path}`", icon=":material/description:"):
            text = path.read_text(encoding="utf-8") if path.exists() else ""
            st.code(text or "# fichier absent : valeurs par défaut", language="yaml", height=360)
            c = st.columns([1, 2], vertical_alignment="bottom")
            c[0].download_button("Télécharger", text, file_name=label, mime="text/yaml", icon=":material/download:",
                                 key=f"files.{label}.download")
            upload = c[1].file_uploader(f"Remplacer par un autre {label}", type=["yaml", "yml"],
                                        key=f"files.{label}.upload")
            if upload is not None and st.button(f"Valider et remplacer {label}", key=f"files.{label}.replace",
                                                icon=":material/upload:"):
                content = upload.getvalue().decode("utf-8")
                try:
                    data = yaml.safe_load(content) or {}
                    if model is not None:
                        model.model_validate(data)
                except ValidationError as exc:
                    st.error("Fichier refusé :\n\n" + "\n".join(f"- {m}" for m in validation_errors(exc)))
                except yaml.YAMLError as exc:
                    st.error(f"YAML invalide : {exc}")
                else:
                    path.write_text(content, encoding="utf-8")
                    st.toast(f"{label} remplacé", icon=":material/check:")
                    st.rerun()

# --- Environnement ----------------------------------------------------------------------------------
with tab_env:
    import platform
    import sys
    from importlib.metadata import PackageNotFoundError, version

    from src import memory
    from src.profiling import find_pyspy

    pod = memory.pod_memory()
    c = st.columns(4)
    c[0].metric("CPU utilisables", memory.available_cpus(), help="Quota du conteneur s'il existe, sinon la machine.")
    c[1].metric("Limite mémoire du pod", f"{pod['limit'] / 2 ** 30:.1f} Go" if pod["limit"] else "—")
    c[2].metric("Mémoire du processus", f"{memory.process_rss() / 2 ** 30:.2f} Go")
    pyspy = find_pyspy()
    c[3].metric("py-spy", "installé" if pyspy else "absent", help=pyspy or "Profil par fonction : repli sur le fil "
                "interne (rien à installer).")
    versions = []
    for package in ("pandas", "numpy", "pyarrow", "scipy", "scikit-learn", "lightgbm", "pydantic", "streamlit",
                    "altair", "pyyaml"):
        try:
            versions.append({"bibliothèque": package, "version": version(package)})
        except PackageNotFoundError:
            versions.append({"bibliothèque": package, "version": "absente"})
    st.dataframe(pd.DataFrame([{"bibliothèque": "python", "version": platform.python_version()}, *versions]),
                 hide_index=True, width="content")
    pandas_major = int(pd.__version__.split(".")[0])
    if pandas_major < 3:
        st.warning(f"pandas {pd.__version__} : la normalisation attend pandas ≥ 3 (chaînes Arrow, expression "
                   "régulière \\p{M}).", icon=":material/warning:")
    st.caption(f"Interpréteur `{sys.executable}` · {platform.platform()}")
