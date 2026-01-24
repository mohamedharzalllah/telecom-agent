"""SQL agent utilities used by the orchestrator graph."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import List, Optional, Tuple

import pandas as pd
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config.setup import get_llm
from src.ingestitions.build_database import (
    build_sqlite_from_excel,
    describe_tables,
    get_sqlite_uri,
    load_excel_frames,
    table_schemas,
)
from src.tools.SQL_tools import TableSelection, UserIds, UserInfo, determine_tables, get_ids, get_info


# @lru_cache(maxsize=1)
# def _llm() -> ChatGoogleGenerativeAI:
#     """Shared LLM instance for SQL tooling."""
#     return ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=_ensure_api_key())


@lru_cache(maxsize=1)
def _db() -> SQLDatabase:
    """Ensure the SQLite DB is built and ready."""
    build_sqlite_from_excel()
    return SQLDatabase.from_uri(get_sqlite_uri(), sample_rows_in_table_info=2)


@lru_cache(maxsize=1)
def _sql_chain():
    """SQL generation chain cached for reuse."""
    try:
        from langchain.chains import create_sql_query_chain  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - defensive
        raise ImportError(
            "create_sql_query_chain not available. Ensure `langchain>=0.3.0` is installed."
        ) from exc
    return create_sql_query_chain(get_llm(), _db())


def schema_context() -> str:
    """Return a compact text description of tables and columns."""
    schemas = table_schemas(sample_rows=0)
    lines: List[str] = []
    for table, meta in schemas.items():
        cols = ", ".join(meta["columns"])
        lines.append(f"- {table}: {cols}")
    return "\n".join(lines)


def _table_description_block() -> str:
    desc = describe_tables()
    return "\n".join(f"{name}: {text}" for name, text in desc)


def _latest_consumption(client_id: Optional[int]) -> Optional[Tuple[str, float, int, int]]:
    """Return latest consumption tuple for the given client_id if available."""
    if not client_id:
        return None
    frames = load_excel_frames()
    df = frames.get("consommation")
    if df is None:
        return None
    subset = df.loc[df["client_id"] == client_id].copy(deep=True)
    if subset.empty:
        return None
    subset_sorted = subset.sort_values("mois", ascending=False)
    row = subset_sorted.iloc[0]
    return (
        str(row["mois"]),
        float(row["data_utilise_gb"]),
        int(row["minutes_utilisees"]),
        int(row["sms_utilises"]),
    )


def _forfait_info(forfait_id: Optional[int]) -> Optional[dict]:
    """Return a normalized forfait dict (data, minutes, price, engagement)."""
    if not forfait_id:
        return None
    frames = load_excel_frames()
    forfaits = frames.get("forfaits")
    if forfaits is None:
        return None
    match = forfaits[forfaits["forfait_id"] == forfait_id]
    if match.empty:
        return None
    row = match.iloc[0]

    def _safe_int(val):
        try:
            return int(val)
        except Exception:
            return str(val)

    return {
        "nom": row["nom_forfait"],
        "data": float(row["data_mensuel_gb"]),
        "minutes": _safe_int(row["minutes_incluses"]),
        "sms": _safe_int(row["sms_inclus"]),
        "prix": float(row["prix_mensuel"]),
        "engagement_mois": int(row["engagement_mois"]),
    }


def _current_abonnement(client_id: Optional[int]) -> Tuple[Optional[pd.Series], Optional[dict]]:
    """Return the most recent abonnement row and its forfait info."""
    if not client_id:
        return None, None
    frames = load_excel_frames()
    abonnements = frames.get("abonnements")
    if abonnements is None:
        return None, None
    subset = abonnements[abonnements["client_id"] == client_id]
    if subset.empty:
        return None, None
    subset = subset.sort_values("date_debut", ascending=False)
    row = subset.iloc[0]
    forfait = _forfait_info(int(row["forfait_id"])) if not pd.isna(row.get("forfait_id")) else None
    return row, forfait


def _tickets_for_client(client_id: Optional[int]) -> Optional[str]:
    """Return a human-readable summary of active or latest tickets for the client."""
    if not client_id:
        return None
    frames = load_excel_frames()
    df = frames.get("tickets_support")
    if df is None:
        return None

    subset = df[df["client_id"] == client_id]
    if subset.empty:
        return None

    active = subset[
        ~subset["statut"]
        .astype(str)
        .str.contains("résolu|resolu|clos|clôturé|cloturé|cloture", case=False, na=False)
    ]
    target = active if not active.empty else subset
    target = target.sort_values("date_creation", ascending=False).head(3)

    lines = []
    for _, row in target.iterrows():
        lines.append(
            f"ticket_id={row['ticket_id']}, sujet={row['sujet']}, "
            f"statut={row['statut']}, priorité={row['priorite']}, "
            f"créé le {row['date_creation']}, résolu le {row.get('date_resolution', 'N/A')}"
        )

    if active.empty:
        return "Aucun ticket actif. Derniers tickets: " + " | ".join(lines)
    return "Tickets actifs: " + " | ".join(lines)


def _next_unpaid_invoice(client_id: Optional[int]) -> Optional[str]:
    """Return the next unpaid invoice details for the client, if any."""
    if not client_id:
        return None
    frames = load_excel_frames()
    df = frames.get("factures")
    if df is None:
        return None
    subset = df[df["client_id"] == client_id]
    if subset.empty:
        return None
    # unpaid considered statut_paiement contains "attente" or not "payé"
    subset = subset.assign(statut_paiement_str=subset["statut_paiement"].astype(str))
    unpaid = subset[
        subset["statut_paiement_str"].str.contains("attente|impay", case=False, na=False)
    ]
    target = unpaid if not unpaid.empty else subset
    target = target.sort_values("date_echeance", ascending=True).head(1)
    row = target.iloc[0]
    return (
        f"Facture la plus proche: mois={row['mois']}, montant={row['montant']} €, "
        f"statut={row['statut_paiement']}, échéance={row['date_echeance']}, "
        f"date_paiement={row.get('date_paiement', 'N/A')}"
    )


def _abonnement_summary(client_id: Optional[int]) -> Optional[str]:
    """Return latest abonnement info and engagement status."""
    if not client_id:
        return None
    frames = load_excel_frames()
    abonnements = frames.get("abonnements")
    forfaits = frames.get("forfaits")
    if abonnements is None:
        return None
    subset = abonnements[abonnements["client_id"] == client_id]
    if subset.empty:
        return None
    subset = subset.sort_values("date_debut", ascending=False)
    row = subset.iloc[0]
    engagement = "engagé" if "engag" in str(row.get("statut", "")).lower() else "sans engagement"
    forfait_info = ""
    if forfaits is not None:
        fr = forfaits[forfaits["forfait_id"] == row["forfait_id"]]
        if not fr.empty:
            fr_row = fr.iloc[0]
            forfait_info = (
                f" | forfait='{fr_row['nom_forfait']}', prix={fr_row['prix_mensuel']} €, "
                f"engagement_mois={fr_row['engagement_mois']}"
            )
    return (
        f"Dernier abonnement: id={row['abonnement_id']}, statut={row['statut']} ({engagement}), "
        f"date_debut={row['date_debut']}, date_fin={row['date_fin']}{forfait_info}"
    )


def _upgrade_suggestion(question_lower: str, forfait: Optional[dict]) -> Optional[str]:
    """
    When the user wants more data, suggest the next forfait and price delta.
    """
    if not forfait:
        return None
    triggers = [
        "plus de data",
        "plus de données",
        "plus de gigas",
        "forfait avec plus de data",
        "forfait avec plus de données",
        "augmenter data",
        "monter en gamme",
        "upgrade",
    ]
    if not any(tok in question_lower for tok in triggers):
        return None

    frames = load_excel_frames()
    forfaits = frames.get("forfaits")
    if forfaits is None:
        return None
    higher = forfaits[forfaits["data_mensuel_gb"] > forfait["data"]]
    if higher.empty:
        return "Déjà sur le forfait avec le plus de data disponible."

    target = higher.sort_values(["data_mensuel_gb", "prix_mensuel"], ascending=[True, True]).iloc[0]
    diff = float(target["prix_mensuel"]) - forfait["prix"]
    return (
        f"Forfait actuel: {forfait['nom']} ({forfait['data']} Go, {forfait['prix']} € /mois). "
        f"Forfait supérieur conseillé: {target['nom_forfait']} ({target['data_mensuel_gb']} Go, {target['prix_mensuel']} € /mois). "
        f"Surcoût mensuel: {diff:.2f} €."
    )


def run_structured_query(question: str) -> str:
    """
    End-to-end pipeline:
    1) Extract user info from the question.
    2) Match IDs from Excel tables.
    3) Choose relevant tables.
    4) Generate and execute SQL on the SQLite mirror.
    """
    llm = get_llm()
    info: UserInfo = get_info(question, llm=llm)
    ids: UserIds = get_ids(info)
    selection: TableSelection = determine_tables(question)
    abonnement_row, forfait_info = _current_abonnement(ids.client_id)

    # Opportunistic direct lookups to ensure coverage without relying solely on SQL generation.
    question_lower = question.lower()
    direct_consumption = None
    if any(keyword in question_lower for keyword in ["consommation", "data", "internet"]) and ids.client_id:
        latest = _latest_consumption(ids.client_id)
        if latest:
            mois, data_gb, minutes, sms = latest
            if forfait_info:
                remaining = forfait_info["data"] - data_gb
                direct_consumption = (
                    f"Consommation la plus récente (mois={mois}) client_id={ids.client_id}: "
                    f"{data_gb} Go utilisés sur {forfait_info['data']} Go inclus "
                    f"(~{remaining:.2f} Go restants), minutes={minutes}, sms={sms}."
                )
            else:
                direct_consumption = (
                    f"Consommation la plus récente (mois={mois}) "
                    f"pour client_id={ids.client_id}: "
                    f"data={data_gb} Go, minutes={minutes}, sms={sms}."
                )
    direct_tickets = None
    if any(keyword in question_lower for keyword in ["ticket", "support", "incident"]) and ids.client_id:
        direct_tickets = _tickets_for_client(ids.client_id)
    direct_invoice = None
    if any(keyword in question_lower for keyword in ["facture", "paiement", "payer", "montant", "échéance", "echeance"]) and ids.client_id:
        direct_invoice = _next_unpaid_invoice(ids.client_id)
    direct_abonnement = None
    if any(keyword in question_lower for keyword in ["résiliation", "resiliation", "resilier", "résilier", "abonnement", "forfait", "engagement"]) and ids.client_id:
        direct_abonnement = _abonnement_summary(ids.client_id)
    direct_upgrade = _upgrade_suggestion(question_lower, forfait_info)

    sql_question = (
        f"{question}\n\n"
        f"Tables suggérées: {', '.join(selection.tables)}\n"
        f"Indices disponibles: client_id={ids.client_id}, forfait_id={ids.forfait_id}, forfait_nom={info.forfait}\n"
        f"Descriptions des tables:\n{_table_description_block()}"
    )

    try:
        if direct_consumption:
            sql = "(lookup consommation directe)"
        elif direct_tickets:
            sql = "(lookup tickets_support direct)"
        elif direct_invoice:
            sql = "(lookup factures direct)"
        elif direct_abonnement:
            sql = "(lookup abonnements direct)"
        elif direct_upgrade:
            sql = "(calcul upgrade direct)"
        else:
            sql = _sql_chain().invoke({"question": sql_question})
    except Exception as exc:  # pragma: no cover - defensive
        return f"Echec de génération SQL: {exc}"

    try:
        if direct_consumption:
            rows = direct_consumption
        elif direct_tickets:
            rows = direct_tickets
        elif direct_invoice:
            rows = direct_invoice
        elif direct_abonnement:
            rows = direct_abonnement
        elif direct_upgrade:
            rows = direct_upgrade
        else:
            rows = _db().run(sql)
    except Exception as exc:  # pragma: no cover - defensive
        rows = f"Erreur lors de l'exécution SQL: {exc}"

    direct_blocks = []
    if direct_consumption:
        direct_blocks.append(f"Consommation: {direct_consumption}")
    if forfait_info:
        direct_blocks.append(
            f"Forfait détecté: {forfait_info['nom']} - {forfait_info['data']} Go, "
            f"{forfait_info['minutes']} min, {forfait_info['sms']} SMS, {forfait_info['prix']} € /mois."
        )
    if direct_invoice:
        direct_blocks.append(f"Facture: {direct_invoice}")
    if direct_abonnement and not forfait_info:
        direct_blocks.append(f"Abonnement: {direct_abonnement}")
    if direct_tickets:
        direct_blocks.append(f"Tickets: {direct_tickets}")
    if direct_upgrade:
        direct_blocks.append(f"Upgrade data: {direct_upgrade}")

    context_parts = [
        "=== Réponses directes (prioritaires) ===",
        "\n".join(direct_blocks) if direct_blocks else "Aucune",
        "=== Infos utilisateur (structuré) ===",
        json.dumps(info.dict(), ensure_ascii=False),
        "=== Identifiants détectés ===",
        json.dumps(ids.dict(), ensure_ascii=False),
        "=== Tables ciblées ===",
        json.dumps(selection.tables, ensure_ascii=False),
        "=== SQL généré ===",
        sql,
        "=== Résultat SQL ===",
        rows if rows else "Aucun résultat retourné.",
    ]
    return "\n".join(context_parts)
