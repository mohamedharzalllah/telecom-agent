"""Structured helper tools used by the SQL agent."""

from __future__ import annotations

import re
from typing import List, Optional
from src.config.setup import get_llm

try:
    from langchain_core.pydantic_v1 import BaseModel, Field  # type: ignore
except ImportError:  # pragma: no cover - fallback for older installs
    try:
        from pydantic import BaseModel, Field  # type: ignore
    except ImportError as exc:  # pragma: no cover - defensive
        raise ImportError(
            "Pydantic not available. Install `langchain-core` or `pydantic<2`."
        ) from exc

from langchain_google_genai import ChatGoogleGenerativeAI

from src.config.setup import _ensure_api_key
from src.ingestitions.build_database import TABLE_DESCRIPTIONS, load_excel_frames


class UserInfo(BaseModel):
    user_name: str = Field(
        "",
        description="Prénom ou nom d'usage de l'utilisateur mentionné dans la question.",
    )
    family_name: str = Field(
        "",
        description="Nom de famille de l'utilisateur s'il est fourni.",
    )
    email: str = Field(
        "",
        description="Adresse email du client si elle est donnée, sinon chaîne vide.",
    )
    phone_number: str = Field(
        "",
        description="Numéro de téléphone (chiffres uniquement) s'il est présent.",
    )
    forfait: str = Field(
        "",
        description="Nom du forfait ou offre citée dans la question, sinon chaîne vide.",
    )


class UserIds(BaseModel):
    client_id: Optional[int] = Field(
        default=None, description="Identifiant client (colonne client_id dans clients.xlsx)."
    )
    forfait_id: Optional[int] = Field(
        default=None, description="Identifiant forfait (colonne forfait_id dans forfaits.xlsx)."
    )
    match_status: str = Field(
        default="indeterminable",
        description="Résumé court de la qualité de correspondance trouvée.",
    )


class TableSelection(BaseModel):
    tables: List[str] = Field(
        ..., description="Liste minimale des tables nécessaires pour répondre."
    )
    rationale: str = Field(
        ..., description="Raisonnement bref expliquant le choix des tables."
    )

def get_info(question: str, llm: Optional[ChatGoogleGenerativeAI] = None) -> UserInfo:
    """
    Extract structured user information directly from the question.

    Uses the LLM structured output capability to keep parsing logic simple.
    """
    structured_llm = (llm or get_llm()).with_structured_output(UserInfo)
    print(structured_llm.invoke(question))
    return structured_llm.invoke(question)


def _normalize_phone(raw: str) -> str:
    digits = re.findall(r"\d+", str(raw))
    return "".join(digits)


def _match_client(info: UserInfo) -> Optional[int]:
    frames = load_excel_frames()
    clients = frames.get("clients")
    if clients is None:
        return None

    subset = clients.copy()
    if info.family_name:
        subset = subset[subset["nom"].str.contains(info.family_name, case=False, na=False)]
    if info.user_name:
        subset = subset[subset["prenom"].str.contains(info.user_name, case=False, na=False)]
    if info.email:
        subset = subset[subset["email"].str.contains(info.email, case=False, na=False)]
    if info.phone_number:
        phone_norm = _normalize_phone(info.phone_number)
        subset["telephone_norm"] = subset["telephone"].astype(str).apply(_normalize_phone)
        subset = subset[subset["telephone_norm"].str.contains(phone_norm, na=False)]

    if len(subset) == 1:
        return int(subset.iloc[0]["client_id"])

    
    return None


def _match_forfait(info: UserInfo) -> Optional[int]:
    frames = load_excel_frames()
    forfaits = frames.get("forfaits")
    if forfaits is None or not info.forfait:
        return None
    subset = forfaits[forfaits["nom_forfait"].str.contains(info.forfait, case=False, na=False)]
    if len(subset) == 1:
        return int(subset.iloc[0]["forfait_id"])
    return None


def get_ids(info: UserInfo) -> UserIds:
    """
    Resolve the client_id and forfait_id from Excel tables using the extracted info.
    """
    client_id = _match_client(info)
    forfait_id = _match_forfait(info)

    if client_id and forfait_id:
        status = "client_id et forfait_id trouvés"
    elif client_id:
        status = "client_id trouvé, forfait_id manquant"
    elif forfait_id:
        status = "forfait_id trouvé, client_id manquant"
    else:
        status = "aucune correspondance unique trouvée"
    print(client_id, forfait_id, status)
    return UserIds(client_id=client_id, forfait_id=forfait_id, match_status=status)


def determine_tables(question: str) -> TableSelection:
    """
    Determine which tables are relevant for the question using simple heuristics.
    """
    question_lower = question.lower()
    selected: List[str] = []

    def add(table: str):
        if table not in selected:
            selected.append(table)

    if any(word in question_lower for word in ["facture", "paiement", "montant", "echeance"]):
        add("factures")
    if any(word in question_lower for word in ["consommation", "data", "minutes", "sms"]):
        add("consommation")
    if any(word in question_lower for word in ["forfait", "offre", "abonnement"]):
        add("forfaits")
        add("abonnements")
    if any(word in question_lower for word in ["ticket", "support", "incident", "priorite"]):
        add("tickets_support")
    if any(word in question_lower for word in ["client", "email", "telephone", "adresse", "nom", "prenom"]):
        add("clients")

    # Default to all if nothing detected so the agent still has context.
    if not selected:
        selected = list(TABLE_DESCRIPTIONS.keys())

    rationale = "; ".join(f"{t}: {TABLE_DESCRIPTIONS.get(t, '')}" for t in selected)
    print(selected)
    return TableSelection(tables=selected, rationale=rationale)

