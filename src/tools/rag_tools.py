from typing import List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from src.agents.SQL_agent import run_structured_query, schema_context
from src.config.setup import get_llm
from src.ingestitions.vectore_store import get_vector_store
from src.prompts.prompts import build_prompt
from src.state.graphState import GraphState

vector_store = get_vector_store()
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
prompt = build_prompt()
llm = get_llm()

# Static fallback facts extracted from FAQ_Catalogue_Telephones.pdf (04/11/2025).
CATALOG_FALLBACK = """
- iPhone 15 (2023) : 6.1\" OLED, puce A16, prix 256GB = 1099€, couleurs variées.
- iPhone 16 (2024) : caméra Fusion 48 Mpx, 22h vidéo, prix 256GB = 1149€.
- iPhone 17 (2025) : triple caméra 48 Mpx avec périscope 5x et LiDAR, 25h vidéo, prix 256GB = 1299€.
- iPhone 14 (2022) : Photonic Engine double 12 Mpx, 20h vidéo, prix 256GB = 999€.
- iPhone 13 (2021) : Cinematic Mode double 12 Mpx, 19h vidéo, prix 256GB = 929€.
- iPhone 12 (2020) : double 12 Mpx, prix 256GB = 829€.
- iPhone 11 (2019) : double 12 Mpx, 17h vidéo, prix 256GB = 659€.
- iPhone X (2017) : double 12 Mpx, 13h vidéo, prix 256GB = 749€.
Prix 256GB récap : X 749€, 11 659€, 12 829€, 13 929€, 14 999€, 15 1099€, 16 1149€, 17 1299€.
""".strip()

EU_COUNTRIES = {
    "allemagne",
    "autriche",
    "belgique",
    "bulgarie",
    "chypre",
    "croatie",
    "danemark",
    "espagne",
    "estonie",
    "finlande",
    "france",
    "grèce",
    "grece",
    "hongrie",
    "irlande",
    "italie",
    "lettonie",
    "lituanie",
    "luxembourg",
    "malte",
    "pays-bas",
    "pologne",
    "portugal",
    "roumanie",
    "slovaquie",
    "slovénie",
    "slovenie",
    "suède",
    "suede",
    "tchéquie",
    "republique tcheque",
    "tchequie",
    "grande-bretagne",
}


def _eu_roaming_note(question: str) -> Optional[str]:
    q = question.lower()
    if any(word in q for word in ["ue", "union européenne", "europe", "roaming", "etranger", "étranger"]):
        for country in EU_COUNTRIES:
            if country in q:
                return (
                    f"{country.title()} est dans l'UE : le roaming est inclus ('roaming comme à la maison'), "
                    "pas de surcoût dans la limite du forfait."
                )
    return None


def retrieve(state: GraphState) -> GraphState:
    question = _latest_question(state["messages"])
    # RAG context
    docs = retriever.invoke(question)
    rag_context = "\n\n".join(doc.page_content for doc in docs)
    # Structured data context (SQL over Excel tables)
    structured = run_structured_query(question)

    context_parts = [
        "=== Contexte FAQ ===",
        rag_context,
        "=== Contexte Données Structurées ===",
        structured,
    ]

    roam_note = _eu_roaming_note(question)
    if roam_note:
        context_parts.extend(["=== Note roaming UE ===", roam_note])

    context_parts.extend(
        [
            "=== Schéma tables ===",
            schema_context(),
            "=== Synthèse catalogue téléphones (secours) ===",
            CATALOG_FALLBACK,
        ]
    )

    context = "\n\n".join(context_parts)
    return {"context": context}


def generate(state: GraphState) -> GraphState:
    question = _latest_question(state["messages"])
    context = state.get("context", "")
    qa_chain = prompt | llm
    response = qa_chain.invoke({"question": question, "context": context})
    return {"messages": [AIMessage(content=response.content)]}


def _latest_question(messages: List[BaseMessage]) -> str:
    """Return the latest user question from the message list."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    raise ValueError("No user message found in state.")


