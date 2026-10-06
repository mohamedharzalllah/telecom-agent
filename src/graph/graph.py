"""LangGraph pipeline that routes between RAG, SQL, or hybrid answers per question."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from src.config.setup import _ensure_api_key
from src.ingestitions.vectore_store import get_vector_store
from src.prompts.prompts import build_prompt, router_prompt
from src.state.graphState import GraphState
from src.tools.rag_tools import CATALOG_FALLBACK, _eu_roaming_note, _latest_question
from src.agents.SQL_agent import run_structured_query, schema_context


@lru_cache(maxsize=1)
def _build_retriever():
    return get_vector_store().as_retriever(search_kwargs={"k": 5})


def _parse_route(decision: str) -> Literal["rag", "sql", "hybrid"]:
    decision = decision.lower()
    if any(key in decision for key in ["hybrid", "both", "mix"]):
        return "hybrid"
    if "sql" in decision:
        return "sql"
    if "rag" in decision:
        return "rag"
    # Fallback to hybrid to be conservative when unsure.
    return "hybrid"


@lru_cache(maxsize=1)
def build_graph():
    """
    Build and compile the LangGraph that routes per-question:
    - rag only
    - sql only
    - hybrid (rag then sql)
    """
    load_dotenv()
    api_key = _ensure_api_key()
    retriever = _build_retriever()
    prompt = build_prompt()
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_key)

    def route(state: GraphState) -> GraphState:
        question = _latest_question(state["messages"])
        router_prompt_text = router_prompt(question)
        decision_raw = llm.invoke(router_prompt_text).content.strip()
        mode = _parse_route(decision_raw)
        return {"mode": mode}

    def retrieve_rag(state: GraphState) -> GraphState:
        question = _latest_question(state["messages"])
        docs = retriever.invoke(question)
        rag_context = "\n\n".join(doc.page_content for doc in docs)
        return {"rag_context": rag_context}

    def retrieve_sql(state: GraphState) -> GraphState:
        question = _latest_question(state["messages"])
        structured = run_structured_query(question)
        return {"sql_context": structured}

    def generate(state: GraphState) -> GraphState:
        question = _latest_question(state["messages"])
        ctx_parts = []
        if state.get("rag_context"):
            ctx_parts.append("=== Contexte FAQ ===")
            ctx_parts.append(state["rag_context"])
        if state.get("sql_context"):
            ctx_parts.append("=== Contexte Données Structurées ===")
            ctx_parts.append(state["sql_context"])
        roam_note = _eu_roaming_note(question)
        if roam_note:
            ctx_parts.append("=== Note roaming UE ===")
            ctx_parts.append(roam_note)
        ctx_parts.append("=== Schéma tables ===")
        ctx_parts.append(schema_context())
        ctx_parts.append("=== Catalogue téléphones (secours) ===")
        ctx_parts.append(CATALOG_FALLBACK)
        context = "\n\n".join(ctx_parts)
        qa_chain = prompt | llm
        response = qa_chain.invoke({"question": question, "context": context})
        return {"messages": [AIMessage(content=response.content)]}

    graph = StateGraph(GraphState)
    graph.add_node("route", route)
    graph.add_node("retrieve_rag", retrieve_rag)
    graph.add_node("retrieve_sql", retrieve_sql)
    graph.add_node("generate", generate)

    def after_route(state: GraphState) -> str:
        if state["mode"] == "sql":
            return "retrieve_sql"
        if state["mode"] == "hybrid":
            return "retrieve_rag"
        return "retrieve_rag"

    graph.add_conditional_edges(
        "route",
        after_route,
        {
            "retrieve_rag": "retrieve_rag",
            "retrieve_sql": "retrieve_sql",
        },
    )

    def after_rag(state: GraphState) -> str:
        return "retrieve_sql" if state["mode"] == "hybrid" else "generate"

    graph.add_conditional_edges(
        "retrieve_rag",
        after_rag,
        {"retrieve_sql": "retrieve_sql", "generate": "generate"},
    )

    graph.add_edge("retrieve_sql", "generate")
    graph.add_edge("generate", END)
    graph.set_entry_point("route")

    return graph.compile(checkpointer=None, interrupt_before=[])


def run_graph(question: str) -> str:
    """Convenience wrapper to execute the compiled graph for a single question."""
    app = build_graph()
    initial_state: GraphState = {
        "messages": [HumanMessage(content=question)],
        "rag_context": "",
        "sql_context": "",
        "mode": "rag",
    }
    result = app.invoke(initial_state)
    return result["messages"][-1].content

