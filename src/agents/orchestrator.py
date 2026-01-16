"""
Agentic orchestration graph (LangGraph) to route between:
- RAG over FAQ PDFs
- Structured SQL over Excel tables
- Hybrid (both)

Routing logic:
1) Route node asks the LLM to pick rag / sql / hybrid based on the question.
2) Retrieve nodes gather the needed context (FAQ chunks and/or SQL results).
3) Generate node produces the final grounded answer.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from src.graph.graph import build_graph  # type: ignore[import-not-found]


def build_orchestrator_graph():
    """
    Legacy alias kept for compatibility.

    Returns the compiled LangGraph that routes to RAG, SQL, or both.
    """
    return build_graph()


def answer_orchestrated(question: str) -> str:
    """
    Public entrypoint used by the Streamlit app for agentic routing.

    Args:
        question: user question in French.
    """
    app = build_orchestrator_graph()
    initial_state = {
        "messages": [HumanMessage(content=question)],
        "rag_context": "",
        "sql_context": "",
        "mode": "rag",
    }
    result = app.invoke(initial_state)
    return result["messages"][-1].content

