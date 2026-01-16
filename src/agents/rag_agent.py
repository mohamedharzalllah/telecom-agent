"""
RAG pipeline built with LangGraph for answering questions from PDF FAQs.

The pipeline:
1) Load and chunk PDFs from data/pdfs
2) Embed chunks with Google Generative AI embeddings
3) Store in an in-memory FAISS vector store
4) LangGraph runs retrieve -> generate steps
"""

from __future__ import annotations

from functools import lru_cache
from src.tools.rag_tools import retrieve, generate  # type: ignore[import-not-found]
from src.state.graphState import GraphState
from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage




@lru_cache(maxsize=1)
def build_rag_graph():
    """
    Build and compile the LangGraph RAG pipeline.

    Steps:
    - start -> retrieve -> generate -> END
    """
    graph = StateGraph(GraphState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)

    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    graph.set_entry_point("retrieve")

    return graph.compile(name="rag_agent",checkpointer=None, interrupt_before=[])

def answer_question(question: str) -> str:
    """
    Public entrypoint used by the Streamlit app.

    Args:
        question: user question in French.
    """
    # Build (or reuse cached) graph
    app = build_rag_graph()
    initial_state: GraphState = {"messages": [HumanMessage(content=question)], "context": ""}
    result = app.invoke(initial_state)
    # The latest AI message is the answer
    return result["messages"][-1].content