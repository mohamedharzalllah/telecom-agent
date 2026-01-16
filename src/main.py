"""Main module for the TelecomPlus multi-agent support system."""

from src.graph.graph import run_graph


def answer(question: str) -> str:
    """Answer customer questions using the unified LangGraph pipeline."""
    return run_graph(question)
