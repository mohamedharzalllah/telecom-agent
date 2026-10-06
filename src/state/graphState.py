from typing import Annotated, List, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class GraphState(TypedDict, total=False):
    """LangGraph state for the agent orchestrator."""

    messages: Annotated[List[BaseMessage], add_messages]
    rag_context: str
    sql_context: str
    mode: str  # "rag" | "sql" | "hybrid"
    expected_answer: str  # optional: provided only when evaluation is needed
    
