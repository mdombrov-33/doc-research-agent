from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.core.nodes import (
    _timed,
    generate_node,
    grade_documents_node,
    retrieve_node,
    router_node,
    web_search_node,
)
from src.core.state import AgentState
from src.utils.logger import logger


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("router", _timed("router", router_node))
    workflow.add_node("retrieve", _timed("retrieve", retrieve_node))
    workflow.add_node("websearch", _timed("websearch", web_search_node))
    workflow.add_node("grade_documents", _timed("grade_documents", grade_documents_node))
    workflow.add_node("generate", _timed("generate", generate_node))

    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router",
        lambda state: ["retrieve", "websearch"] if state.get("web_search") else ["retrieve"],
    )

    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_edge("websearch", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        lambda state: "websearch"
        if state.get("web_fallback_needed") and not state.get("web_search_done")
        else "generate",
    )
    workflow.add_edge("generate", END)

    app = workflow.compile(checkpointer=MemorySaver())
    logger.info("agent_graph_compiled")
    return app
