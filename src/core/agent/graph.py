from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from src.core.agent.nodes import (
    abstain_node,
    agent_node,
    answer_node,
    evidence_assessment_node,
    web_fallback_node,
)
from src.core.agent.state import AgentState
from src.core.agent.tools import retrieve_documents
from src.utils.logger import logger
from src.utils.node_timer import timed


def build_graph(checkpointer: BaseCheckpointSaver):
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", timed("agent", agent_node))
    workflow.add_node(
        "tools",
        ToolNode(
            [retrieve_documents],
            handle_tool_errors="Document retrieval is temporarily unavailable.",
        ),
    )
    workflow.add_node("assess_evidence", timed("assess_evidence", evidence_assessment_node))
    workflow.add_node("web_fallback", timed("web_fallback", web_fallback_node))
    workflow.add_node("answer", timed("answer", answer_node))
    workflow.add_node("abstain", timed("abstain", abstain_node))

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent", _retrieval_requested, {"retrieve": "tools", "end": "abstain"}
    )
    workflow.add_edge("tools", "assess_evidence")
    workflow.add_conditional_edges(
        "assess_evidence",
        _after_assessment,
        {"answer": "answer", "web": "web_fallback", "abstain": "abstain"},
    )
    workflow.add_edge("web_fallback", "assess_evidence")
    workflow.add_edge("answer", END)
    workflow.add_edge("abstain", END)

    app = workflow.compile(checkpointer=checkpointer)
    logger.info("agent_graph_compiled")
    return app


def _retrieval_requested(state: AgentState) -> str:
    message = state["messages"][-1]
    if any(call["name"] == "retrieve_documents" for call in getattr(message, "tool_calls", [])):
        return "retrieve"
    return "end"


def _after_assessment(state: AgentState) -> str:
    if state.get("evidence_sufficient"):
        return "answer"
    if any(getattr(message, "name", None) == "web_search" for message in state["messages"]):
        return "abstain"
    return "web"
