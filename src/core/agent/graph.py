from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.core.agent.nodes import agent_node, post_tools_node
from src.core.agent.state import AgentState
from src.core.agent.tools import TOOLS
from src.utils.logger import logger
from src.utils.node_timer import timed


def build_graph(checkpointer: BaseCheckpointSaver):
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", timed("agent", agent_node))
    workflow.add_node("tools", ToolNode(TOOLS))
    workflow.add_node("post_tools", timed("post_tools", post_tools_node))

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "post_tools")
    workflow.add_edge("post_tools", "agent")

    app = workflow.compile(checkpointer=checkpointer)
    logger.info("agent_graph_compiled")
    return app
