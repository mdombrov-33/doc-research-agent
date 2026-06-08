from pathlib import Path

import aiosqlite
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.config import get_settings
from src.core.agent.nodes import agent_node, post_tools_node
from src.core.agent.state import AgentState
from src.core.agent.tools import TOOLS
from src.utils.logger import logger
from src.utils.node_timer import timed


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", timed("agent", agent_node))
    workflow.add_node("tools", ToolNode(TOOLS))
    workflow.add_node("post_tools", timed("post_tools", post_tools_node))

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "post_tools")
    workflow.add_edge("post_tools", "agent")

    # The serving path is async (astream_events), so the checkpointer must be async too.
    # AsyncSqliteSaver self-initializes its tables on first use; its connection is shared for
    # the app's lifetime. Tests inject a sync MemorySaver instead. Must be built inside a
    # running loop (the FastAPI lifespan) — AsyncSqliteSaver grabs the loop in __init__.
    if checkpointer is None:
        db_path = get_settings().checkpoints_db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        checkpointer = AsyncSqliteSaver(aiosqlite.connect(db_path))

    app = workflow.compile(checkpointer=checkpointer)
    logger.info("agent_graph_compiled")
    return app
