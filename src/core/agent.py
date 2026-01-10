from langgraph.graph import END, StateGraph

from src.core.nodes import (
    decide_to_generate,
    generate_node,
    grade_answer_quality_node,
    grade_documents_node,
    grade_generation_grounded_node,
    grade_generation_quality,
    retrieve_node,
    router_node,
    web_search_node,
)
from src.core.state import AgentState
from src.utils.logger import logger


def build_graph():
    logger.info("Building RAG agent graph")

    workflow = StateGraph(AgentState)

    workflow.add_node("router", router_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("websearch", web_search_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("check_hallucination", grade_generation_grounded_node)
    workflow.add_node("check_quality", grade_answer_quality_node)

    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router",
        lambda state: "websearch" if state.get("web_search") else "retrieve",
        {
            "websearch": "websearch",
            "retrieve": "retrieve",
        },
    )

    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_edge("websearch", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "websearch": "websearch",
            "generate": "generate",
        },
    )

    workflow.add_edge("generate", "check_hallucination")
    workflow.add_edge("check_hallucination", "check_quality")

    workflow.add_conditional_edges(
        "check_quality",
        grade_generation_quality,
        {
            "useful": END,
            "not useful": "generate",
        },
    )

    app = workflow.compile()

    logger.info("Graph compiled successfully")

    return app


def get_agent():
    return build_graph()
