import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from src.core.agent import nodes, tools
from src.core.agent.graph import build_graph
from src.core.agent.nodes import EvidenceAssessment

CORPUS_DIR = Path(__file__).parents[2] / "evals" / "corpus"


@pytest.fixture
def graph():
    return build_graph(checkpointer=MemorySaver())


def _patch_llms(monkeypatch, agent_script, assessments):
    """Script query/synthesis and the independently structured evidence assessor."""
    agent = MagicMock()
    agent.bind_tools.return_value = agent
    agent.invoke.side_effect = agent_script
    classifier = MagicMock()
    structured = MagicMock()
    structured.invoke.side_effect = assessments
    classifier.with_structured_output.return_value = structured

    def get_llm(model=None, **_):
        return classifier if model == "classifier" else agent

    monkeypatch.setattr(nodes, "get_llm", get_llm)
    monkeypatch.setattr(
        nodes, "get_settings", lambda: SimpleNamespace(CLASSIFIER_MODEL="classifier")
    )
    return agent


def _run(graph, question, model=None, top_k=None, thread_id=None):
    configurable = {"thread_id": thread_id or str(uuid.uuid4())}
    if model is not None:
        configurable["model"] = model
    if top_k is not None:
        configurable["top_k"] = top_k
    config = {"configurable": configurable}
    inputs = {"messages": [HumanMessage(content=question)]}
    return graph.invoke(inputs, config=config)


def _retrieve_call(cid="c1"):
    return AIMessage(
        content="", tool_calls=[{"name": "retrieve_documents", "args": {"query": "q"}, "id": cid}]
    )


def _corpus_artifact(filename: str, chunk_id: str) -> dict:
    return {
        "content": (CORPUS_DIR / filename).read_text(),
        "document_id": filename.removesuffix(".txt"),
        "chunk_id": chunk_id,
        "filename": filename,
        "source": "document",
    }


def test_graph_retrieves_then_answers(graph, monkeypatch):
    agent = _patch_llms(
        monkeypatch,
        [_retrieve_call(), AIMessage(content="final answer [document:photosynthesis:0]")],
        [EvidenceAssessment(sufficient=True, supporting_source_ids=["document:photosynthesis:0"])],
    )
    seen = {}
    monkeypatch.setattr(
        tools,
        "hybrid_search",
        lambda q, k: seen.update(top_k=k)
        or [
            _corpus_artifact("photosynthesis.txt", "photosynthesis:0"),
            {
                "content": "UNSELECTED_ARTIFACT_MARKER",
                "document_id": "b",
                "filename": "b.pdf",
                "source": "document",
            },
        ],
    )

    state = _run(graph, "How do plants convert sunlight into chemical energy?", top_k=11)

    assert seen["top_k"] == 11
    assert state["messages"][-1].content == "final answer [document:photosynthesis:0]"
    assert state["outcome"] == "document_answer"
    assert state["stop_reason"] == "document_evidence_sufficient"
    answer_input = agent.invoke.call_args_list[-1].args[0][-1].content
    assert "photosynthesis" in answer_input.lower()
    assert "UNSELECTED_ARTIFACT_MARKER" not in answer_input


def test_graph_uses_the_agent_standalone_follow_up_query(graph, monkeypatch):
    _patch_llms(
        monkeypatch,
        [
            _retrieve_call("initial-retrieval"),
            AIMessage(content="Rust is covered [document:rust:0]."),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "retrieve_documents",
                        "args": {
                            "query": "Zero to Production in Rust database chapters"
                        },
                        "id": "follow-up-retrieval",
                    }
                ],
            ),
            AIMessage(content="It covers sqlx [document:rust:0]."),
        ],
        [
            EvidenceAssessment(sufficient=True, supporting_source_ids=["document:rust:0"]),
            EvidenceAssessment(sufficient=True, supporting_source_ids=["document:rust:0"]),
        ],
    )
    queries = []
    monkeypatch.setattr(
        tools,
        "hybrid_search",
        lambda query, top_k: queries.append(query)
        or [
            {
                "content": "context",
                "document_id": "rust",
                "filename": "rust.pdf",
                "source": "document",
            }
        ],
    )
    thread_id = str(uuid.uuid4())

    _run(graph, "What do we have about Rust?", thread_id=thread_id)
    state = _run(graph, "What about its database chapters?", thread_id=thread_id)

    assert queries == ["q", "Zero to Production in Rust database chapters"]
    human_questions = [
        message.content for message in state["messages"] if isinstance(message, HumanMessage)
    ]
    assert human_questions == [
        "What do we have about Rust?",
        "What about its database chapters?",
    ]
    assert state["messages"][-1].content == "It covers sqlx [document:rust:0]."


def test_graph_falls_back_to_web_search(graph, monkeypatch):
    _patch_llms(
        monkeypatch,
        [_retrieve_call(), AIMessage(content="final answer [web:https://example.com]")],
        [
            EvidenceAssessment(sufficient=False),
            EvidenceAssessment(
                sufficient=True, supporting_source_ids=["web:https://example.com"]
            ),
        ],
    )
    monkeypatch.setattr(
        tools,
        "hybrid_search",
        lambda q, k: [
            _corpus_artifact("photosynthesis.txt", "photosynthesis:0")
        ],
    )
    web_search = MagicMock(
        return_value=(
            "[Source ID: web:https://example.com]\n[Web: Web result]\nweb result",
            [
                {
                    "content": "web result",
                    "title": "Web result",
                    "url": "https://example.com",
                    "source": "web",
                }
            ],
        )
    )
    monkeypatch.setattr(nodes, "search_web", web_search)

    state = _run(graph, "q")

    web_search.assert_called_once_with("q")
    assert state["messages"][-1].content == "final answer [web:https://example.com]"
    assert state["outcome"] == "web_answer"
    assert state["stop_reason"] == "web_evidence_sufficient"


def test_graph_abstains_when_no_evidence_passes_assessment(graph, monkeypatch):
    _patch_llms(
        monkeypatch,
        [_retrieve_call()],
        [EvidenceAssessment(sufficient=True, supporting_source_ids=["document:not-retrieved:0"])],
    )
    monkeypatch.setattr(
        tools,
        "hybrid_search",
        lambda q, k: [
            _corpus_artifact("photosynthesis.txt", "photosynthesis:0")
        ],
    )
    monkeypatch.setattr(nodes, "search_web", lambda _: ("No documents found.", []))

    state = _run(graph, "q")

    assert state["messages"][-1].content == nodes.NO_EVIDENCE_RESPONSE
    assert state["outcome"] == "abstained"
    assert state["stop_reason"] == "insufficient_evidence_after_web"


def test_graph_abstains_when_query_model_does_not_request_retrieval(graph, monkeypatch):
    _patch_llms(monkeypatch, [AIMessage(content="I can answer without searching.")], [])

    state = _run(graph, "q")

    assert state["messages"][-1].content == nodes.NO_EVIDENCE_RESPONSE
    assert state["outcome"] == "abstained"
    assert state["stop_reason"] == "retrieval_not_requested"
