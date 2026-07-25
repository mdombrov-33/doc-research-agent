from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from evals.run_graph_eval import evaluate_graph_state, evaluate_live_graph


def _row() -> dict:
    return {
        "question": "How do plants convert sunlight into chemical energy?",
        "relevant_filenames": ["photosynthesis.txt"],
    }


def _state(outcome: str = "document_answer", answer: str = "") -> dict:
    artifact = {
        "content": "Photosynthesis converts light into chemical energy.",
        "document_id": "photosynthesis",
        "chunk_id": "photosynthesis:0",
        "filename": "photosynthesis.txt",
        "source": "document",
    }
    return {
        "messages": [
            HumanMessage(content=_row()["question"]),
            ToolMessage(
                content="document evidence",
                tool_call_id="retrieve-1",
                name="retrieve_documents",
                artifact=[artifact],
            ),
            AIMessage(content=answer or "It does. [document:photosynthesis:0]"),
        ],
        "outcome": outcome,
    }


def test_graph_eval_requires_document_answer_and_expected_citation():
    result = evaluate_graph_state(_row(), _state())

    assert result.passed is True
    assert result.cited_filenames == {"photosynthesis.txt"}


def test_graph_eval_rejects_web_path_even_with_document_citation():
    result = evaluate_graph_state(_row(), _state(outcome="web_answer"))

    assert result.passed is False


def test_graph_eval_accepts_expected_abstention_without_citations():
    row = {
        "question": "What is Project Zephyr's launch date?",
        "relevant_filenames": [],
        "expected_outcome": "abstained",
    }

    result = evaluate_graph_state(row, _state(outcome="abstained", answer="No evidence."))

    assert result.passed is True


def test_live_graph_eval_continues_after_one_graph_error():
    class Graph:
        def __init__(self):
            self.calls = 0

        def invoke(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("provider failure")
            return _state()

    results = evaluate_live_graph(Graph(), [_row(), _row()])

    assert [result.passed for result in results] == [False, True]
    assert results[0].error == "RuntimeError"


def test_live_graph_eval_reuses_one_thread_for_follow_up_turns():
    class Graph:
        def __init__(self):
            self.configs = []

        def invoke(self, _inputs, config):
            self.configs.append(config)
            return _state()

    row = {
        "turns": ["Who discovered the moons?", "When did he do it?"],
        "question": "When did he do it?",
        "relevant_filenames": ["photosynthesis.txt"],
    }
    graph = Graph()

    result = evaluate_live_graph(graph, [row])[0]

    assert result.passed is True
    assert len(graph.configs) == 2
    assert (
        graph.configs[0]["configurable"]["thread_id"]
        == graph.configs[1]["configurable"]["thread_id"]
    )
