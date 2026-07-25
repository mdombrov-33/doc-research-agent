from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall
from ir_measures import P, calc_aggregate
from ragas.metrics.collections import ExactMatch


def test_ragas_metric_runs_without_a_provider():
    result = ExactMatch().score(reference="supported answer", response="supported answer")

    assert result.value == 1.0


def test_deepeval_metric_runs_without_a_provider():
    retrieval = ToolCall(name="retrieve_documents", input_parameters={"query": "policy"})
    test_case = LLMTestCase(
        input="What is the policy?",
        actual_output="The policy is documented.",
        tools_called=[retrieval],
        expected_tools=[retrieval],
    )
    metric = ToolCorrectnessMetric(
        async_mode=False,
        include_reason=False,
        should_exact_match=True,
    )

    assert metric.measure(test_case, _show_indicator=False, _log_metric_to_confident=False) == 1.0


def test_ir_measures_metric_runs_without_a_provider():
    scores = calc_aggregate(
        [P @ 10],
        {"case-1": {"document-1": 2}},
        {"case-1": {"document-1": 1.0}},
    )

    assert scores[P @ 10] == 0.1
