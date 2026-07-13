from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from opentelemetry import trace
from pydantic import BaseModel, Field

from src.config import get_settings
from src.core.agent.prompts import (
    AGENT_SYSTEM_PROMPT,
    ANSWER_SYSTEM_PROMPT,
    EVIDENCE_ASSESSMENT_SYSTEM_PROMPT,
)
from src.core.agent.state import AgentState
from src.core.agent.tools import format_docs, retrieve_documents, search_web
from src.core.citations import citation_from_artifact, citations_from_artifacts
from src.core.llm import get_llm
from src.utils.logger import logger

_tracer = trace.get_tracer(__name__)

NO_EVIDENCE_RESPONSE = "I couldn't find enough reliable evidence to answer that."


class EvidenceAssessment(BaseModel):
    """Validated verdict on whether the current turn's evidence can answer its question."""

    sufficient: bool
    supporting_source_ids: list[str] = Field(default_factory=list)


def _configurable(config: RunnableConfig | None) -> dict:
    return (config or {}).get("configurable") or {}


def agent_node(state: AgentState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """Form a history-aware standalone query and call document retrieval."""
    model = _configurable(config).get("model")
    llm = get_llm(model).bind_tools([retrieve_documents])
    history = _conversation_messages(state["messages"], get_settings().CONVERSATION_HISTORY_TURNS)
    messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT), *history]
    with _tracer.start_as_current_span("agent.llm_call") as span:
        span.set_attribute("agent.model", model or "default")
        response = llm.invoke(messages)
        span.set_attribute("agent.has_tool_calls", bool(getattr(response, "tool_calls", None)))
    return {"messages": [response]}


def evidence_assessment_node(
    state: AgentState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    """Assess the current turn's evidence; malformed or unsupported verdicts fail closed."""
    question = _current_question(state)
    artifacts = _turn_artifacts(state)
    source_ids = {citation.source_id for citation in citations_from_artifacts(artifacts)}
    if not question or not source_ids:
        return {"evidence_sufficient": False, "supporting_source_ids": []}

    evidence = "\n\n".join(_turn_tool_content(state))
    try:
        structured = get_llm(get_settings().CLASSIFIER_MODEL).with_structured_output(
            EvidenceAssessment
        )
        assessment = EvidenceAssessment.model_validate(
            structured.invoke(
                [
                    SystemMessage(content=EVIDENCE_ASSESSMENT_SYSTEM_PROMPT),
                    HumanMessage(content=f"Question:\n{question}\n\nEvidence:\n{evidence}"),
                ]
            )
        )
    except Exception as error:
        logger.warning("evidence_assessment_failed", failure_type=type(error).__name__)
        return {"evidence_sufficient": False, "supporting_source_ids": []}

    supported = set(assessment.supporting_source_ids)
    sufficient = assessment.sufficient and bool(supported) and supported <= source_ids
    if assessment.sufficient and not sufficient:
        logger.warning("evidence_assessment_invalid_source_ids")
    return {
        "evidence_sufficient": sufficient,
        "supporting_source_ids": assessment.supporting_source_ids if sufficient else [],
    }


def answer_node(state: AgentState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """Synthesize a cited answer from evidence already judged sufficient."""
    model = _configurable(config).get("model")
    evidence = format_docs(_supporting_artifacts(state))
    response = get_llm(model).invoke(
        [
            SystemMessage(content=ANSWER_SYSTEM_PROMPT),
            HumanMessage(content=f"Question:\n{_current_question(state)}\n\nEvidence:\n{evidence}"),
        ]
    )
    outcome = "web_answer" if _used_web_fallback(state) else "document_answer"
    stop_reason = (
        "web_evidence_sufficient" if outcome == "web_answer" else "document_evidence_sufficient"
    )
    return {"messages": [response], "outcome": outcome, "stop_reason": stop_reason}


def web_fallback_node(
    state: AgentState, config: RunnableConfig | None = None
) -> dict[str, list[ToolMessage]]:
    """Search the web once using the standalone document-retrieval query."""
    query = _retrieval_query(state) or _current_question(state)
    content, artifacts = search_web(query)
    return {
        "messages": [
            ToolMessage(
                content=content,
                tool_call_id="web-fallback",
                name="web_search",
                artifact=artifacts,
            )
        ]
    }


def abstain_node(state: AgentState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """Return the safe final response when no evidence passed assessment."""
    stop_reason = (
        "insufficient_evidence_after_web"
        if _used_web_fallback(state)
        else "retrieval_not_requested"
    )
    return {
        "messages": [AIMessage(content=NO_EVIDENCE_RESPONSE)],
        "outcome": "abstained",
        "stop_reason": stop_reason,
    }


def _current_question(state: AgentState) -> str:
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage) and isinstance(message.content, str):
            return message.content
    return ""


def _conversation_messages(messages: list[Any], max_turns: int) -> list[Any]:
    """Keep recent user/answer turns while excluding workflow scratchpad messages."""
    human_indexes = [
        index for index, message in enumerate(messages) if isinstance(message, HumanMessage)
    ]
    if not human_indexes:
        return []

    retained_indexes = human_indexes[-max_turns:]
    conversation: list[Any] = []
    for start, end in zip(retained_indexes, [*retained_indexes[1:], len(messages)]):
        conversation.append(messages[start])
        answer = next(
            (
                message
                for message in reversed(messages[start + 1 : end])
                if isinstance(message, AIMessage) and not message.tool_calls
            ),
            None,
        )
        if answer is not None:
            conversation.append(answer)
    return conversation


def _turn_messages(state: AgentState) -> list[Any]:
    messages = state["messages"]
    last_human_idx = max(
        (index for index, message in enumerate(messages) if isinstance(message, HumanMessage)),
        default=-1,
    )
    return messages[last_human_idx + 1 :]


def _turn_artifacts(state: AgentState) -> list[dict]:
    return [
        artifact
        for message in _turn_messages(state)
        if isinstance(message, ToolMessage) and isinstance(message.artifact, list)
        for artifact in message.artifact
        if isinstance(artifact, dict)
    ]


def _turn_tool_content(state: AgentState) -> list[str]:
    return [
        message.content
        for message in _turn_messages(state)
        if isinstance(message, ToolMessage) and isinstance(message.content, str)
    ]


def _supporting_artifacts(state: AgentState) -> list[dict]:
    source_ids = set(state.get("supporting_source_ids", []))
    return [
        artifact
        for artifact in _turn_artifacts(state)
        if (citation := citation_from_artifact(artifact)) is not None
        and citation.source_id in source_ids
    ]


def _used_web_fallback(state: AgentState) -> bool:
    return any(
        isinstance(message, ToolMessage) and message.name == "web_search"
        for message in _turn_messages(state)
    )


def _retrieval_query(state: AgentState) -> str:
    for message in reversed(_turn_messages(state)):
        if not isinstance(message, AIMessage):
            continue
        for tool_call in message.tool_calls:
            if tool_call["name"] == "retrieve_documents":
                query = tool_call["args"].get("query")
                if isinstance(query, str):
                    return query
    return ""
