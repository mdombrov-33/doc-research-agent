import json
from collections.abc import AsyncGenerator

from fastapi.responses import StreamingResponse

from src.api.schemas import QueryRequest
from src.core.agent import get_agent
from src.guardrails.guardrails_wrapper import get_guardrails
from src.utils.logger import logger


async def _token_generator(request: QueryRequest) -> AsyncGenerator[str, None]:
    guardrails = get_guardrails()

    refusal = await guardrails.check_input(request.question)
    if refusal:
        yield f"data: {json.dumps({'token': refusal, 'done': True})}\n\n"
        return

    agent = get_agent()
    inputs = {
        "question": request.question,
        "generation": "",
        "web_search": False,
        "explicit_web_search": False,
        "documents": [],
        "retrieval_attempts": 0,
        "generation_attempts": 0,
        "model": request.model,
        "top_k": request.top_k,
    }
    config = {"configurable": {"thread_id": request.session_id}}

    accumulated: list[str] = []
    sources_count = 0

    try:
        async for event in agent.astream_events(inputs, config=config, version="v2"):
            kind = event["event"]

            if (
                kind == "on_chat_model_stream"
                and event.get("metadata", {}).get("langgraph_node") == "generate"
            ):
                token = event["data"]["chunk"].content
                if token:
                    accumulated.append(token)
                    yield f"data: {json.dumps({'token': token})}\n\n"

            elif kind == "on_chain_end" and event.get("name") == "LangGraph":
                output = event.get("data", {}).get("output", {})
                sources_count = len(output.get("documents", []))

    except Exception as e:
        logger.error(f"Stream error: {e}")
        yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"
        return

    full_response = "".join(accumulated)
    correction = await guardrails.check_output(full_response)
    if correction:
        yield f"data: {json.dumps({'token': correction, 'done': True, 'correction': True})}\n\n"
    else:
        yield f"data: {json.dumps({'done': True, 'sources_count': sources_count, 'session_id': request.session_id})}\n\n"


async def handle_stream(request: QueryRequest) -> StreamingResponse:
    return StreamingResponse(
        _token_generator(request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
