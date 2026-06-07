from typing import Any

from fastapi import Request
from langchain_qdrant import QdrantVectorStore
from spacy.language import Language

from src.config import Settings, get_settings
from src.core.monitoring.tracker import MetricsTracker

__all__ = [
    "Settings",
    "get_settings",
    "get_vector_store",
    "get_nlp",
    "get_agent",
    "get_metrics_tracker",
]


def get_vector_store(request: Request) -> QdrantVectorStore:
    return request.app.state.vector_store


def get_nlp(request: Request) -> Language:
    return request.app.state.nlp


def get_agent(request: Request) -> Any:
    return request.app.state.agent


def get_metrics_tracker(request: Request) -> MetricsTracker:
    return request.app.state.metrics_tracker
