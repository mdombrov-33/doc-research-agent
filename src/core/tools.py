from langchain_community.tools import DuckDuckGoSearchRun

from src.core.retrieval.search import get_vector_store


def get_vector_store_tool():
    return get_vector_store()


def get_web_search_tool():
    return DuckDuckGoSearchRun()
