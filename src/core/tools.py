from langchain_community.tools import DuckDuckGoSearchRun

from src.core.retrieval.search import get_retriever, get_vector_store
from src.utils.logger import logger


def get_retriever_tool():
    logger.info("Creating retriever tool")
    retriever = get_retriever()
    return retriever


def get_vector_store_tool():
    logger.info("Getting vector store for similarity search with scores")
    return get_vector_store()


def get_web_search_tool():
    logger.info("Creating web search tool")
    web_search = DuckDuckGoSearchRun()
    return web_search
