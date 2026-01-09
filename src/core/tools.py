from langchain_community.tools import DuckDuckGoSearchRun

from src.core.retrieval.search import get_retriever
from src.utils.logger import logger


def get_retriever_tool():
    logger.info("Creating retriever tool")
    retriever = get_retriever()
    return retriever


def get_web_search_tool():
    logger.info("Creating web search tool")
    web_search = DuckDuckGoSearchRun()
    return web_search
