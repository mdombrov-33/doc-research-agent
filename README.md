# Document Research Agent

Agentic document research assistant using LangGraph for intelligent query routing and response synthesis.

## Tech Stack

- **FastAPI** - REST API framework
- **LangGraph** - Agent orchestration and state management
- **LangChain** - LLM integration layer
- **Qdrant** - Vector database for document storage
- **OpenAI/OpenRouter** - LLM providers
- **DuckDuckGo** - Web search integration
- **Pydantic** - Configuration and validation
- **Docker** - Containerization

## How It Works

The agent receives a user query and autonomously decides whether to:

1. Search uploaded documents (vector similarity via Qdrant)
2. Search the web (DuckDuckGo API)
3. Combine both sources

Results are synthesized into a coherent answer using the configured LLM provider.
