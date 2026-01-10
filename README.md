# Document Research Agent

Production RAG system with LangGraph state machine, hybrid search, and NeMo Guardrails security layer.

## Architecture

### System Overview
```
┌─────────────┐
│   FastAPI   │  REST API (upload, query endpoints)
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│ NeMo Guardrails  │  Security layer (jailbreak detection, prompt injection prevention)
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  LangGraph Agent │  State machine orchestration (see flow below)
└──────────────────┘
```

### LangGraph Agent State Machine

```
                    ┌─────────┐
                    │  START  │
                    └────┬────┘
                         │
                         ▼
                   ┌───────────┐
                   │  Router   │ (Classify: vectorstore or websearch?)
                   └─────┬─────┘
                         │
              ┌──────────┴──────────┐
              │                     │
         web_search=true       web_search=false
              │                     │
              ▼                     ▼
        ┌──────────┐          ┌──────────┐
        │WebSearch │          │ Retrieve │ (Hybrid: Vector + BM25)
        └────┬─────┘          └────┬─────┘
             │                     │
             └──────────┬──────────┘
                        │
                        ▼
                 ┌─────────────┐
                 │ Grade Docs  │ (Batch LLM: relevant?)
                 └──────┬──────┘
                        │
              ┌─────────┴─────────┐
              │                   │
       docs < threshold      docs >= threshold
              │                   │
              ▼                   ▼
        ┌──────────┐          ┌──────────┐
        │WebSearch │          │ Generate │ (Synthesize answer)
        │ (retry)  │          └────┬─────┘
        └────┬─────┘               │
             │                     ▼
             │              ┌────────────────┐
             │              │Check Hallucin. │ (Grounded in docs?)
             │              └───────┬────────┘
             │                      │
             │                      ▼
             │              ┌────────────────┐
             │              │ Check Quality  │ (Answers question?)
             │              └───────┬────────┘
             │                      │
             └──────────────────────┤
                              ┌─────┴─────┐
                              │           │
                          useful    not useful
                              │           │
                              ▼           │
                          ┌─────┐        │
                          │ END │        │
                          └─────┘        │
                                         │
                              (max 3 attempts)
                                         │
                              ┌──────────┘
                              │
                              ▼
                        ┌──────────┐
                        │ Generate │ (Retry)
                        │ (again)  │
                        └──────────┘
```

**Node Descriptions:**
- **Router**: LLM classifies query type (vectorstore vs websearch)
- **Retrieve**: Hybrid search (60% vector similarity + 40% BM25 keyword)
- **WebSearch**: DuckDuckGo fallback when docs insufficient
- **Grade Docs**: Batch LLM grading (10 docs → 1 API call)
- **Generate**: Synthesize answer from graded documents
- **Check Hallucination**: Verify answer grounded in sources
- **Check Quality**: Verify answer resolves user question

## How It Works

### 1. Document Upload & Processing

Documents (PDF, DOCX, TXT) are chunked with 1000-character overlap, embedded using `text-embedding-3-small` (1536 dimensions), and stored in Qdrant with metadata (filename, page numbers, chunk index).

### 2. Query Flow

**Router Node:**
- LLM classifies query as `vectorstore` (document-based) or `websearch` (external knowledge)
- Explicit phrases ("search web", "check online") force web search path
- Routes to appropriate retrieval strategy

**Retrieve Node (Hybrid Search):**
- Query rewriting via LLM for better semantic matching
- Vector search: Qdrant cosine similarity (k=10)
- BM25 search: Keyword-based ranking using spaCy tokenization
- Fusion ranking: Weighted combination (60% vector, 40% BM25)
- Scores normalized and combined for final ranking

**Grade Documents Node:**
- Batch LLM grading of all retrieved documents in parallel
- Binary relevance scoring (yes/no) per document
- Filters to relevant documents only
- If below threshold (default: 3 docs), triggers web search fallback
- Web search results merged with vector results and re-graded

**Generate Node:**
- Synthesizes answer from graded documents
- Uses structured prompt with document context
- Tracks generation attempts for retry logic

**Quality Check:**
- Hallucination detection: Verifies answer is grounded in source documents
- Answer quality: Checks if response resolves the original question
- Regenerates if quality checks fail (max 3 attempts)

### 3. Security Layer (NeMo Guardrails)

Colang flows wrap the RAG pipeline:
- Pre-query: Jailbreak detection, prompt injection filtering
- Post-generation: Output validation, PII redaction
- Rejects malicious inputs before reaching LLM

### 4. State Management

LangGraph `AgentState` (TypedDict) tracks:
- `question`: Original user query
- `documents`: Retrieved/graded document list
- `generation`: Current answer
- `web_search`: Boolean flag for web search routing
- `retrieval_attempts`: Retry counter for document retrieval
- `generation_attempts`: Retry counter for answer generation

Nodes return partial state updates (dicts), LangGraph merges them automatically.

### 5. Performance Optimizations

- Batch document grading: Single LLM call for N documents (vs N sequential calls)
- Fusion retrieval: Combines semantic + keyword search strengths
- Async processing: FastAPI async handlers with concurrent LLM calls
- Connection pooling: Qdrant client reuse across requests

### 6. Evaluation & Monitoring

RAG-specific metrics tracked per query:
- **Retrieval Precision**: Ratio of relevant documents to total retrieved
- **Faithfulness**: Hallucination detection via LLM-as-judge (answer grounded in sources)
- **Answer Quality**: LLM evaluation of response usefulness
- **Latency Tracking**: End-to-end query processing time
- **Web Search Rate**: Percentage of queries requiring external knowledge

Metrics aggregated in-memory and accessible via `/api/evaluation/stats` endpoint. Logs structured evaluation data for each query (JSON format) enabling post-hoc analysis and performance monitoring.

## API Endpoints

**POST /api/upload**
- Upload documents (PDF, DOCX, TXT)
- Chunks, embeds, and stores in Qdrant with metadata
- Response: `{document_id, filename, chunks_created, file_size}`

**POST /api/query**
- Query documents with RAG pipeline
- Request: `{question}`
- Response: `{question, answer, sources_count}`
- Triggers full agent flow: routing → retrieval → grading → generation → quality checks

**GET /api/evaluation/stats**
- Aggregated evaluation metrics across all queries
- Response: `{total_queries, hallucination_pass_rate, quality_pass_rate, avg_retrieval_precision, avg_latency_ms, web_search_rate, avg_docs_retrieved, avg_docs_relevant, avg_generation_attempts}`

**HEAD /api/ping**
- Health check endpoint for monitoring (UptimeRobot, etc.)

## Tech Stack

- **LangGraph** - State machine for agent flow control
- **LangChain** - LLM integration and prompt management
- **NeMo Guardrails** - Security and safety layer
- **Qdrant** - Vector database for embeddings
- **FastAPI** - Async REST API framework
- **OpenAI** - Embeddings and LLM inference
- **BM25 + spaCy** - Keyword-based retrieval and tokenization
- **Pydantic** - Configuration and type safety
- **Docker** - Containerized deployment
