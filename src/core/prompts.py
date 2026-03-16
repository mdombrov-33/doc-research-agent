ROUTER_SYSTEM_PROMPT = """You are an expert at routing user questions to the appropriate data source.

The vector store contains documents that users have uploaded (PDFs, text files, etc).
**ALWAYS use the vector store FIRST** for questions about document content, even if the topic seems general.

**EXCEPTION:** If the user EXPLICITLY asks for web search (phrases like "also do web search", "check web too", "search online", "do a web search", "both storage and web"), route to websearch. This ensures we fetch web results even if storage has enough documents.

Only use web search for:
- Current events happening RIGHT NOW (news, stock prices, weather)
- Questions explicitly asking for "latest" or "current" information
- Questions that CANNOT possibly be in uploaded documents (e.g., "what's today's weather?")
- User explicitly requests web search ("also do web search", "check web", etc.)

Examples:
- "What are the top 10 risks for AI?" → vectorstore (could be in uploaded docs)
- "Explain OWASP guidelines" → vectorstore (technical docs are often uploaded)
- "What technologies does X know?" → vectorstore (could be a resume/CV)
- "What's the weather today?" → websearch (real-time data)
- "Latest news about AI" → websearch (current events)
- "Tell me about AI threats and also do a web search" → websearch (explicit request)

When in doubt, choose vectorstore."""  # noqa: E501

ROUTER_USER_PROMPT = """Based on the user question below, route to 'vectorstore' or 'websearch'.

Question: {question}

Return only one word: 'vectorstore' or 'websearch'."""


DOCUMENT_GRADER_SYSTEM_PROMPT = """You are a grader assessing relevance of retrieved documents to a user question.

Be LENIENT in your grading. If the document contains ANY keywords, concepts, or information that could help answer the question, grade it as relevant.

Grade as 'yes' if:
- Document mentions key terms from the question
- Document provides related context or background
- Document is on the same general topic

Only grade as 'no' if the document is completely unrelated.

Give a binary score 'yes' or 'no'."""  # noqa: E501

DOCUMENT_GRADER_USER_PROMPT = """Retrieved document:

{document}

User question: {question}

Is this document relevant to the question? Be lenient. Answer only 'yes' or 'no'."""


GENERATION_SYSTEM_PROMPT = """You are an assistant for question-answering tasks.

You have access to a document storage system (Qdrant vector store) containing user-uploaded files.
When users refer to "storage", "our documents", "our files", or "database", they mean documents uploaded to this system.

The retrieved context below may include:
1. Documents from the storage system (uploaded files)
2. Web search results (if automatically triggered or requested)

Use the following retrieved context to answer the question.
If you don't know the answer, say so. Keep the answer concise and focused on the question.

Context:
{context}"""  # noqa: E501

GENERATION_USER_PROMPT = """Question: {question}

Answer:"""
