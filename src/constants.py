CLASSIFIER_MODEL = "openai/gpt-5.4-mini"
JUDGE_MODEL = "openai/gpt-5.4-mini"
# Must match the sparse vector name used by langchain_qdrant's QdrantVectorStore.
SPARSE_VECTOR_NAME = "langchain-sparse"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Evals
TOP_K = 10
K = 5
RETRIEVAL_THRESHOLDS = {"recall@5": 0.8, "mrr": 0.7, "ndcg@5": 0.7, "map": 0.6}
EMBEDDING_THRESHOLD = {"embedding_separation": 1.0}
GENERATION_THRESHOLDS = {"faithfulness": 0.75, "answer_relevance": 0.75}
