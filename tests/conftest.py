import pytest
from fastapi.testclient import TestClient

from src.main import app


@pytest.fixture
def client() -> TestClient:
    # Plain TestClient (no context manager) does not run the lifespan handler,
    # so ensure_collection_exists / Qdrant is never touched.
    return TestClient(app)
