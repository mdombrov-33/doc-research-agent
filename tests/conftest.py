import pytest
from fastapi.testclient import TestClient

from src.main import app


@pytest.fixture
def client():
    # Plain TestClient (no context manager) does not run the lifespan handler,
    # so app.state is never populated and Qdrant is never touched - tests inject
    # their own resources via app.dependency_overrides (cleared after each test).
    yield TestClient(app)
    app.dependency_overrides.clear()
