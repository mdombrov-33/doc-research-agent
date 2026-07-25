import os

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

import pytest
from fastapi.testclient import TestClient

from src.api.rate_limit import limiter
from src.main import app


@pytest.fixture
def client():
    # Plain TestClient (no context manager) does not run the lifespan handler,
    # so app.state is never populated and Qdrant is never touched - tests inject
    # their own resources via app.dependency_overrides (cleared after each test).
    # Disable rate limiting so request counts don't make tests order-dependent;
    # tests/integration/test_api.py exercises the limit explicitly via limiter.enabled.
    limiter.enabled = False
    yield TestClient(app)
    limiter.enabled = True
    app.dependency_overrides.clear()
