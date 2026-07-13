from unittest.mock import MagicMock

import pytest

from src.utils import retry
from src.utils.retry import with_retry


def test_with_retry_recovers_after_one_failure():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("malformed output")
        return "ok"

    assert with_retry(flaky) == "ok"
    assert calls["n"] == 2


def test_with_retry_reraises_after_exhausting_attempts():
    def always_fails():
        raise ValueError("nope")

    with pytest.raises(ValueError):
        with_retry(always_fails, attempts=2)


def test_with_retry_logs_only_the_failure_class(monkeypatch):
    logger = MagicMock()
    monkeypatch.setattr(retry, "logger", logger)
    calls = {"count": 0}

    def flaky():
        calls["count"] += 1
        if calls["count"] == 1:
            raise ValueError("private model output")
        return "ok"

    assert with_retry(flaky) == "ok"
    assert logger.warning.call_args.kwargs == {
        "attempt": 1,
        "failure_type": "ValueError",
    }
