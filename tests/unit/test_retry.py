import pytest

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
