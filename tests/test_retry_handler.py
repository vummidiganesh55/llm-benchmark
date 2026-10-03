from backend.benchmark.retry_handler import RetryHandler


def test_success_without_retry():
    handler = RetryHandler(
        max_retries=2,
        retry_delay=0
    )

    calls = {"count": 0}

    def provider():
        calls["count"] += 1
        return {"response": "success"}

    result = handler.execute(provider)

    assert result["success"] is True
    assert result["attempts"] == 1
    assert result["retries"] == 0
    assert result["result"]["response"] == "success"
    assert calls["count"] == 1


def test_retry_after_timeout():
    handler = RetryHandler(
        max_retries=2,
        retry_delay=0
    )

    calls = {"count": 0}

    def provider():
        calls["count"] += 1

        if calls["count"] < 3:
            raise TimeoutError("request timed out")

        return {"response": "success"}

    result = handler.execute(provider)

    assert result["success"] is True
    assert result["attempts"] == 3
    assert result["retries"] == 2
    assert result["result"]["response"] == "success"


def test_non_retryable_error():
    handler = RetryHandler(
        max_retries=2,
        retry_delay=0
    )

    calls = {"count": 0}

    def provider():
        calls["count"] += 1
        raise ValueError("invalid request")

    result = handler.execute(provider)

    assert result["success"] is False
    assert result["attempts"] == 1
    assert result["retries"] == 0
    assert calls["count"] == 1


def test_fallback_after_primary_failure():
    handler = RetryHandler(
        max_retries=1,
        retry_delay=0
    )

    primary_calls = {"count": 0}
    fallback_calls = {"count": 0}

    def primary():
        primary_calls["count"] += 1
        raise TimeoutError("primary timeout")

    def fallback():
        fallback_calls["count"] += 1
        return {"response": "fallback success"}

    result = handler.execute_with_fallback(
        primary_function=primary,
        fallback_function=fallback
    )

    assert result["success"] is True
    assert result["source"] == "fallback"
    assert result["fallback_used"] is True
    assert result["result"]["response"] == "fallback success"

    assert primary_calls["count"] == 2
    assert fallback_calls["count"] == 1


def test_fallback_not_used_when_primary_succeeds():
    handler = RetryHandler(
        max_retries=2,
        retry_delay=0
    )

    primary_calls = {"count": 0}
    fallback_calls = {"count": 0}

    def primary():
        primary_calls["count"] += 1
        return {"response": "primary success"}

    def fallback():
        fallback_calls["count"] += 1
        return {"response": "fallback success"}

    result = handler.execute_with_fallback(
        primary_function=primary,
        fallback_function=fallback
    )

    assert result["success"] is True
    assert result["source"] == "primary"
    assert result["fallback_used"] is False
    assert result["result"]["response"] == "primary success"

    assert primary_calls["count"] == 1
    assert fallback_calls["count"] == 0