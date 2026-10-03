from backend.benchmark.metrics import BenchmarkMetrics


def test_retry_metrics_without_retries():

    results = [
        {
            "status": "success",
            "retry": {
                "attempts": 1,
                "retries": 0,
                "fallback_used": False,
                "source": "primary",
            },
        },
        {
            "status": "success",
            "retry": {
                "attempts": 1,
                "retries": 0,
                "fallback_used": False,
                "source": "primary",
            },
        },
    ]

    metrics = BenchmarkMetrics.calculate_retry_metrics(results)

    assert metrics["retry_rate"] == 0.0
    assert metrics["fallback_rate"] == 0.0
    assert metrics["fallback_success_rate"] == 0.0
    assert metrics["average_attempts"] == 1.0
    assert metrics["average_retries"] == 0.0
    assert metrics["total_retries"] == 0
    assert metrics["total_attempts"] == 2


def test_retry_metrics_with_retry():

    results = [
        {
            "status": "success",
            "retry": {
                "attempts": 3,
                "retries": 2,
                "fallback_used": False,
                "source": "primary",
            },
        },
        {
            "status": "success",
            "retry": {
                "attempts": 1,
                "retries": 0,
                "fallback_used": False,
                "source": "primary",
            },
        },
    ]

    metrics = BenchmarkMetrics.calculate_retry_metrics(results)

    assert metrics["retry_rate"] == 0.5
    assert metrics["fallback_rate"] == 0.0
    assert metrics["average_attempts"] == 2.0
    assert metrics["average_retries"] == 1.0
    assert metrics["total_retries"] == 2
    assert metrics["total_attempts"] == 4


def test_fallback_success_metrics():

    results = [
        {
            "status": "success",
            "retry": {
                "attempts": 4,
                "retries": 2,
                "fallback_used": True,
                "source": "fallback",
            },
        },
        {
            "status": "error",
            "retry": {
                "attempts": 4,
                "retries": 2,
                "fallback_used": True,
                "source": "fallback",
            },
        },
        {
            "status": "success",
            "retry": {
                "attempts": 1,
                "retries": 0,
                "fallback_used": False,
                "source": "primary",
            },
        },
    ]

    metrics = BenchmarkMetrics.calculate_retry_metrics(results)

    assert metrics["retry_rate"] == 0.6667
    assert metrics["fallback_rate"] == 0.6667
    assert metrics["fallback_success_rate"] == 0.5
    assert metrics["average_attempts"] == 3.0
    assert metrics["average_retries"] == 1.3333
    assert metrics["total_retries"] == 4
    assert metrics["total_attempts"] == 9


def test_empty_results():

    metrics = BenchmarkMetrics.calculate_retry_metrics([])

    assert metrics["retry_rate"] == 0.0
    assert metrics["fallback_rate"] == 0.0
    assert metrics["fallback_success_rate"] == 0.0
    assert metrics["average_attempts"] == 0.0
    assert metrics["average_retries"] == 0.0
    assert metrics["total_retries"] == 0
    assert metrics["total_attempts"] == 0