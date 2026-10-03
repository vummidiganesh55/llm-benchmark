import time

import pytest

from backend.benchmark.concurrency_runner import (
    ConcurrencyRunner,
)


def successful_worker(value):
    time.sleep(0.01)
    return value * 2


def failing_worker(value):
    if value == 2:
        raise RuntimeError("intentional test failure")

    return value


def test_empty_inputs():
    runner = ConcurrencyRunner(
        worker=successful_worker,
        max_workers=2,
    )

    result = runner.run([])

    assert result["total_requests"] == 0
    assert result["successful_requests"] == 0
    assert result["failed_requests"] == 0
    assert result["success_rate"] == 0.0
    assert result["error_rate"] == 0.0
    assert result["throughput"] == 0.0


def test_successful_concurrent_execution():
    runner = ConcurrencyRunner(
        worker=successful_worker,
        max_workers=2,
    )

    result = runner.run([1, 2, 3, 4])

    assert result["total_requests"] == 4
    assert result["successful_requests"] == 4
    assert result["failed_requests"] == 0
    assert result["success_rate"] == pytest.approx(1.0)
    assert result["error_rate"] == pytest.approx(0.0)

    assert result["total_time"] > 0
    assert result["average_latency"] > 0
    assert result["throughput"] > 0

    assert result["max_workers"] == 2


def test_results_are_returned():
    runner = ConcurrencyRunner(
        worker=successful_worker,
        max_workers=2,
    )

    result = runner.run([1, 2, 3])

    values = [
        item["result"]
        for item in result["results"]
    ]

    assert values == [2, 4, 6]


def test_failed_requests_are_recorded():
    runner = ConcurrencyRunner(
        worker=failing_worker,
        max_workers=2,
    )

    result = runner.run([1, 2, 3])

    assert result["total_requests"] == 3
    assert result["successful_requests"] == 2
    assert result["failed_requests"] == 1

    assert result["success_rate"] == pytest.approx(
        2 / 3
    )

    assert result["error_rate"] == pytest.approx(
        1 / 3
    )


def test_failure_contains_error_message():
    runner = ConcurrencyRunner(
        worker=failing_worker,
        max_workers=2,
    )

    result = runner.run([1, 2, 3])

    failed = [
        item
        for item in result["results"]
        if not item["success"]
    ]

    assert len(failed) == 1
    assert (
        "intentional test failure"
        in failed[0]["error"]
    )


def test_max_workers_validation():
    with pytest.raises(ValueError):
        ConcurrencyRunner(
            worker=successful_worker,
            max_workers=0,
        )


def test_results_preserve_input_order():
    runner = ConcurrencyRunner(
        worker=successful_worker,
        max_workers=3,
    )

    result = runner.run([5, 1, 3, 2])

    values = [
        item["result"]
        for item in result["results"]
    ]

    assert values == [10, 2, 6, 4]