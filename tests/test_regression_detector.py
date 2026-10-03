from backend.benchmark.regression_detector import RegressionDetector
import pytest

def make_experiment(
    experiment_id: str,
    model: str,
    accuracy: float,
    latency: float,
    error_rate: float,
    tokens_per_second: float,
    cost: float,
):
    return {
        "experiment_id": experiment_id,
        "model": model,
        "metrics": {
            "accuracy": accuracy,
            "average_latency": latency,
            "error_rate": error_rate,
            "tokens_per_second": tokens_per_second,
            "cost": {
                "total_cost_usd": cost,
            },
        },
    }


def test_percentage_change():
    detector = RegressionDetector()

    result = detector.percentage_change(
        baseline=100.0,
        current=120.0,
    )

    assert result == 0.20


def test_accuracy_regression():
    detector = RegressionDetector()

    result = detector.check_accuracy(
        baseline=0.95,
        current=0.89,
    )

    assert result["regression"] is True
    assert result["absolute_change"] == pytest.approx(-0.06)


def test_accuracy_no_regression():
    detector = RegressionDetector()

    result = detector.check_accuracy(
        baseline=0.95,
        current=0.92,
    )

    assert result["regression"] is False


def test_latency_regression():
    detector = RegressionDetector()

    result = detector.check_latency(
        baseline=5.0,
        current=6.5,
    )

    assert result["regression"] is True


def test_latency_no_regression():
    detector = RegressionDetector()

    result = detector.check_latency(
        baseline=5.0,
        current=5.5,
    )

    assert result["regression"] is False


def test_error_rate_regression():
    detector = RegressionDetector()

    result = detector.check_error_rate(
        baseline=0.02,
        current=0.10,
    )

    assert result["regression"] is True


def test_tokens_per_second_regression():
    detector = RegressionDetector()

    result = detector.check_tokens_per_second(
        baseline=20.0,
        current=15.0,
    )

    assert result["regression"] is True


def test_cost_regression():
    detector = RegressionDetector()

    result = detector.check_cost(
        baseline=0.01,
        current=0.02,
    )

    assert result["regression"] is True


def test_full_experiment_comparison():
    detector = RegressionDetector()

    baseline = make_experiment(
        experiment_id="baseline",
        model="qwen2.5:3b",
        accuracy=0.95,
        latency=5.0,
        error_rate=0.02,
        tokens_per_second=20.0,
        cost=0.01,
    )

    current = make_experiment(
        experiment_id="current",
        model="qwen2.5:3b",
        accuracy=0.85,
        latency=7.0,
        error_rate=0.10,
        tokens_per_second=14.0,
        cost=0.02,
    )

    result = detector.compare(
        baseline_experiment=baseline,
        current_experiment=current,
    )

    assert result["has_regression"] is True

    assert result["regressions_detected"] == 5

    assert set(result["regressed_metrics"]) == {
        "accuracy",
        "average_latency",
        "error_rate",
        "tokens_per_second",
        "total_cost_usd",
    }


def test_full_experiment_no_regression():
    detector = RegressionDetector()

    baseline = make_experiment(
        experiment_id="baseline",
        model="qwen2.5:3b",
        accuracy=0.90,
        latency=5.0,
        error_rate=0.05,
        tokens_per_second=20.0,
        cost=0.01,
    )

    current = make_experiment(
        experiment_id="current",
        model="qwen2.5:3b",
        accuracy=0.92,
        latency=5.5,
        error_rate=0.04,
        tokens_per_second=19.0,
        cost=0.011,
    )

    result = detector.compare(
        baseline_experiment=baseline,
        current_experiment=current,
    )

    assert result["has_regression"] is False
    assert result["regressions_detected"] == 0
    assert result["regressed_metrics"] == []