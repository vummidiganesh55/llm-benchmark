from unittest.mock import patch

import pytest

from backend.benchmark.regression_service import RegressionService


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


def test_compare_experiment_objects():
    service = RegressionService()

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

    result = service.compare_experiment_objects(
        baseline_experiment=baseline,
        current_experiment=current,
    )

    assert result["baseline_experiment_id"] == "baseline"
    assert result["current_experiment_id"] == "current"
    assert result["has_regression"] is True
    assert result["regressions_detected"] == 5


def test_compare_experiment_objects_no_regression():
    service = RegressionService()

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

    result = service.compare_experiment_objects(
        baseline_experiment=baseline,
        current_experiment=current,
    )

    assert result["has_regression"] is False
    assert result["regressions_detected"] == 0
    assert result["regressed_metrics"] == []


def test_compare_experiments_from_storage():
    service = RegressionService()

    baseline = make_experiment(
        experiment_id="exp-baseline",
        model="qwen2.5:3b",
        accuracy=0.95,
        latency=5.0,
        error_rate=0.02,
        tokens_per_second=20.0,
        cost=0.01,
    )

    current = make_experiment(
        experiment_id="exp-current",
        model="qwen2.5:3b",
        accuracy=0.85,
        latency=7.0,
        error_rate=0.10,
        tokens_per_second=14.0,
        cost=0.02,
    )

    with patch(
        "backend.benchmark.regression_service.BenchmarkStorage.get_by_id"
    ) as mock_get_by_id:

        def get_experiment(experiment_id):
            if experiment_id == "exp-baseline":
                return baseline

            if experiment_id == "exp-current":
                return current

            return None

        mock_get_by_id.side_effect = get_experiment

        result = service.compare_experiments(
            baseline_experiment_id="exp-baseline",
            current_experiment_id="exp-current",
        )

    assert result["baseline_experiment_id"] == "exp-baseline"
    assert result["current_experiment_id"] == "exp-current"
    assert result["has_regression"] is True
    assert result["regressions_detected"] == 5

    assert mock_get_by_id.call_count == 2


def test_baseline_experiment_not_found():
    service = RegressionService()

    with patch(
        "backend.benchmark.regression_service.BenchmarkStorage.get_by_id",
        return_value=None,
    ):

        with pytest.raises(
            ValueError,
            match="Baseline experiment not found",
        ):
            service.compare_experiments(
                baseline_experiment_id="missing-baseline",
                current_experiment_id="current",
            )


def test_current_experiment_not_found():
    service = RegressionService()

    baseline = make_experiment(
        experiment_id="baseline",
        model="qwen2.5:3b",
        accuracy=0.95,
        latency=5.0,
        error_rate=0.02,
        tokens_per_second=20.0,
        cost=0.01,
    )

    with patch(
        "backend.benchmark.regression_service.BenchmarkStorage.get_by_id"
    ) as mock_get_by_id:

        def get_experiment(experiment_id):
            if experiment_id == "baseline":
                return baseline

            return None

        mock_get_by_id.side_effect = get_experiment

        with pytest.raises(
            ValueError,
            match="Current experiment not found",
        ):
            service.compare_experiments(
                baseline_experiment_id="baseline",
                current_experiment_id="missing-current",
            )