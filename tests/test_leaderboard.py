import pytest

from backend.benchmark.leaderboard import (
    ModelLeaderboard,
)


@pytest.fixture
def experiments():
    return [
        {
            "experiment_id": "exp_qwen",
            "model": "qwen2.5:3b",
            "timestamp": "2026-10-02T10:00:00",
            "metrics": {
                "total": 50,
                "successful": 50,
                "accuracy": 0.94,
                "average_latency": 7.5,
                "p95_latency": 10.2,
                "tokens_per_second": 6.7,
                "error_rate": 0.0,
                "cost": {
                    "total_cost_usd": 0.0,
                },
            },
        },
        {
            "experiment_id": "exp_llama",
            "model": "llama3.2:latest",
            "timestamp": "2026-10-02T11:00:00",
            "metrics": {
                "total": 50,
                "successful": 50,
                "accuracy": 0.96,
                "average_latency": 7.6,
                "p95_latency": 9.7,
                "tokens_per_second": 3.9,
                "error_rate": 0.0,
                "cost": {
                    "total_cost_usd": 0.0,
                },
            },
        },
        {
            "experiment_id": "exp_qwen_old",
            "model": "qwen2.5:3b",
            "timestamp": "2026-10-01T10:00:00",
            "metrics": {
                "total": 20,
                "successful": 20,
                "accuracy": 0.90,
                "average_latency": 8.0,
                "p95_latency": 11.0,
                "tokens_per_second": 6.0,
                "error_rate": 0.0,
                "cost": {
                    "total_cost_usd": 0.0,
                },
            },
        },
    ]


def test_build_leaderboard(experiments):
    leaderboard = ModelLeaderboard(
        experiments
    )

    result = leaderboard.build()

    assert len(result) == 3

    assert result[0]["model"] == "llama3.2:latest"
    assert result[0]["accuracy"] == pytest.approx(0.96)


def test_sort_by_latency(experiments):
    leaderboard = ModelLeaderboard(
        experiments
    )

    result = leaderboard.build(
        sort_by="p95_latency",
        descending=False,
    )

    assert result[0]["model"] == "llama3.2:latest"
    assert result[0]["p95_latency"] == pytest.approx(
        9.7
    )


def test_sort_by_tokens_per_second(experiments):
    leaderboard = ModelLeaderboard(
        experiments
    )

    result = leaderboard.build(
        sort_by="tokens_per_second",
        descending=True,
    )

    assert result[0]["model"] == "qwen2.5:3b"
    assert result[0]["tokens_per_second"] == pytest.approx(
        6.7
    )


def test_sort_by_cost(experiments):
    leaderboard = ModelLeaderboard(
        experiments
    )

    result = leaderboard.build(
        sort_by="total_cost_usd",
        descending=False,
    )

    assert len(result) == 3

    for entry in result:
        assert entry["total_cost_usd"] == pytest.approx(
            0.0
        )


def test_invalid_sort_metric(experiments):
    leaderboard = ModelLeaderboard(
        experiments
    )

    with pytest.raises(ValueError):
        leaderboard.build(
            sort_by="invalid_metric"
        )


def test_group_by_model(experiments):
    leaderboard = ModelLeaderboard(
        experiments
    )

    result = leaderboard.group_by_model()

    assert len(result) == 2

    assert "qwen2.5:3b" in result
    assert "llama3.2:latest" in result

    assert len(
        result["qwen2.5:3b"]
    ) == 2

    assert len(
        result["llama3.2:latest"]
    ) == 1


def test_latest_by_model(experiments):
    leaderboard = ModelLeaderboard(
        experiments
    )

    result = leaderboard.latest_by_model()

    assert len(result) == 2

    latest_models = {
        entry["model"]: entry
        for entry in result
    }

    assert (
        latest_models["qwen2.5:3b"][
            "experiment_id"
        ]
        == "exp_qwen"
    )

    assert (
        latest_models["llama3.2:latest"][
            "experiment_id"
        ]
        == "exp_llama"
    )


def test_build_latest(experiments):
    leaderboard = ModelLeaderboard(
        experiments
    )

    result = leaderboard.build_latest(
        sort_by="accuracy",
        descending=True,
    )

    assert len(result) == 2

    assert result[0]["model"] == "llama3.2:latest"
    assert result[0]["accuracy"] == pytest.approx(
        0.96
    )


def test_summary(experiments):
    leaderboard = ModelLeaderboard(
        experiments
    )

    result = leaderboard.summary()

    assert result["experiment_count"] == 3
    assert result["model_count"] == 2

    assert "qwen2.5:3b" in result["models"]
    assert "llama3.2:latest" in result["models"]

    assert "accuracy" in result["available_metrics"]
    assert "p95_latency" in result["available_metrics"]
    assert "tokens_per_second" in result["available_metrics"]
    assert "total_cost_usd" in result["available_metrics"]


def test_empty_experiments():
    leaderboard = ModelLeaderboard([])

    result = leaderboard.build()

    assert result == []

    summary = leaderboard.summary()

    assert summary["experiment_count"] == 0
    assert summary["model_count"] == 0
    assert summary["models"] == []