from backend.benchmark.statistics import StatisticalAnalyzer


def test_summarize_basic_statistics():
    values = [1, 2, 3, 4, 5]

    result = StatisticalAnalyzer.summarize(values)

    assert result["count"] == 5
    assert result["mean"] == 3.0
    assert result["median"] == 3.0
    assert result["std_dev"] == 1.5811
    assert result["variance"] == 2.5
    assert result["min"] == 1.0
    assert result["max"] == 5.0
    assert result["range"] == 4.0


def test_summarize_empty_values():
    result = StatisticalAnalyzer.summarize([])

    assert result["count"] == 0
    assert result["mean"] is None
    assert result["median"] is None
    assert result["std_dev"] is None
    assert result["variance"] is None
    assert result["min"] is None
    assert result["max"] is None
    assert result["range"] is None


def test_summarize_single_value():
    result = StatisticalAnalyzer.summarize([10])

    assert result["count"] == 1
    assert result["mean"] == 10.0
    assert result["median"] == 10.0
    assert result["std_dev"] is None
    assert result["variance"] is None
    assert result["min"] == 10.0
    assert result["max"] == 10.0
    assert result["range"] == 0.0


def test_summarize_cleans_invalid_values():
    values = [
        1,
        2,
        None,
        "3",
        "invalid",
        True,
        False,
        4,
    ]

    result = StatisticalAnalyzer.summarize(values)

    # "3" is converted to a numeric value.
    # None, "invalid", True, and False are excluded.
    # Cleaned values = [1, 2, 3, 4]
    assert result["count"] == 4
    assert result["mean"] == 2.5
    assert result["median"] == 2.5
    assert result["min"] == 1.0
    assert result["max"] == 4.0
    assert result["range"] == 3.0


def test_confidence_interval_95():
    values = [1, 2, 3, 4, 5]

    result = StatisticalAnalyzer.confidence_interval(
        values,
        confidence=0.95,
    )

    assert result["confidence_level"] == 0.95
    assert result["count"] == 5
    assert result["mean"] == 3.0
    assert result["margin_of_error"] == 1.3859
    assert result["lower"] == 1.6141
    assert result["upper"] == 4.3859


def test_confidence_interval_90():
    values = [1, 2, 3, 4, 5]

    result = StatisticalAnalyzer.confidence_interval(
        values,
        confidence=0.90,
    )

    assert result["confidence_level"] == 0.90
    assert result["count"] == 5
    assert result["mean"] == 3.0
    assert result["margin_of_error"] == 1.1632
    assert result["lower"] == 1.8368
    assert result["upper"] == 4.1632


def test_confidence_interval_99():
    values = [1, 2, 3, 4, 5]

    result = StatisticalAnalyzer.confidence_interval(
        values,
        confidence=0.99,
    )

    assert result["confidence_level"] == 0.99
    assert result["count"] == 5
    assert result["mean"] == 3.0

    # Implementation uses z = 2.576
    assert result["margin_of_error"] == 1.8215
    assert result["lower"] == 1.1785
    assert result["upper"] == 4.8215


def test_confidence_interval_empty():
    result = StatisticalAnalyzer.confidence_interval([])

    assert result["confidence_level"] == 0.95
    assert result["count"] == 0
    assert result["mean"] is None
    assert result["margin_of_error"] is None
    assert result["lower"] is None
    assert result["upper"] is None


def test_confidence_interval_single_value():
    result = StatisticalAnalyzer.confidence_interval([10])

    assert result["confidence_level"] == 0.95
    assert result["count"] == 1
    assert result["mean"] == 10.0
    assert result["margin_of_error"] is None
    assert result["lower"] is None
    assert result["upper"] is None


def test_confidence_interval_invalid_confidence():
    try:
        StatisticalAnalyzer.confidence_interval(
            [1, 2, 3],
            confidence=0.80,
        )
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_analyze_metric():
    experiments = [
        {
            "experiment_id": "exp1",
            "model": "model-a",
            "metrics": {
                "accuracy": 0.90,
            },
        },
        {
            "experiment_id": "exp2",
            "model": "model-a",
            "metrics": {
                "accuracy": 0.95,
            },
        },
        {
            "experiment_id": "exp3",
            "model": "model-a",
            "metrics": {
                "accuracy": 1.0,
            },
        },
    ]

    result = StatisticalAnalyzer.analyze_metric(
        experiments,
        "accuracy",
    )

    assert result["metric"] == "accuracy"
    assert result["summary"]["count"] == 3
    assert result["summary"]["mean"] == 0.95
    assert result["summary"]["median"] == 0.95

    assert result["confidence_interval"]["count"] == 3
    assert result["confidence_interval"]["mean"] == 0.95


def test_analyze_metric_missing_values():
    experiments = [
        {
            "experiment_id": "exp1",
            "model": "model-a",
            "metrics": {
                "accuracy": 0.90,
            },
        },
        {
            "experiment_id": "exp2",
            "model": "model-a",
            "metrics": {},
        },
        {
            "experiment_id": "exp3",
            "model": "model-a",
            "metrics": {
                "accuracy": 1.0,
            },
        },
    ]

    result = StatisticalAnalyzer.analyze_metric(
        experiments,
        "accuracy",
    )

    assert result["metric"] == "accuracy"
    assert result["summary"]["count"] == 2
    assert result["summary"]["mean"] == 0.95


def test_analyze_metric_invalid_experiments():
    experiments = [
        {},
        {
            "model": "model-a",
        },
        {
            "model": "model-a",
            "metrics": None,
        },
        {
            "model": "model-a",
            "metrics": {
                "accuracy": 0.90,
            },
        },
    ]

    result = StatisticalAnalyzer.analyze_metric(
        experiments,
        "accuracy",
    )

    assert result["metric"] == "accuracy"
    assert result["summary"]["count"] == 1
    assert result["summary"]["mean"] == 0.90


def test_analyze_model():
    experiments = [
        {
            "experiment_id": "exp1",
            "model": "qwen2.5:3b",
            "metrics": {
                "accuracy": 0.90,
                "error_rate": 0.10,
                "average_latency": 7.5,
                "p50_latency": 7.0,
                "p95_latency": 10.0,
                "average_provider_latency": 1.2,
                "tokens_per_second": 6.5,
                "total_tokens": 4500,
            },
        },
        {
            "experiment_id": "exp2",
            "model": "qwen2.5:3b",
            "metrics": {
                "accuracy": 0.95,
                "error_rate": 0.05,
                "average_latency": 7.0,
                "p50_latency": 6.8,
                "p95_latency": 9.5,
                "average_provider_latency": 1.1,
                "tokens_per_second": 7.0,
                "total_tokens": 4600,
            },
        },
    ]

    result = StatisticalAnalyzer.analyze_model(experiments)

    assert result["model"] == "qwen2.5:3b"
    assert result["run_count"] == 2

    assert "accuracy" in result["metrics"]
    assert "error_rate" in result["metrics"]
    assert "average_latency" in result["metrics"]
    assert "p50_latency" in result["metrics"]
    assert "p95_latency" in result["metrics"]
    assert "average_provider_latency" in result["metrics"]
    assert "tokens_per_second" in result["metrics"]
    assert "total_tokens" in result["metrics"]

    assert result["metrics"]["accuracy"]["summary"]["mean"] == 0.925
    assert result["metrics"]["error_rate"]["summary"]["mean"] == 0.075
    assert result["metrics"]["average_latency"]["summary"]["mean"] == 7.25
    assert result["metrics"]["p50_latency"]["summary"]["mean"] == 6.9
    assert result["metrics"]["p95_latency"]["summary"]["mean"] == 9.75
    assert result["metrics"]["average_provider_latency"]["summary"]["mean"] == 1.15
    assert result["metrics"]["tokens_per_second"]["summary"]["mean"] == 6.75
    assert result["metrics"]["total_tokens"]["summary"]["mean"] == 4550.0


def test_analyze_model_empty():
    result = StatisticalAnalyzer.analyze_model([])

    assert result["model"] is None
    assert result["run_count"] == 0
    assert result["metrics"] == {}


def test_group_by_model():
    experiments = [
        {
            "experiment_id": "exp1",
            "model": "model-a",
            "metrics": {
                "accuracy": 0.90,
            },
        },
        {
            "experiment_id": "exp2",
            "model": "model-b",
            "metrics": {
                "accuracy": 0.80,
            },
        },
        {
            "experiment_id": "exp3",
            "model": "model-a",
            "metrics": {
                "accuracy": 0.95,
            },
        },
    ]

    result = StatisticalAnalyzer.group_by_model(experiments)

    assert set(result.keys()) == {"model-a", "model-b"}

    assert len(result["model-a"]) == 2
    assert len(result["model-b"]) == 1

    assert result["model-a"][0]["experiment_id"] == "exp1"
    assert result["model-a"][1]["experiment_id"] == "exp3"


def test_group_by_model_ignores_invalid_experiments():
    experiments = [
        {},
        {
            "experiment_id": None,
            "model": "model-a",
            "metrics": {
                "accuracy": 0.90,
            },
        },
        {
            "experiment_id": "exp2",
            "model": None,
            "metrics": {
                "accuracy": 0.80,
            },
        },
        {
            "experiment_id": "exp3",
            "model": "model-b",
            "metrics": {},
        },
        {
            "experiment_id": "exp4",
            "model": "model-c",
            "metrics": {
                "accuracy": 0.95,
            },
        },
    ]

    result = StatisticalAnalyzer.group_by_model(experiments)

    assert set(result.keys()) == {"model-c"}
    assert len(result["model-c"]) == 1
    assert result["model-c"][0]["experiment_id"] == "exp4"


def test_group_by_model_empty():
    result = StatisticalAnalyzer.group_by_model([])

    assert result == {}


def test_analyze_all_models():
    experiments = [
        {
            "experiment_id": "exp1",
            "model": "model-a",
            "metrics": {
                "accuracy": 0.90,
                "average_latency": 5.0,
            },
        },
        {
            "experiment_id": "exp2",
            "model": "model-a",
            "metrics": {
                "accuracy": 0.95,
                "average_latency": 6.0,
            },
        },
        {
            "experiment_id": "exp3",
            "model": "model-b",
            "metrics": {
                "accuracy": 0.80,
                "average_latency": 8.0,
            },
        },
    ]

    result = StatisticalAnalyzer.analyze_all_models(experiments)

    assert result["model_count"] == 2

    assert "model-a" in result["models"]
    assert "model-b" in result["models"]

    assert result["models"]["model-a"]["model"] == "model-a"
    assert result["models"]["model-a"]["run_count"] == 2

    assert result["models"]["model-b"]["model"] == "model-b"
    assert result["models"]["model-b"]["run_count"] == 1

    assert (
        result["models"]["model-a"]["metrics"]["accuracy"]["summary"]["mean"]
        == 0.925
    )

    assert (
        result["models"]["model-b"]["metrics"]["accuracy"]["summary"]["mean"]
        == 0.80
    )


def test_analyze_all_models_empty():
    result = StatisticalAnalyzer.analyze_all_models([])

    assert result["model_count"] == 0
    assert result["models"] == {}