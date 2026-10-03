from backend.benchmark.comparison_engine import ComparisonEngine


def test_compare_two_models():
    engine = ComparisonEngine()

    experiment_a = {
        "experiment_id": "exp_a",
        "model": "model-a",
        "metrics": {
            "accuracy": 0.90,
        },
    }

    experiment_b = {
        "experiment_id": "exp_b",
        "model": "model-b",
        "metrics": {
            "accuracy": 0.95,
        },
    }

    result = engine.compare_two(
        experiment_a,
        experiment_b,
    )

    assert isinstance(result, dict)


def test_compare_preserves_model_names():
    engine = ComparisonEngine()

    experiment_a = {
        "experiment_id": "exp_a",
        "model": "qwen2.5:3b",
        "metrics": {
            "accuracy": 0.90,
        },
    }

    experiment_b = {
        "experiment_id": "exp_b",
        "model": "mistral:latest",
        "metrics": {
            "accuracy": 0.95,
        },
    }

    result = engine.compare_two(
        experiment_a,
        experiment_b,
    )

    assert result["first_model"] == "qwen2.5:3b"
    assert result["second_model"] == "mistral:latest"


def test_compare_preserves_experiment_ids():
    engine = ComparisonEngine()

    experiment_a = {
        "experiment_id": "exp_a",
        "model": "model-a",
        "metrics": {
            "accuracy": 0.90,
        },
    }

    experiment_b = {
        "experiment_id": "exp_b",
        "model": "model-b",
        "metrics": {
            "accuracy": 0.95,
        },
    }

    result = engine.compare_two(
        experiment_a,
        experiment_b,
    )

    assert result["first_experiment_id"] == "exp_a"
    assert result["second_experiment_id"] == "exp_b"


def test_compare_accuracy_difference():
    engine = ComparisonEngine()

    experiment_a = {
        "experiment_id": "exp_a",
        "model": "model-a",
        "metrics": {
            "accuracy": 0.90,
        },
    }

    experiment_b = {
        "experiment_id": "exp_b",
        "model": "model-b",
        "metrics": {
            "accuracy": 0.95,
        },
    }

    result = engine.compare_two(
        experiment_a,
        experiment_b,
    )

    # first - second
    assert result["differences"]["accuracy"] == -0.05


def test_compare_latency_difference():
    engine = ComparisonEngine()

    experiment_a = {
        "experiment_id": "exp_a",
        "model": "model-a",
        "metrics": {
            "average_latency": 5.0,
        },
    }

    experiment_b = {
        "experiment_id": "exp_b",
        "model": "model-b",
        "metrics": {
            "average_latency": 3.0,
        },
    }

    result = engine.compare_two(
        experiment_a,
        experiment_b,
    )

    # first - second
    assert result["differences"]["average_latency"] == 2.0


def test_compare_missing_metric():
    engine = ComparisonEngine()

    experiment_a = {
        "experiment_id": "exp_a",
        "model": "model-a",
        "metrics": {
            "accuracy": 0.90,
        },
    }

    experiment_b = {
        "experiment_id": "exp_b",
        "model": "model-b",
        "metrics": {
            "accuracy": 0.95,
            "average_latency": 4.0,
        },
    }

    result = engine.compare_two(
        experiment_a,
        experiment_b,
    )

    assert "accuracy" in result["differences"]
    assert "average_latency" in result["differences"]

    # Missing metric must remain None.
    assert result["differences"]["average_latency"] is None


def test_compare_nested_metrics():
    engine = ComparisonEngine()

    experiment_a = {
        "experiment_id": "exp_a",
        "model": "model-a",
        "metrics": {
            "llm_judge": {
                "judge_success_rate": 0.90,
                "average_correctness": 0.80,
                "average_relevance": 0.85,
                "average_reasoning_quality": 0.75,
                "average_confidence": 0.90,
            }
        },
    }

    experiment_b = {
        "experiment_id": "exp_b",
        "model": "model-b",
        "metrics": {
            "llm_judge": {
                "judge_success_rate": 0.95,
                "average_correctness": 0.90,
                "average_relevance": 0.95,
                "average_reasoning_quality": 0.85,
                "average_confidence": 0.95,
            }
        },
    }

    result = engine.compare_two(
        experiment_a,
        experiment_b,
    )

    assert result["differences"]["judge_success_rate"] == -0.05
    assert result["differences"]["judge_correctness"] == -0.10
    assert result["differences"]["judge_relevance"] == -0.10
    assert result["differences"]["judge_reasoning_quality"] == -0.10
    assert result["differences"]["judge_confidence"] == -0.05


def test_compare_evaluation_metrics():
    engine = ComparisonEngine()

    experiment_a = {
        "experiment_id": "exp_a",
        "model": "model-a",
        "metrics": {
            "evaluation": {
                "exact_match": 0.90,
                "fuzzy_match": 0.92,
                "average_fuzzy_similarity": 0.91,
                "keyword_match": 0.88,
                "semantic_match": 0.80,
                "average_semantic_similarity": 0.85,
            }
        },
    }

    experiment_b = {
        "experiment_id": "exp_b",
        "model": "model-b",
        "metrics": {
            "evaluation": {
                "exact_match": 0.95,
                "fuzzy_match": 0.96,
                "average_fuzzy_similarity": 0.94,
                "keyword_match": 0.90,
                "semantic_match": 0.85,
                "average_semantic_similarity": 0.90,
            }
        },
    }

    result = engine.compare_two(
        experiment_a,
        experiment_b,
    )

    assert result["differences"]["exact_match"] == -0.05
    assert result["differences"]["fuzzy_match"] == -0.04
    assert result["differences"]["average_fuzzy_similarity"] == -0.03
    assert result["differences"]["keyword_match"] == -0.02
    assert result["differences"]["semantic_match"] == -0.05
    assert result["differences"]["semantic_similarity"] == -0.05


def test_compare_cost_metrics():
    engine = ComparisonEngine()

    experiment_a = {
        "experiment_id": "exp_a",
        "model": "model-a",
        "metrics": {
            "cost": {
                "input_cost_usd": 0.001,
                "output_cost_usd": 0.002,
                "total_cost_usd": 0.003,
            }
        },
    }

    experiment_b = {
        "experiment_id": "exp_b",
        "model": "model-b",
        "metrics": {
            "cost": {
                "input_cost_usd": 0.002,
                "output_cost_usd": 0.003,
                "total_cost_usd": 0.005,
            }
        },
    }

    result = engine.compare_two(
        experiment_a,
        experiment_b,
    )

    assert result["differences"]["input_cost_usd"] == -0.001
    assert result["differences"]["output_cost_usd"] == -0.001
    assert result["differences"]["total_cost_usd"] == -0.002


def test_compare_invalid_experiment_returns_empty_dict():
    engine = ComparisonEngine()

    invalid_experiment = {
        "experiment_id": None,
        "model": "invalid",
        "metrics": {},
    }

    valid_experiment = {
        "experiment_id": "exp_b",
        "model": "model-b",
        "metrics": {
            "accuracy": 0.95,
        },
    }

    result = engine.compare_two(
        invalid_experiment,
        valid_experiment,
    )

    assert result == {}


def test_compare_malformed_experiments_are_ignored():
    engine = ComparisonEngine()

    experiments = [
        {
            "experiment_id": None,
            "model": "legacy-model",
            "metrics": {},
        },
        {
            "experiment_id": "exp_1",
            "model": "model-a",
            "metrics": {
                "accuracy": 0.90,
            },
        },
        {
            "experiment_id": "exp_2",
            "model": "model-b",
            "metrics": {
                "accuracy": 0.95,
            },
        },
    ]

    result = engine.compare(experiments)

    assert result["experiment_count"] == 2
    assert result["ignored_experiments"] == 1
    assert len(result["experiments"]) == 2
    assert len(result["comparisons"]) == 1


def test_compare_empty_experiments():
    engine = ComparisonEngine()

    result = engine.compare([])

    assert result["experiment_count"] == 0
    assert result["ignored_experiments"] == 0
    assert result["experiments"] == []
    assert result["comparisons"] == []


def test_compare_single_experiment():
    engine = ComparisonEngine()

    experiments = [
        {
            "experiment_id": "exp_1",
            "model": "model-a",
            "metrics": {
                "accuracy": 0.90,
            },
        }
    ]

    result = engine.compare(experiments)

    assert result["experiment_count"] == 1
    assert result["ignored_experiments"] == 0
    assert len(result["experiments"]) == 1
    assert result["comparisons"] == []