import pytest

from backend.benchmark.evaluation_harness import EvaluationHarness


@pytest.fixture
def experiment():
    return {
        "experiment_id": "test_eval_001",
        "model": "qwen2.5:3b",
        "timestamp": "2026-10-03T10:00:00",
        "dataset": {
            "name": "LLM Benchmark Dataset",
            "version": "1.0.0",
            "question_count": 10,
        },
        "metrics": {
            "total": 10,
            "successful": 9,
            "failed": 1,
            "correct": 8,
            "accuracy": 0.8,
            "error_rate": 0.1,
            "exact_match": 0.7,
            "fuzzy_match": 0.8,
            "average_fuzzy_similarity": 0.85,
            "semantic_match": 0.6,
            "average_semantic_similarity": 0.75,
            "average_latency": 2.5,
            "p50_latency": 2.2,
            "p95_latency": 4.1,
            "tokens_per_second": 20.0,
            "retry": {
                "retry_rate": 0.1,
                "fallback_rate": 0.0,
                "fallback_success_rate": 0.0,
            },
            "cost": {
                "input_cost_usd": 0.001,
                "output_cost_usd": 0.002,
                "total_cost_usd": 0.003,
            },
            "categories": {
                "math": {
                    "total": 2,
                    "successful": 2,
                    "failed": 0,
                    "correct": 2,
                    "accuracy": 1.0,
                },
                "coding": {
                    "total": 2,
                    "successful": 1,
                    "failed": 1,
                    "correct": 1,
                    "accuracy": 0.5,
                },
            },
        },
    }


def test_quality_evaluation(experiment):
    harness = EvaluationHarness(experiment)

    result = harness.evaluate_quality()

    assert result["accuracy"] == pytest.approx(0.8)
    assert result["exact_match"] == pytest.approx(0.7)
    assert result["fuzzy_match"] == pytest.approx(0.8)
    assert result["average_fuzzy_similarity"] == pytest.approx(0.85)


def test_reliability_evaluation(experiment):
    harness = EvaluationHarness(experiment)

    result = harness.evaluate_reliability()

    assert result["error_rate"] == pytest.approx(0.1)
    assert result["retry_rate"] == pytest.approx(0.1)
    assert result["fallback_rate"] == pytest.approx(0.0)


def test_performance_evaluation(experiment):
    harness = EvaluationHarness(experiment)

    result = harness.evaluate_performance()

    assert result["average_latency"] == pytest.approx(2.5)
    assert result["p50_latency"] == pytest.approx(2.2)
    assert result["p95_latency"] == pytest.approx(4.1)
    assert result["tokens_per_second"] == pytest.approx(20.0)


def test_cost_evaluation(experiment):
    harness = EvaluationHarness(experiment)

    result = harness.evaluate_cost()

    assert result["input_cost_usd"] == pytest.approx(0.001)
    assert result["output_cost_usd"] == pytest.approx(0.002)
    assert result["total_cost_usd"] == pytest.approx(0.003)
    assert result["cost_per_question_usd"] == pytest.approx(0.0003)


def test_category_evaluation(experiment):
    harness = EvaluationHarness(experiment)

    result = harness.evaluate_categories()

    assert result["math"]["accuracy"] == pytest.approx(1.0)
    assert result["coding"]["accuracy"] == pytest.approx(0.5)
    assert result["coding"]["failed"] == pytest.approx(1.0)


def test_summary_evaluation(experiment):
    harness = EvaluationHarness(experiment)

    result = harness.evaluate_summary()

    assert result["total_questions"] == pytest.approx(10)
    assert result["successful_questions"] == pytest.approx(9)
    assert result["failed_questions"] == pytest.approx(1)
    assert result["correct_answers"] == pytest.approx(8)
    assert result["accuracy"] == pytest.approx(0.8)
    assert result["success_rate"] == pytest.approx(0.9)


def test_complete_evaluation(experiment):
    harness = EvaluationHarness(experiment)

    result = harness.evaluate()

    assert result["experiment_id"] == "test_eval_001"
    assert result["model"] == "qwen2.5:3b"

    assert "quality" in result
    assert "reliability" in result
    assert "performance" in result
    assert "cost" in result
    assert "categories" in result
    assert "summary" in result


def test_empty_metrics_are_handled():
    experiment = {
        "experiment_id": "empty_test",
        "model": "test-model",
        "metrics": {},
    }

    harness = EvaluationHarness(experiment)

    result = harness.evaluate()

    assert result["summary"]["total_questions"] == 0
    assert result["summary"]["success_rate"] == 0.0
    assert result["quality"]["accuracy"] == 0.0
    assert result["performance"]["average_latency"] == 0.0
    assert result["cost"]["total_cost_usd"] == 0.0