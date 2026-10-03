import pytest

from backend.benchmark.metrics import BenchmarkMetrics


@pytest.fixture
def sample_results():
    return [
        {
            "status": "success",
            "response": "Paris",
            "latency": 2.0,
            "provider_latency": 1.5,
            "tokens": {
                "input": 10,
                "output": 20,
                "total": 30,
            },
            "evaluation": {
                "exact_match": 1.0,
                "fuzzy_match": 1.0,
                "fuzzy_similarity": 1.0,
                "keyword_match": 1.0,
            },
            "semantic_evaluation": {
                "semantic_match": 1.0,
                "semantic_similarity": 0.95,
            },
            "llm_judge": {
                "correctness": 0.9,
                "relevance": 0.8,
                "reasoning_quality": 0.7,
                "confidence": 0.95,
            },
            "cost": {
                "input_cost_usd": 0.001,
                "output_cost_usd": 0.002,
                "total_cost_usd": 0.003,
            },
            "retry": {
                "attempts": 1,
                "retries": 0,
                "fallback_used": False,
            },
            "category": "knowledge",
        },
        {
            "status": "success",
            "response": "105",
            "latency": 4.0,
            "provider_latency": 3.0,
            "tokens": {
                "input": 20,
                "output": 30,
                "total": 50,
            },
            "evaluation": {
                "exact_match": 0.0,
                "fuzzy_match": 1.0,
                "fuzzy_similarity": 0.9,
                "keyword_match": 0.5,
            },
            "semantic_evaluation": {
                "semantic_match": 0.0,
                "semantic_similarity": 0.70,
            },
            "llm_judge": {
                "correctness": 0.8,
                "relevance": 0.9,
                "reasoning_quality": 0.6,
                "confidence": 0.85,
            },
            "cost": {
                "input_cost_usd": 0.002,
                "output_cost_usd": 0.003,
                "total_cost_usd": 0.005,
            },
            "retry": {
                "attempts": 2,
                "retries": 1,
                "fallback_used": True,
            },
            "category": "math",
        },
    ]


def test_calculate_basic_metrics(sample_results):
    metrics = BenchmarkMetrics.calculate_basic_metrics(
        sample_results
    )

    assert metrics["total"] == 2
    assert metrics["successful"] == 2
    assert metrics["failed"] == 0
    assert metrics["correct"] == 1
    assert metrics["accuracy"] == 0.5
    assert metrics["error_rate"] == 0.0


def test_calculate_basic_metrics_with_failure():
    results = [
        {
            "status": "success",
            "evaluation": {
                "exact_match": 1.0,
            },
        },
        {
            "status": "error",
        },
    ]

    metrics = BenchmarkMetrics.calculate_basic_metrics(
        results
    )

    assert metrics["total"] == 2
    assert metrics["successful"] == 1
    assert metrics["failed"] == 1
    assert metrics["correct"] == 1
    assert metrics["accuracy"] == 1.0
    assert metrics["error_rate"] == 0.5


def test_calculate_latency_metrics(sample_results):
    metrics = BenchmarkMetrics.calculate_latency_metrics(
        sample_results
    )

    assert metrics["average_latency"] == 3.0
    assert metrics["p50_latency"] == 3.0
    assert metrics["p95_latency"] == 3.9
    assert metrics["average_provider_latency"] == 2.25


def test_calculate_latency_metrics_empty():
    metrics = BenchmarkMetrics.calculate_latency_metrics([])

    assert metrics["average_latency"] == 0.0
    assert metrics["p50_latency"] == 0.0
    assert metrics["p95_latency"] == 0.0
    assert metrics["average_provider_latency"] == 0.0


def test_calculate_token_metrics(sample_results):
    metrics = BenchmarkMetrics.calculate_token_metrics(
        sample_results
    )

    assert metrics["input_tokens"] == 30
    assert metrics["output_tokens"] == 50
    assert metrics["total_tokens"] == 80

    # Total latency = 2 + 4 = 6
    # 80 / 6 = 13.3333
    assert metrics["tokens_per_second"] == 13.3333


def test_calculate_evaluation_metrics(sample_results):
    metrics = BenchmarkMetrics.calculate_evaluation_metrics(
        sample_results
    )

    assert metrics["exact_match"] == 0.5
    assert metrics["fuzzy_match"] == 1.0
    assert metrics["average_fuzzy_similarity"] == 0.95
    assert metrics["keyword_match"] == 0.75
    assert metrics["semantic_match"] == 0.5
    assert metrics["average_semantic_similarity"] == 0.825


def test_calculate_evaluation_metrics_empty():
    metrics = BenchmarkMetrics.calculate_evaluation_metrics([])

    assert metrics["exact_match"] == 0.0
    assert metrics["fuzzy_match"] == 0.0
    assert metrics["average_fuzzy_similarity"] == 0.0
    assert metrics["keyword_match"] == 0.0
    assert metrics["semantic_match"] == 0.0
    assert metrics["average_semantic_similarity"] == 0.0


def test_calculate_judge_metrics(sample_results):
    metrics = BenchmarkMetrics.calculate_judge_metrics(
        sample_results
    )

    assert metrics["judge_enabled"] is True
    assert metrics["judge_success_rate"] == 1.0
    assert metrics["average_correctness"] == 0.85
    assert metrics["average_relevance"] == 0.85
    assert metrics["average_reasoning_quality"] == 0.65
    assert metrics["average_confidence"] == 0.9


def test_calculate_judge_metrics_without_judge():
    results = [
        {
            "status": "success",
        }
    ]

    metrics = BenchmarkMetrics.calculate_judge_metrics(
        results
    )

    assert metrics["judge_enabled"] is False
    assert metrics["judge_success_rate"] == 0.0
    assert metrics["average_correctness"] == 0.0


def test_calculate_cost_metrics(sample_results):
    metrics = BenchmarkMetrics.calculate_cost_metrics(
        sample_results
    )

    assert metrics["input_cost_usd"] == 0.003
    assert metrics["output_cost_usd"] == 0.005
    assert metrics["total_cost_usd"] == 0.008


def test_calculate_cost_metrics_empty():
    metrics = BenchmarkMetrics.calculate_cost_metrics([])

    assert metrics["input_cost_usd"] == 0.0
    assert metrics["output_cost_usd"] == 0.0
    assert metrics["total_cost_usd"] == 0.0


def test_calculate_retry_metrics(sample_results):
    metrics = BenchmarkMetrics.calculate_retry_metrics(
        sample_results
    )

    assert metrics["retry_rate"] == 0.5
    assert metrics["fallback_rate"] == 0.5
    assert metrics["fallback_success_rate"] == 1.0

    assert metrics["average_attempts"] == 1.5
    assert metrics["average_retries"] == 0.5

    assert metrics["total_retries"] == 1
    assert metrics["total_attempts"] == 3


def test_calculate_retry_metrics_empty():
    metrics = BenchmarkMetrics.calculate_retry_metrics([])

    assert metrics["retry_rate"] == 0.0
    assert metrics["fallback_rate"] == 0.0
    assert metrics["fallback_success_rate"] == 0.0
    assert metrics["average_attempts"] == 0.0
    assert metrics["average_retries"] == 0.0
    assert metrics["total_retries"] == 0
    assert metrics["total_attempts"] == 0


def test_calculate_category_metrics(sample_results):
    metrics = BenchmarkMetrics.calculate_category_metrics(
        sample_results
    )

    assert "knowledge" in metrics
    assert "math" in metrics

    assert metrics["knowledge"]["total"] == 1
    assert metrics["knowledge"]["successful"] == 1
    assert metrics["knowledge"]["failed"] == 0
    assert metrics["knowledge"]["correct"] == 1
    assert metrics["knowledge"]["accuracy"] == 1.0

    assert metrics["math"]["total"] == 1
    assert metrics["math"]["successful"] == 1
    assert metrics["math"]["failed"] == 0
    assert metrics["math"]["correct"] == 0
    assert metrics["math"]["accuracy"] == 0.0


def test_calculate_category_metrics_failure():
    results = [
        {
            "status": "error",
            "category": "coding",
        }
    ]

    metrics = BenchmarkMetrics.calculate_category_metrics(
        results
    )

    assert metrics["coding"]["total"] == 1
    assert metrics["coding"]["successful"] == 0
    assert metrics["coding"]["failed"] == 1
    assert metrics["coding"]["correct"] == 0
    assert metrics["coding"]["accuracy"] == 0.0


def test_calculate_returns_all_metrics(sample_results):
    metrics = BenchmarkMetrics.calculate(sample_results)

    assert metrics["total"] == 2
    assert metrics["successful"] == 2
    assert metrics["failed"] == 0
    assert metrics["accuracy"] == 0.5

    assert "average_latency" in metrics
    assert "p50_latency" in metrics
    assert "p95_latency" in metrics

    assert "input_tokens" in metrics
    assert "output_tokens" in metrics
    assert "total_tokens" in metrics
    assert "tokens_per_second" in metrics

    assert "exact_match" in metrics
    assert "fuzzy_match" in metrics
    assert "average_fuzzy_similarity" in metrics
    assert "keyword_match" in metrics
    assert "semantic_match" in metrics
    assert "average_semantic_similarity" in metrics

    assert "judge" in metrics
    assert "cost" in metrics
    assert "retry" in metrics
    assert "categories" in metrics


def test_calculate_empty_results():
    metrics = BenchmarkMetrics.calculate([])

    assert metrics["total"] == 0
    assert metrics["successful"] == 0
    assert metrics["failed"] == 0
    assert metrics["accuracy"] == 0.0
    assert metrics["error_rate"] == 0.0

    assert metrics["average_latency"] == 0.0
    assert metrics["total_tokens"] == 0
    assert metrics["tokens_per_second"] == 0.0

    assert metrics["exact_match"] == 0.0
    assert metrics["semantic_match"] == 0.0

    assert metrics["judge"]["judge_enabled"] is False
    assert metrics["cost"]["total_cost_usd"] == 0.0