import pytest

from backend.benchmark.experiment_report import (
    ExperimentReportGenerator,
)


@pytest.fixture
def experiment():
    return {
        "experiment_id": "report_test_001",
        "model": "qwen2.5:3b",
        "timestamp": "2026-10-03T11:00:00",
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


def test_generate_summary(experiment):
    generator = ExperimentReportGenerator(
        experiment
    )

    result = generator.generate_summary()

    assert result["experiment_id"] == "report_test_001"
    assert result["model"] == "qwen2.5:3b"
    assert result["total_questions"] == 10
    assert result["successful_questions"] == 9
    assert result["failed_questions"] == 1
    assert result["accuracy"] == pytest.approx(0.8)


def test_generate_quality_section(experiment):
    generator = ExperimentReportGenerator(
        experiment
    )

    result = generator.generate_quality_section()

    assert result["accuracy"] == pytest.approx(0.8)
    assert result["exact_match"] == pytest.approx(0.7)
    assert result["fuzzy_match"] == pytest.approx(0.8)
    assert result[
        "average_fuzzy_similarity"
    ] == pytest.approx(0.85)


def test_generate_reliability_section(experiment):
    generator = ExperimentReportGenerator(
        experiment
    )

    result = generator.generate_reliability_section()

    assert result["error_rate"] == pytest.approx(0.1)
    assert result["retry_rate"] == pytest.approx(0.1)
    assert result["fallback_rate"] == pytest.approx(0.0)


def test_generate_performance_section(experiment):
    generator = ExperimentReportGenerator(
        experiment
    )

    result = generator.generate_performance_section()

    assert result["average_latency"] == pytest.approx(2.5)
    assert result["p50_latency"] == pytest.approx(2.2)
    assert result["p95_latency"] == pytest.approx(4.1)
    assert result["tokens_per_second"] == pytest.approx(20.0)


def test_generate_cost_section(experiment):
    generator = ExperimentReportGenerator(
        experiment
    )

    result = generator.generate_cost_section()

    assert result["total_cost_usd"] == pytest.approx(0.003)
    assert result[
        "cost_per_question_usd"
    ] == pytest.approx(0.0003)


def test_generate_category_section(experiment):
    generator = ExperimentReportGenerator(
        experiment
    )

    result = generator.generate_category_section()

    assert "math" in result
    assert "coding" in result
    assert result["math"]["accuracy"] == pytest.approx(1.0)
    assert result["coding"]["accuracy"] == pytest.approx(0.5)


def test_generate_complete_report(experiment):
    generator = ExperimentReportGenerator(
        experiment
    )

    result = generator.generate_report()

    assert (
        result["report_type"]
        == "LLM Benchmark Experiment Report"
    )

    assert "experiment" in result
    assert "summary" in result
    assert "quality" in result
    assert "reliability" in result
    assert "performance" in result
    assert "cost" in result
    assert "categories" in result


def test_generate_text_report(experiment):
    generator = ExperimentReportGenerator(
        experiment
    )

    result = generator.generate_text_report()

    assert isinstance(result, str)

    assert "LLM BENCHMARK EXPERIMENT REPORT" in result
    assert "SUMMARY" in result
    assert "QUALITY" in result
    assert "RELIABILITY" in result
    assert "PERFORMANCE" in result
    assert "COST" in result
    assert "CATEGORY PERFORMANCE" in result

    assert "qwen2.5:3b" in result
    assert "80.00%" in result
    assert "2.5000 s" in result