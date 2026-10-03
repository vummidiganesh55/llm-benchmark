from backend.benchmark.config import (
    BenchmarkConfig,
    EvaluationConfig,
    RetryConfig,
)


def test_default_retry_config():
    config = RetryConfig()

    assert config.max_retries == 2
    assert config.retry_delay == 1.0
    assert config.backoff_factor == 2.0


def test_default_evaluation_config():
    config = EvaluationConfig()

    assert config.exact_match is True
    assert config.fuzzy_match is True
    assert config.keyword_match is True
    assert config.semantic_evaluation is True
    assert config.llm_judge is True


def test_benchmark_config():
    config = BenchmarkConfig(
        model="qwen2.5:3b",
        provider="ollama",
        question_limit=10,
    )

    assert config.model == "qwen2.5:3b"
    assert config.provider == "ollama"
    assert config.question_limit == 10


def test_benchmark_config_to_dict():
    config = BenchmarkConfig(
        model="qwen2.5:3b",
        provider="ollama",
        question_limit=10,
    )

    data = config.to_dict()

    assert data["model"] == "qwen2.5:3b"
    assert data["provider"] == "ollama"
    assert data["question_limit"] == 10

    assert data["evaluation"]["semantic_evaluation"] is True
    assert data["evaluation"]["llm_judge"] is True

    assert data["retry"]["max_retries"] == 2