from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetryConfig:
    max_retries: int = 2
    retry_delay: float = 1.0
    backoff_factor: float = 2.0


@dataclass
class EvaluationConfig:
    exact_match: bool = True
    fuzzy_match: bool = True
    keyword_match: bool = True
    semantic_evaluation: bool = True
    llm_judge: bool = True


@dataclass
class BenchmarkConfig:
    model: str
    provider: str | None = None

    dataset_path: str = "datasets/benchmark.json"
    dataset_metadata_path: str = "datasets/metadata.json"

    question_limit: int | None = None

    evaluation: EvaluationConfig = field(
        default_factory=EvaluationConfig
    )

    retry: RetryConfig = field(
        default_factory=RetryConfig
    )

    enable_fallback: bool = False
    fallback_model: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "provider": self.provider,
            "dataset_path": self.dataset_path,
            "dataset_metadata_path": self.dataset_metadata_path,
            "question_limit": self.question_limit,
            "evaluation": {
                "exact_match": self.evaluation.exact_match,
                "fuzzy_match": self.evaluation.fuzzy_match,
                "keyword_match": self.evaluation.keyword_match,
                "semantic_evaluation": self.evaluation.semantic_evaluation,
                "llm_judge": self.evaluation.llm_judge,
            },
            "retry": {
                "max_retries": self.retry.max_retries,
                "retry_delay": self.retry.retry_delay,
                "backoff_factor": self.retry.backoff_factor,
            },
            "enable_fallback": self.enable_fallback,
            "fallback_model": self.fallback_model,
            "metadata": self.metadata,
        }