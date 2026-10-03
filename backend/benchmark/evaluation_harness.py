from typing import Any


class EvaluationHarness:
    """
    Aggregates benchmark metrics into a standardized
    evaluation report for an experiment.
    """

    QUALITY_METRICS = (
        "accuracy",
        "exact_match",
        "fuzzy_match",
        "average_fuzzy_similarity",
        "semantic_match",
        "average_semantic_similarity",
    )

    RELIABILITY_METRICS = (
        "error_rate",
    )

    PERFORMANCE_METRICS = (
        "average_latency",
        "p50_latency",
        "p95_latency",
        "tokens_per_second",
    )

    def __init__(self, experiment: dict[str, Any]):
        self.experiment = experiment
        self.metrics = experiment.get("metrics", {})

    @staticmethod
    def _number(
        value: Any,
        default: float = 0.0,
    ) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def evaluate_quality(self) -> dict[str, float]:
        """
        Extract quality-related evaluation metrics.
        """

        return {
            metric: self._number(self.metrics.get(metric))
            for metric in self.QUALITY_METRICS
        }

    def evaluate_reliability(self) -> dict[str, float]:
        """
        Extract reliability-related metrics.
        """

        retry = self.metrics.get("retry", {})

        return {
            "error_rate": self._number(
                self.metrics.get("error_rate")
            ),
            "retry_rate": self._number(
                retry.get("retry_rate")
            ),
            "fallback_rate": self._number(
                retry.get("fallback_rate")
            ),
            "fallback_success_rate": self._number(
                retry.get("fallback_success_rate")
            ),
        }

    def evaluate_performance(self) -> dict[str, float]:
        """
        Extract latency and throughput metrics.
        """

        return {
            metric: self._number(self.metrics.get(metric))
            for metric in self.PERFORMANCE_METRICS
        }

    def evaluate_cost(self) -> dict[str, float]:
        """
        Extract cost-related metrics.
        """

        cost = self.metrics.get("cost", {})

        total_cost = self._number(
            cost.get("total_cost_usd")
        )

        total_questions = self._number(
            self.metrics.get("total")
        )

        cost_per_question = (
            total_cost / total_questions
            if total_questions > 0
            else 0.0
        )

        return {
            "input_cost_usd": self._number(
                cost.get("input_cost_usd")
            ),
            "output_cost_usd": self._number(
                cost.get("output_cost_usd")
            ),
            "total_cost_usd": total_cost,
            "cost_per_question_usd": cost_per_question,
        }

    def evaluate_categories(self) -> dict[str, dict[str, float]]:
        """
        Extract category-level benchmark performance.
        """

        categories = self.metrics.get("categories", {})

        result: dict[str, dict[str, float]] = {}

        for category, values in categories.items():
            if not isinstance(values, dict):
                continue

            result[category] = {
                "total": self._number(
                    values.get("total")
                ),
                "successful": self._number(
                    values.get("successful")
                ),
                "failed": self._number(
                    values.get("failed")
                ),
                "correct": self._number(
                    values.get("correct")
                ),
                "accuracy": self._number(
                    values.get("accuracy")
                ),
            }

        return result

    def evaluate_summary(self) -> dict[str, Any]:
        """
        Generate a high-level evaluation summary.
        """

        total = self._number(
            self.metrics.get("total")
        )

        successful = self._number(
            self.metrics.get("successful")
        )

        failed = self._number(
            self.metrics.get("failed")
        )

        correct = self._number(
            self.metrics.get("correct")
        )

        return {
            "total_questions": total,
            "successful_questions": successful,
            "failed_questions": failed,
            "correct_answers": correct,
            "accuracy": self._number(
                self.metrics.get("accuracy")
            ),
            "success_rate": (
                successful / total
                if total > 0
                else 0.0
            ),
            "error_rate": self._number(
                self.metrics.get("error_rate")
            ),
        }

    def evaluate(self) -> dict[str, Any]:
        """
        Generate the complete evaluation report.
        """

        return {
            "experiment_id": self.experiment.get(
                "experiment_id"
            ),
            "model": self.experiment.get("model"),
            "timestamp": self.experiment.get("timestamp"),
            "dataset": self.experiment.get("dataset"),
            "quality": self.evaluate_quality(),
            "reliability": self.evaluate_reliability(),
            "performance": self.evaluate_performance(),
            "cost": self.evaluate_cost(),
            "categories": self.evaluate_categories(),
            "summary": self.evaluate_summary(),
        }