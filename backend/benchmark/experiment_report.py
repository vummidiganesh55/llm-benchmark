from typing import Any

from backend.benchmark.evaluation_harness import EvaluationHarness


class ExperimentReportGenerator:
    """
    Generates a structured human-readable report
    from a stored benchmark experiment.
    """

    def __init__(self, experiment: dict[str, Any]):
        self.experiment = experiment
        self.evaluation = EvaluationHarness(
            experiment
        ).evaluate()

    def _format_percentage(self, value: Any) -> str:
        try:
            return f"{float(value) * 100:.2f}%"
        except (TypeError, ValueError):
            return "0.00%"

    def _format_number(
        self,
        value: Any,
        decimals: int = 4,
    ) -> str:
        try:
            return f"{float(value):.{decimals}f}"
        except (TypeError, ValueError):
            return "0.0000"

    def generate_summary(self) -> dict[str, Any]:
        summary = self.evaluation["summary"]

        return {
            "experiment_id": self.evaluation.get(
                "experiment_id"
            ),
            "model": self.evaluation.get("model"),
            "total_questions": summary[
                "total_questions"
            ],
            "successful_questions": summary[
                "successful_questions"
            ],
            "failed_questions": summary[
                "failed_questions"
            ],
            "accuracy": summary["accuracy"],
            "error_rate": summary["error_rate"],
        }

    def generate_quality_section(self) -> dict[str, Any]:
        quality = self.evaluation["quality"]

        return {
            "accuracy": quality["accuracy"],
            "exact_match": quality["exact_match"],
            "fuzzy_match": quality["fuzzy_match"],
            "average_fuzzy_similarity": quality[
                "average_fuzzy_similarity"
            ],
            "semantic_match": quality[
                "semantic_match"
            ],
            "average_semantic_similarity": quality[
                "average_semantic_similarity"
            ],
        }

    def generate_reliability_section(self) -> dict[str, Any]:
        return dict(
            self.evaluation["reliability"]
        )

    def generate_performance_section(self) -> dict[str, Any]:
        return dict(
            self.evaluation["performance"]
        )

    def generate_cost_section(self) -> dict[str, Any]:
        return dict(
            self.evaluation["cost"]
        )

    def generate_category_section(self) -> dict[str, Any]:
        return dict(
            self.evaluation["categories"]
        )

    def generate_report(self) -> dict[str, Any]:
        """
        Generate the complete structured experiment report.
        """

        return {
            "report_type": "LLM Benchmark Experiment Report",
            "experiment": {
                "experiment_id": self.evaluation.get(
                    "experiment_id"
                ),
                "model": self.evaluation.get(
                    "model"
                ),
                "timestamp": self.evaluation.get(
                    "timestamp"
                ),
                "dataset": self.evaluation.get(
                    "dataset"
                ),
            },
            "summary": self.generate_summary(),
            "quality": self.generate_quality_section(),
            "reliability": (
                self.generate_reliability_section()
            ),
            "performance": (
                self.generate_performance_section()
            ),
            "cost": self.generate_cost_section(),
            "categories": (
                self.generate_category_section()
            ),
        }

    def generate_text_report(self) -> str:
        """
        Generate a human-readable text report.
        """

        report = self.generate_report()

        summary = report["summary"]
        quality = report["quality"]
        reliability = report["reliability"]
        performance = report["performance"]
        cost = report["cost"]
        categories = report["categories"]

        lines = [
            "=" * 60,
            "LLM BENCHMARK EXPERIMENT REPORT",
            "=" * 60,
            "",
            "EXPERIMENT",
            "-" * 60,
            f"Experiment ID : "
            f"{summary['experiment_id']}",
            f"Model         : "
            f"{summary['model']}",
            "",
            "SUMMARY",
            "-" * 60,
            f"Total Questions      : "
            f"{summary['total_questions']}",
            f"Successful Questions : "
            f"{summary['successful_questions']}",
            f"Failed Questions     : "
            f"{summary['failed_questions']}",
            f"Accuracy             : "
            f"{self._format_percentage(summary['accuracy'])}",
            f"Error Rate           : "
            f"{self._format_percentage(summary['error_rate'])}",
            "",
            "QUALITY",
            "-" * 60,
            f"Accuracy                  : "
            f"{self._format_percentage(quality['accuracy'])}",
            f"Exact Match               : "
            f"{self._format_percentage(quality['exact_match'])}",
            f"Fuzzy Match               : "
            f"{self._format_percentage(quality['fuzzy_match'])}",
            f"Avg Fuzzy Similarity     : "
            f"{self._format_number(quality['average_fuzzy_similarity'])}",
            f"Semantic Match            : "
            f"{self._format_percentage(quality['semantic_match'])}",
            f"Avg Semantic Similarity  : "
            f"{self._format_number(quality['average_semantic_similarity'])}",
            "",
            "RELIABILITY",
            "-" * 60,
            f"Error Rate               : "
            f"{self._format_percentage(reliability['error_rate'])}",
            f"Retry Rate               : "
            f"{self._format_percentage(reliability['retry_rate'])}",
            f"Fallback Rate            : "
            f"{self._format_percentage(reliability['fallback_rate'])}",
            f"Fallback Success Rate    : "
            f"{self._format_percentage(reliability['fallback_success_rate'])}",
            "",
            "PERFORMANCE",
            "-" * 60,
            f"Average Latency          : "
            f"{self._format_number(performance['average_latency'])} s",
            f"P50 Latency              : "
            f"{self._format_number(performance['p50_latency'])} s",
            f"P95 Latency              : "
            f"{self._format_number(performance['p95_latency'])} s",
            f"Tokens / Second          : "
            f"{self._format_number(performance['tokens_per_second'])}",
            "",
            "COST",
            "-" * 60,
            f"Input Cost               : "
            f"${self._format_number(cost['input_cost_usd'], 6)}",
            f"Output Cost              : "
            f"${self._format_number(cost['output_cost_usd'], 6)}",
            f"Total Cost               : "
            f"${self._format_number(cost['total_cost_usd'], 6)}",
            f"Cost / Question         : "
            f"${self._format_number(cost['cost_per_question_usd'], 6)}",
            "",
            "CATEGORY PERFORMANCE",
            "-" * 60,
        ]

        for category, values in categories.items():
            lines.extend(
                [
                    f"{category.upper()}",
                    f"  Total      : "
                    f"{values['total']}",
                    f"  Successful : "
                    f"{values['successful']}",
                    f"  Failed     : "
                    f"{values['failed']}",
                    f"  Correct    : "
                    f"{values['correct']}",
                    f"  Accuracy   : "
                    f"{self._format_percentage(values['accuracy'])}",
                    "",
                ]
            )

        lines.append("=" * 60)

        return "\n".join(lines)