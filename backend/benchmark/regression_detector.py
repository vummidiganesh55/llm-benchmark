from typing import Any


class RegressionDetector:
    """
    Detects metric regressions between a baseline experiment
    and a new experiment.
    """

    DEFAULT_THRESHOLDS = {
        "accuracy": 0.05,
        "average_latency": 0.20,
        "error_rate": 0.05,
        "tokens_per_second": 0.20,
        "total_cost_usd": 0.20,
    }

    def __init__(
        self,
        thresholds: dict[str, float] | None = None,
    ):
        self.thresholds = {
            **self.DEFAULT_THRESHOLDS,
            **(thresholds or {}),
        }

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    @staticmethod
    def _number(
        value: Any,
        default: float = 0.0,
    ) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    # ---------------------------------------------------------
    # Percentage change
    # ---------------------------------------------------------

    @staticmethod
    def percentage_change(
        baseline: float,
        current: float,
    ) -> float:
        if baseline == 0:
            if current == 0:
                return 0.0
            return float("inf")

        return (
            (current - baseline)
            / abs(baseline)
        )

    # ---------------------------------------------------------
    # Accuracy regression
    # ---------------------------------------------------------

    def check_accuracy(
        self,
        baseline: float,
        current: float,
    ) -> dict[str, Any]:

        change = current - baseline

        threshold = self.thresholds["accuracy"]

        regression = change <= -threshold

        return {
            "metric": "accuracy",
            "baseline": baseline,
            "current": current,
            "absolute_change": change,
            "relative_change": self.percentage_change(
                baseline,
                current,
            ),
            "threshold": threshold,
            "regression": regression,
            "direction": "decrease",
        }

    # ---------------------------------------------------------
    # Latency regression
    # ---------------------------------------------------------

    def check_latency(
        self,
        baseline: float,
        current: float,
    ) -> dict[str, Any]:

        relative_change = self.percentage_change(
            baseline,
            current,
        )

        threshold = self.thresholds["average_latency"]

        regression = relative_change >= threshold

        return {
            "metric": "average_latency",
            "baseline": baseline,
            "current": current,
            "absolute_change": current - baseline,
            "relative_change": relative_change,
            "threshold": threshold,
            "regression": regression,
            "direction": "increase",
        }

    # ---------------------------------------------------------
    # Error-rate regression
    # ---------------------------------------------------------

    def check_error_rate(
        self,
        baseline: float,
        current: float,
    ) -> dict[str, Any]:

        change = current - baseline

        threshold = self.thresholds["error_rate"]

        regression = change >= threshold

        return {
            "metric": "error_rate",
            "baseline": baseline,
            "current": current,
            "absolute_change": change,
            "relative_change": self.percentage_change(
                baseline,
                current,
            ),
            "threshold": threshold,
            "regression": regression,
            "direction": "increase",
        }

    # ---------------------------------------------------------
    # Token efficiency regression
    # ---------------------------------------------------------

    def check_tokens_per_second(
        self,
        baseline: float,
        current: float,
    ) -> dict[str, Any]:

        relative_change = self.percentage_change(
            baseline,
            current,
        )

        threshold = self.thresholds["tokens_per_second"]

        regression = relative_change <= -threshold

        return {
            "metric": "tokens_per_second",
            "baseline": baseline,
            "current": current,
            "absolute_change": current - baseline,
            "relative_change": relative_change,
            "threshold": threshold,
            "regression": regression,
            "direction": "decrease",
        }

    # ---------------------------------------------------------
    # Cost regression
    # ---------------------------------------------------------

    def check_cost(
        self,
        baseline: float,
        current: float,
    ) -> dict[str, Any]:

        relative_change = self.percentage_change(
            baseline,
            current,
        )

        threshold = self.thresholds["total_cost_usd"]

        regression = relative_change >= threshold

        return {
            "metric": "total_cost_usd",
            "baseline": baseline,
            "current": current,
            "absolute_change": current - baseline,
            "relative_change": relative_change,
            "threshold": threshold,
            "regression": regression,
            "direction": "increase",
        }

    # ---------------------------------------------------------
    # Experiment comparison
    # ---------------------------------------------------------

    def compare(
        self,
        baseline_experiment: dict[str, Any],
        current_experiment: dict[str, Any],
    ) -> dict[str, Any]:

        baseline_metrics = (
            baseline_experiment.get("metrics")
            or {}
        )

        current_metrics = (
            current_experiment.get("metrics")
            or {}
        )

        baseline_accuracy = self._number(
            baseline_metrics.get("accuracy")
        )

        current_accuracy = self._number(
            current_metrics.get("accuracy")
        )

        baseline_latency = self._number(
            baseline_metrics.get("average_latency")
        )

        current_latency = self._number(
            current_metrics.get("average_latency")
        )

        baseline_error_rate = self._number(
            baseline_metrics.get("error_rate")
        )

        current_error_rate = self._number(
            current_metrics.get("error_rate")
        )

        baseline_tokens_per_second = self._number(
            baseline_metrics.get("tokens_per_second")
        )

        current_tokens_per_second = self._number(
            current_metrics.get("tokens_per_second")
        )

        baseline_cost = self._number(
            (
                baseline_metrics.get("cost")
                or {}
            ).get("total_cost_usd")
        )

        current_cost = self._number(
            (
                current_metrics.get("cost")
                or {}
            ).get("total_cost_usd")
        )

        checks = {
            "accuracy": self.check_accuracy(
                baseline_accuracy,
                current_accuracy,
            ),
            "average_latency": self.check_latency(
                baseline_latency,
                current_latency,
            ),
            "error_rate": self.check_error_rate(
                baseline_error_rate,
                current_error_rate,
            ),
            "tokens_per_second": self.check_tokens_per_second(
                baseline_tokens_per_second,
                current_tokens_per_second,
            ),
            "total_cost_usd": self.check_cost(
                baseline_cost,
                current_cost,
            ),
        }

        regressions = [
            metric
            for metric, result in checks.items()
            if result["regression"]
        ]

        return {
            "baseline_experiment_id": (
                baseline_experiment.get("experiment_id")
            ),
            "current_experiment_id": (
                current_experiment.get("experiment_id")
            ),
            "baseline_model": (
                baseline_experiment.get("model")
            ),
            "current_model": (
                current_experiment.get("model")
            ),
            "checks": checks,
            "regressions_detected": len(regressions),
            "regressed_metrics": regressions,
            "has_regression": bool(regressions),
        }