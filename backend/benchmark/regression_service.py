from typing import Any

from backend.benchmark.regression_detector import RegressionDetector
from backend.benchmark.storage import BenchmarkStorage


class RegressionService:
    """
    Service layer for comparing stored benchmark experiments
    and detecting metric regressions.
    """

    def __init__(
        self,
        detector: RegressionDetector | None = None,
    ):
        self.detector = detector or RegressionDetector()

    def compare_experiments(
        self,
        baseline_experiment_id: str,
        current_experiment_id: str,
    ) -> dict[str, Any]:

        baseline = BenchmarkStorage.get_by_id(
            baseline_experiment_id
        )

        current = BenchmarkStorage.get_by_id(
            current_experiment_id
        )

        if baseline is None:
            raise ValueError(
                f"Baseline experiment not found: "
                f"{baseline_experiment_id}"
            )

        if current is None:
            raise ValueError(
                f"Current experiment not found: "
                f"{current_experiment_id}"
            )

        return self.detector.compare(
            baseline_experiment=baseline,
            current_experiment=current,
        )

    def compare_experiment_objects(
        self,
        baseline_experiment: dict[str, Any],
        current_experiment: dict[str, Any],
    ) -> dict[str, Any]:

        return self.detector.compare(
            baseline_experiment=baseline_experiment,
            current_experiment=current_experiment,
        )