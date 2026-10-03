from typing import Any


class ModelLeaderboard:
    """
    Builds a model leaderboard from stored benchmark
    experiments.

    The leaderboard does not assign an overall winner.
    It exposes measurable metrics so users can compare
    models according to their own requirements.
    """

    METRICS = (
        "accuracy",
        "average_latency",
        "p95_latency",
        "tokens_per_second",
        "total_cost_usd",
        "error_rate",
    )

    def __init__(
        self,
        experiments: list[dict[str, Any]],
    ):
        self.experiments = experiments

    @staticmethod
    def _number(
        value: Any,
        default: float = 0.0,
    ) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _extract_cost(
        self,
        metrics: dict[str, Any],
    ) -> float:
        cost = metrics.get("cost", {})

        if not isinstance(cost, dict):
            return 0.0

        return self._number(
            cost.get("total_cost_usd")
        )

    def _build_entry(
        self,
        experiment: dict[str, Any],
    ) -> dict[str, Any] | None:

        experiment_id = experiment.get(
            "experiment_id"
        )

        model = experiment.get("model")

        metrics = experiment.get(
            "metrics",
            {},
        )

        if not experiment_id or not model:
            return None

        if not isinstance(metrics, dict):
            return None

        return {
            "experiment_id": experiment_id,
            "model": model,
            "timestamp": experiment.get(
                "timestamp"
            ),
            "accuracy": self._number(
                metrics.get("accuracy")
            ),
            "average_latency": self._number(
                metrics.get("average_latency")
            ),
            "p95_latency": self._number(
                metrics.get("p95_latency")
            ),
            "tokens_per_second": self._number(
                metrics.get("tokens_per_second")
            ),
            "total_cost_usd": self._extract_cost(
                metrics
            ),
            "error_rate": self._number(
                metrics.get("error_rate")
            ),
            "total_questions": self._number(
                metrics.get("total")
            ),
            "successful_questions": self._number(
                metrics.get("successful")
            ),
        }

    def build(
        self,
        sort_by: str = "accuracy",
        descending: bool = True,
    ) -> list[dict[str, Any]]:

        if sort_by not in self.METRICS:
            raise ValueError(
                f"Unsupported sort metric: {sort_by}"
            )

        leaderboard = []

        for experiment in self.experiments:

            entry = self._build_entry(
                experiment
            )

            if entry is not None:
                leaderboard.append(entry)

        leaderboard.sort(
            key=lambda item: item[sort_by],
            reverse=descending,
        )

        return leaderboard

    def group_by_model(
        self,
    ) -> dict[str, list[dict[str, Any]]]:

        grouped: dict[
            str,
            list[dict[str, Any]],
        ] = {}

        for experiment in self.experiments:

            entry = self._build_entry(
                experiment
            )

            if entry is None:
                continue

            model = entry["model"]

            grouped.setdefault(
                model,
                [],
            ).append(entry)

        return grouped

    def latest_by_model(
        self,
    ) -> list[dict[str, Any]]:

        grouped = self.group_by_model()

        latest = []

        for model, experiments in grouped.items():

            experiments.sort(
                key=lambda item: (
                    item.get("timestamp")
                    or ""
                ),
                reverse=True,
            )

            latest.append(
                experiments[0]
            )

        return latest

    def build_latest(
        self,
        sort_by: str = "accuracy",
        descending: bool = True,
    ) -> list[dict[str, Any]]:

        if sort_by not in self.METRICS:
            raise ValueError(
                f"Unsupported sort metric: {sort_by}"
            )

        leaderboard = self.latest_by_model()

        leaderboard.sort(
            key=lambda item: item[sort_by],
            reverse=descending,
        )

        return leaderboard

    def summary(self) -> dict[str, Any]:

        leaderboard = self.build()

        models = {
            entry["model"]
            for entry in leaderboard
        }

        return {
            "experiment_count": len(
                leaderboard
            ),
            "model_count": len(models),
            "models": sorted(models),
            "available_metrics": list(
                self.METRICS
            ),
        }