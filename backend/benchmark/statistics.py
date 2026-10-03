from math import sqrt
from statistics import mean, median, stdev, variance
from typing import Any


class StatisticalAnalyzer:

    @staticmethod
    def _clean_values(
        values: list[Any]
    ) -> list[float]:
        """
        Keep only numeric, non-None values.
        """

        cleaned = []

        for value in values:

            if value is None:
                continue

            if isinstance(
                value,
                bool
            ):
                continue

            try:
                cleaned.append(
                    float(value)
                )
            except (
                TypeError,
                ValueError
            ):
                continue

        return cleaned

    @staticmethod
    def summarize(
        values: list[Any]
    ) -> dict[str, Any]:
        """
        Calculate descriptive statistics for a numeric sample.

        Standard deviation and variance use sample statistics
        when at least two observations are available.
        """

        values = StatisticalAnalyzer._clean_values(
            values
        )

        count = len(values)

        if count == 0:

            return {
                "count": 0,
                "mean": None,
                "median": None,
                "std_dev": None,
                "variance": None,
                "min": None,
                "max": None,
                "range": None
            }

        average = mean(values)

        if count >= 2:

            std_dev = stdev(
                values
            )

            variance_value = variance(
                values
            )

        else:

            std_dev = None

            variance_value = None

        return {

            "count": count,

            "mean": round(
                average,
                4
            ),

            "median": round(
                median(values),
                4
            ),

            "std_dev": (
                round(
                    std_dev,
                    4
                )
                if std_dev is not None
                else None
            ),

            "variance": (
                round(
                    variance_value,
                    4
                )
                if variance_value is not None
                else None
            ),

            "min": round(
                min(values),
                4
            ),

            "max": round(
                max(values),
                4
            ),

            "range": round(
                max(values) - min(values),
                4
            )
        }

    @staticmethod
    def confidence_interval(
        values: list[Any],
        confidence: float = 0.95
    ) -> dict[str, Any]:
        """
        Calculate an approximate two-sided confidence interval
        for the population mean.

        This implementation uses a normal critical value
        approximation.

        For small samples, the interval should be interpreted
        cautiously; a future version can use the Student's
        t-distribution.
        """

        values = StatisticalAnalyzer._clean_values(
            values
        )

        count = len(values)

        if count == 0:

            return {
                "confidence_level": confidence,
                "count": 0,
                "mean": None,
                "margin_of_error": None,
                "lower": None,
                "upper": None
            }

        average = mean(values)

        if count < 2:

            return {
                "confidence_level": confidence,
                "count": count,
                "mean": round(
                    average,
                    4
                ),
                "margin_of_error": None,
                "lower": None,
                "upper": None
            }

        sample_std = stdev(
            values
        )

        # Normal critical values for common confidence levels.
        critical_values = {
            0.90: 1.645,
            0.95: 1.960,
            0.99: 2.576
        }

        z_value = critical_values.get(
            confidence
        )

        if z_value is None:
            raise ValueError(
                "Supported confidence levels: "
                "0.90, 0.95, 0.99"
            )

        standard_error = (
            sample_std
            / sqrt(count)
        )

        margin_of_error = (
            z_value
            * standard_error
        )

        lower = (
            average
            - margin_of_error
        )

        upper = (
            average
            + margin_of_error
        )

        return {

            "confidence_level": confidence,

            "count": count,

            "mean": round(
                average,
                4
            ),

            "margin_of_error": round(
                margin_of_error,
                4
            ),

            "lower": round(
                lower,
                4
            ),

            "upper": round(
                upper,
                4
            )
        }

    @staticmethod
    def analyze_metric(
        experiments: list[dict[str, Any]],
        metric: str
    ) -> dict[str, Any]:
        """
        Analyze one metric across experiments.

        Example:
            metric = "accuracy"
        """

        values = []

        for experiment in experiments:

            metrics = experiment.get(
                "metrics",
                {}
            )

            if not isinstance(
                metrics,
                dict
            ):
                continue

            value = metrics.get(
                metric
            )

            if value is not None:
                values.append(
                    value
                )

        summary = StatisticalAnalyzer.summarize(
            values
        )

        confidence_interval = (
            StatisticalAnalyzer.confidence_interval(
                values
            )
        )

        return {

            "metric": metric,

            "summary": summary,

            "confidence_interval": (
                confidence_interval
            )
        }

    @staticmethod
    def analyze_model(
        experiments: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Calculate statistics across repeated experiments
        for one model.
        """

        if not experiments:

            return {
                "model": None,
                "run_count": 0,
                "metrics": {}
            }

        model = experiments[0].get(
            "model"
        )

        metrics_to_analyze = [
            "accuracy",
            "error_rate",
            "average_latency",
            "p50_latency",
            "p95_latency",
            "average_provider_latency",
            "tokens_per_second",
            "total_tokens"
        ]

        metric_results = {}

        for metric in metrics_to_analyze:

            metric_results[
                metric
            ] = StatisticalAnalyzer.analyze_metric(
                experiments=experiments,
                metric=metric
            )

        return {

            "model": model,

            "run_count": len(
                experiments
            ),

            "metrics": metric_results
        }

    @staticmethod
    def group_by_model(
        experiments: list[dict[str, Any]]
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Group valid experiments by model.
        """

        grouped = {}

        for experiment in experiments:

            experiment_id = experiment.get(
                "experiment_id"
            )

            model = experiment.get(
                "model"
            )

            metrics = experiment.get(
                "metrics"
            )

            if not experiment_id:
                continue

            if not model:
                continue

            if not isinstance(
                metrics,
                dict
            ):
                continue

            if not metrics:
                continue

            grouped.setdefault(
                model,
                []
            ).append(
                experiment
            )

        return grouped

    @staticmethod
    def analyze_all_models(
        experiments: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Calculate repeated-run statistics for every model.
        """

        grouped = StatisticalAnalyzer.group_by_model(
            experiments
        )

        models = {}

        for model, model_experiments in grouped.items():

            models[
                model
            ] = StatisticalAnalyzer.analyze_model(
                model_experiments
            )

        return {

            "model_count": len(
                models
            ),

            "models": models
        }