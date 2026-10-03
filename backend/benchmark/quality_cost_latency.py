from typing import Any


class QualityLatencyCostAnalyzer:

    # ========================================================
    # SAFE NUMBER
    # ========================================================

    @staticmethod
    def _number(
        value: Any,
        default: float = 0.0,
    ) -> float:

        if value is None:
            return default

        try:
            return float(value)

        except (TypeError, ValueError):
            return default

    # ========================================================
    # QUALITY SCORE
    # ========================================================

    @classmethod
    def calculate_quality_score(
        cls,
        metrics: dict[str, Any],
    ) -> float:

        accuracy = cls._number(
            metrics.get("accuracy")
        )

        exact_match = cls._number(
            metrics.get("exact_match")
        )

        fuzzy_match = cls._number(
            metrics.get("fuzzy_match")
        )

        keyword_match = cls._number(
            metrics.get("keyword_match")
        )

        semantic_match = cls._number(
            metrics.get("semantic_match")
        )

        values = [
            accuracy,
            exact_match,
            fuzzy_match,
            keyword_match,
            semantic_match,
        ]

        return round(
            sum(values) / len(values),
            4,
        )

    # ========================================================
    # EXTRACT LATENCY
    # ========================================================

    @classmethod
    def extract_latency(
        cls,
        metrics: dict[str, Any],
    ) -> dict[str, float]:

        return {
            "average_latency": cls._number(
                metrics.get("average_latency")
            ),
            "p50_latency": cls._number(
                metrics.get("p50_latency")
            ),
            "p95_latency": cls._number(
                metrics.get("p95_latency")
            ),
            "average_provider_latency": cls._number(
                metrics.get(
                    "average_provider_latency"
                )
            ),
        }

    # ========================================================
    # EXTRACT COST
    # ========================================================

    @classmethod
    def extract_cost(
        cls,
        metrics: dict[str, Any],
    ) -> dict[str, float]:

        cost = metrics.get(
            "cost",
            {},
        )

        if not isinstance(
            cost,
            dict,
        ):
            cost = {}

        return {
            "input_cost_usd": cls._number(
                cost.get(
                    "input_cost_usd"
                )
            ),
            "output_cost_usd": cls._number(
                cost.get(
                    "output_cost_usd"
                )
            ),
            "total_cost_usd": cls._number(
                cost.get(
                    "total_cost_usd"
                )
            ),
        }

    # ========================================================
    # TOKEN EFFICIENCY
    # ========================================================

    @classmethod
    def calculate_token_efficiency(
        cls,
        metrics: dict[str, Any],
    ) -> float:

        total_tokens = cls._number(
            metrics.get("total_tokens")
        )

        total_cost = cls.extract_cost(
            metrics
        )["total_cost_usd"]

        if total_cost <= 0:
            return 0.0

        if total_tokens <= 0:
            return 0.0

        return round(
            total_tokens / total_cost,
            4,
        )

    # ========================================================
    # COST PER QUESTION
    # ========================================================

    @classmethod
    def calculate_cost_per_question(
        cls,
        metrics: dict[str, Any],
    ) -> float:

        total_cost = cls.extract_cost(
            metrics
        )["total_cost_usd"]

        total_questions = cls._number(
            metrics.get("total")
        )

        if total_questions <= 0:
            return 0.0

        return round(
            total_cost / total_questions,
            8,
        )

    # ========================================================
    # QUALITY / LATENCY RATIO
    # ========================================================

    @classmethod
    def calculate_quality_latency_ratio(
        cls,
        metrics: dict[str, Any],
    ) -> float:

        quality = cls.calculate_quality_score(
            metrics
        )

        latency = cls._number(
            metrics.get("average_latency")
        )

        if latency <= 0:
            return 0.0

        return round(
            quality / latency,
            6,
        )

    # ========================================================
    # QUALITY / COST RATIO
    # ========================================================

    @classmethod
    def calculate_quality_cost_ratio(
        cls,
        metrics: dict[str, Any],
    ) -> float:

        quality = cls.calculate_quality_score(
            metrics
        )

        total_cost = cls.extract_cost(
            metrics
        )["total_cost_usd"]

        if total_cost <= 0:
            return 0.0

        return round(
            quality / total_cost,
            6,
        )

    # ========================================================
    # ANALYZE ONE EXPERIMENT
    # ========================================================

    @classmethod
    def analyze_experiment(
        cls,
        experiment: dict[str, Any],
    ) -> dict[str, Any]:

        metrics = experiment.get(
            "metrics",
            {},
        )

        if not isinstance(
            metrics,
            dict,
        ):
            metrics = {}

        latency = cls.extract_latency(
            metrics
        )

        cost = cls.extract_cost(
            metrics
        )

        return {
            "experiment_id": experiment.get(
                "experiment_id"
            ),
            "timestamp": experiment.get(
                "timestamp"
            ),
            "model": experiment.get(
                "model"
            ),
            "dataset": experiment.get(
                "dataset"
            ),

            "quality": {
                "accuracy": cls._number(
                    metrics.get(
                        "accuracy"
                    )
                ),
                "exact_match": cls._number(
                    metrics.get(
                        "exact_match"
                    )
                ),
                "fuzzy_match": cls._number(
                    metrics.get(
                        "fuzzy_match"
                    )
                ),
                "keyword_match": cls._number(
                    metrics.get(
                        "keyword_match"
                    )
                ),
                "semantic_match": cls._number(
                    metrics.get(
                        "semantic_match"
                    )
                ),
                "quality_score": (
                    cls.calculate_quality_score(
                        metrics
                    )
                ),
            },

            "latency": latency,

            "cost": cost,

            "efficiency": {
                "cost_per_question": (
                    cls.calculate_cost_per_question(
                        metrics
                    )
                ),
                "tokens_per_second": cls._number(
                    metrics.get(
                        "tokens_per_second"
                    )
                ),
                "token_efficiency": (
                    cls.calculate_token_efficiency(
                        metrics
                    )
                ),
                "quality_latency_ratio": (
                    cls.calculate_quality_latency_ratio(
                        metrics
                    )
                ),
                "quality_cost_ratio": (
                    cls.calculate_quality_cost_ratio(
                        metrics
                    )
                ),
            },
        }

    # ========================================================
    # ANALYZE ALL EXPERIMENTS
    # ========================================================

    @classmethod
    def analyze_experiments(
        cls,
        experiments: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        analyzed = []

        for experiment in experiments:

            if not isinstance(
                experiment,
                dict,
            ):
                continue

            experiment_id = experiment.get(
                "experiment_id"
            )

            model = experiment.get(
                "model"
            )

            metrics = experiment.get(
                "metrics"
            )

            # Ignore malformed historical records.
            if not experiment_id:
                continue

            if not model:
                continue

            if not isinstance(
                metrics,
                dict,
            ):
                continue

            analyzed.append(
                cls.analyze_experiment(
                    experiment
                )
            )

        return analyzed