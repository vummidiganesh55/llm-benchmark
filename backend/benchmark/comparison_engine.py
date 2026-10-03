from typing import Any


class ComparisonEngine:

    @staticmethod
    def _safe_difference(
        first: float | None,
        second: float | None
    ) -> float | None:
        """
        Calculate first - second.

        Returns None when either metric is unavailable.
        This prevents missing historical metrics from being
        incorrectly interpreted as measured zero values.
        """

        if first is None or second is None:
            return None

        return round(
            first - second,
            4
        )

    @staticmethod
    def _get_metric(
        metrics: dict[str, Any],
        key: str
    ) -> float | int | None:
        """
        Return a metric only when it actually exists.

        Missing metrics return None instead of 0.0.
        """

        if key not in metrics:
            return None

        value = metrics.get(key)

        if value is None:
            return None

        return value

    @staticmethod
    def _get_nested_metric(
        metrics: dict[str, Any],
        section: str,
        key: str
    ) -> float | int | None:
        """
        Safely retrieve a metric from a nested section.
        """

        nested = metrics.get(
            section
        )

        if not isinstance(
            nested,
            dict
        ):
            return None

        if key not in nested:
            return None

        value = nested.get(key)

        if value is None:
            return None

        return value

    @staticmethod
    def _extract_metrics(
        experiment: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Extract metrics from an experiment.

        Missing metrics remain None so historical experiments
        with older schemas are not treated as having zero values.
        """

        metrics = experiment.get(
            "metrics",
            {}
        )

        if not isinstance(
            metrics,
            dict
        ):
            metrics = {}

        return {

            # -----------------------------------------
            # Experiment metadata
            # -----------------------------------------

            "experiment_id": experiment.get(
                "experiment_id"
            ),

            "model": experiment.get(
                "model"
            ),

            "timestamp": experiment.get(
                "timestamp"
            ),

            # -----------------------------------------
            # Basic benchmark metrics
            # -----------------------------------------

            "accuracy": ComparisonEngine._get_metric(
                metrics,
                "accuracy"
            ),

            "error_rate": ComparisonEngine._get_metric(
                metrics,
                "error_rate"
            ),

            # -----------------------------------------
            # Latency metrics
            # -----------------------------------------

            "average_latency": (
                ComparisonEngine._get_metric(
                    metrics,
                    "average_latency"
                )
            ),

            "p50_latency": (
                ComparisonEngine._get_metric(
                    metrics,
                    "p50_latency"
                )
            ),

            "p95_latency": (
                ComparisonEngine._get_metric(
                    metrics,
                    "p95_latency"
                )
            ),

            "average_provider_latency": (
                ComparisonEngine._get_metric(
                    metrics,
                    "average_provider_latency"
                )
            ),

            # -----------------------------------------
            # Throughput
            # -----------------------------------------

            "tokens_per_second": (
                ComparisonEngine._get_metric(
                    metrics,
                    "tokens_per_second"
                )
            ),

            "total_tokens": (
                ComparisonEngine._get_metric(
                    metrics,
                    "total_tokens"
                )
            ),

            # -----------------------------------------
            # Traditional evaluation
            # -----------------------------------------

            "exact_match": (
                ComparisonEngine._get_nested_metric(
                    metrics,
                    "evaluation",
                    "exact_match"
                )
            ),

            "fuzzy_match": (
                ComparisonEngine._get_nested_metric(
                    metrics,
                    "evaluation",
                    "fuzzy_match"
                )
            ),

            "average_fuzzy_similarity": (
                ComparisonEngine._get_nested_metric(
                    metrics,
                    "evaluation",
                    "average_fuzzy_similarity"
                )
            ),

            "keyword_match": (
                ComparisonEngine._get_nested_metric(
                    metrics,
                    "evaluation",
                    "keyword_match"
                )
            ),

            "semantic_match": (
                ComparisonEngine._get_nested_metric(
                    metrics,
                    "evaluation",
                    "semantic_match"
                )
            ),

            "average_semantic_similarity": (
                ComparisonEngine._get_nested_metric(
                    metrics,
                    "evaluation",
                    "average_semantic_similarity"
                )
            ),

            # -----------------------------------------
            # LLM-as-a-Judge
            # -----------------------------------------

            "judge_success_rate": (
                ComparisonEngine._get_nested_metric(
                    metrics,
                    "llm_judge",
                    "judge_success_rate"
                )
            ),

            "judge_correctness": (
                ComparisonEngine._get_nested_metric(
                    metrics,
                    "llm_judge",
                    "average_correctness"
                )
            ),

            "judge_relevance": (
                ComparisonEngine._get_nested_metric(
                    metrics,
                    "llm_judge",
                    "average_relevance"
                )
            ),

            "judge_reasoning_quality": (
                ComparisonEngine._get_nested_metric(
                    metrics,
                    "llm_judge",
                    "average_reasoning_quality"
                )
            ),

            "judge_confidence": (
                ComparisonEngine._get_nested_metric(
                    metrics,
                    "llm_judge",
                    "average_confidence"
                )
            ),

            # -----------------------------------------
            # Cost
            # -----------------------------------------

            "input_cost_usd": (
                ComparisonEngine._get_nested_metric(
                    metrics,
                    "cost",
                    "input_cost_usd"
                )
            ),

            "output_cost_usd": (
                ComparisonEngine._get_nested_metric(
                    metrics,
                    "cost",
                    "output_cost_usd"
                )
            ),

            "total_cost_usd": (
                ComparisonEngine._get_nested_metric(
                    metrics,
                    "cost",
                    "total_cost_usd"
                )
            )
        }

    @staticmethod
    def _is_valid_experiment(
        experiment: dict[str, Any]
    ) -> bool:
        """
        Validate the minimum required experiment fields.
        """

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
            return False

        if not model:
            return False

        if not isinstance(
            metrics,
            dict
        ):
            return False

        if not metrics:
            return False

        return True

    @staticmethod
    def compare(
        experiments: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Compare all valid experiments pairwise.

        Invalid legacy records are ignored.

        Missing metrics are represented as None and therefore
        produce None differences rather than false zero differences.
        """

        if not experiments:
            return {
                "experiment_count": 0,
                "ignored_experiments": 0,
                "experiments": [],
                "comparisons": []
            }

        # -----------------------------------------
        # Filter malformed experiments
        # -----------------------------------------

        valid_experiments = [
            experiment
            for experiment in experiments
            if ComparisonEngine._is_valid_experiment(
                experiment
            )
        ]

        ignored_experiments = (
            len(experiments)
            - len(valid_experiments)
        )

        # -----------------------------------------
        # Extract metrics
        # -----------------------------------------

        extracted = [
            ComparisonEngine._extract_metrics(
                experiment
            )
            for experiment in valid_experiments
        ]

        comparisons = []

        # -----------------------------------------
        # Pairwise comparison
        # -----------------------------------------

        for index, first in enumerate(
            extracted
        ):

            for second in extracted[
                index + 1:
            ]:

                comparisons.append({

                    "first_model": first[
                        "model"
                    ],

                    "second_model": second[
                        "model"
                    ],

                    "first_experiment_id": first[
                        "experiment_id"
                    ],

                    "second_experiment_id": second[
                        "experiment_id"
                    ],

                    "differences": {

                        # ---------------------------------
                        # Basic metrics
                        # ---------------------------------

                        "accuracy": (
                            ComparisonEngine
                            ._safe_difference(
                                first[
                                    "accuracy"
                                ],
                                second[
                                    "accuracy"
                                ]
                            )
                        ),

                        "error_rate": (
                            ComparisonEngine
                            ._safe_difference(
                                first[
                                    "error_rate"
                                ],
                                second[
                                    "error_rate"
                                ]
                            )
                        ),

                        # ---------------------------------
                        # Latency
                        # ---------------------------------

                        "average_latency": (
                            ComparisonEngine
                            ._safe_difference(
                                first[
                                    "average_latency"
                                ],
                                second[
                                    "average_latency"
                                ]
                            )
                        ),

                        "p50_latency": (
                            ComparisonEngine
                            ._safe_difference(
                                first[
                                    "p50_latency"
                                ],
                                second[
                                    "p50_latency"
                                ]
                            )
                        ),

                        "p95_latency": (
                            ComparisonEngine
                            ._safe_difference(
                                first[
                                    "p95_latency"
                                ],
                                second[
                                    "p95_latency"
                                ]
                            )
                        ),

                        "average_provider_latency": (
                            ComparisonEngine
                            ._safe_difference(
                                first[
                                    "average_provider_latency"
                                ],
                                second[
                                    "average_provider_latency"
                                ]
                            )
                        ),

                        # ---------------------------------
                        # Throughput
                        # ---------------------------------

                        "tokens_per_second": (
                            ComparisonEngine
                            ._safe_difference(
                                first[
                                    "tokens_per_second"
                                ],
                                second[
                                    "tokens_per_second"
                                ]
                            )
                        ),

                        # ---------------------------------
                        # Traditional evaluation
                        # ---------------------------------

                        "exact_match": (
                            ComparisonEngine
                            ._safe_difference(
                                first[
                                    "exact_match"
                                ],
                                second[
                                    "exact_match"
                                ]
                            )
                        ),

                        "fuzzy_match": (
                            ComparisonEngine
                            ._safe_difference(
                                first[
                                    "fuzzy_match"
                                ],
                                second[
                                    "fuzzy_match"
                                ]
                            )
                        ),

                        "average_fuzzy_similarity": (
                            ComparisonEngine
                            ._safe_difference(
                                first[
                                    "average_fuzzy_similarity"
                                ],
                                second[
                                    "average_fuzzy_similarity"
                                ]
                            )
                        ),

                        "keyword_match": (
                            ComparisonEngine
                            ._safe_difference(
                                first[
                                    "keyword_match"
                                ],
                                second[
                                    "keyword_match"
                                ]
                            )
                        ),

                        "semantic_match": (
                            ComparisonEngine
                            ._safe_difference(
                                first[
                                    "semantic_match"
                                ],
                                second[
                                    "semantic_match"
                                ]
                            )
                        ),

                        "semantic_similarity": (
                            ComparisonEngine
                            ._safe_difference(
                                first[
                                    "average_semantic_similarity"
                                ],
                                second[
                                    "average_semantic_similarity"
                                ]
                            )
                        ),

                        # ---------------------------------
                        # LLM Judge
                        # ---------------------------------

                        "judge_success_rate": (
                            ComparisonEngine
                            ._safe_difference(
                                first[
                                    "judge_success_rate"
                                ],
                                second[
                                    "judge_success_rate"
                                ]
                            )
                        ),

                        "judge_correctness": (
                            ComparisonEngine
                            ._safe_difference(
                                first[
                                    "judge_correctness"
                                ],
                                second[
                                    "judge_correctness"
                                ]
                            )
                        ),

                        "judge_relevance": (
                            ComparisonEngine
                            ._safe_difference(
                                first[
                                    "judge_relevance"
                                ],
                                second[
                                    "judge_relevance"
                                ]
                            )
                        ),

                        "judge_reasoning_quality": (
                            ComparisonEngine
                            ._safe_difference(
                                first[
                                    "judge_reasoning_quality"
                                ],
                                second[
                                    "judge_reasoning_quality"
                                ]
                            )
                        ),

                        "judge_confidence": (
                            ComparisonEngine
                            ._safe_difference(
                                first[
                                    "judge_confidence"
                                ],
                                second[
                                    "judge_confidence"
                                ]
                            )
                        ),

                        # ---------------------------------
                        # Cost
                        # ---------------------------------

                        "input_cost_usd": (
                            ComparisonEngine
                            ._safe_difference(
                                first[
                                    "input_cost_usd"
                                ],
                                second[
                                    "input_cost_usd"
                                ]
                            )
                        ),

                        "output_cost_usd": (
                            ComparisonEngine
                            ._safe_difference(
                                first[
                                    "output_cost_usd"
                                ],
                                second[
                                    "output_cost_usd"
                                ]
                            )
                        ),

                        "total_cost_usd": (
                            ComparisonEngine
                            ._safe_difference(
                                first[
                                    "total_cost_usd"
                                ],
                                second[
                                    "total_cost_usd"
                                ]
                            )
                        )
                    }
                })

        return {

            "experiment_count": len(
                extracted
            ),

            "ignored_experiments": (
                ignored_experiments
            ),

            "experiments": extracted,

            "comparisons": comparisons
        }

    @staticmethod
    def compare_two(
        first_experiment: dict[str, Any],
        second_experiment: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Compare exactly two experiments.
        """

        result = ComparisonEngine.compare([
            first_experiment,
            second_experiment
        ])

        if not result[
            "comparisons"
        ]:
            return {}

        return result[
            "comparisons"
        ][0]