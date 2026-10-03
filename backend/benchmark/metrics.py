from statistics import median
from typing import Any


class BenchmarkMetrics:
    """
    Calculate benchmark-level metrics for LLM experiments.

    Supported metrics:
    - Basic accuracy and error rate
    - Latency
    - Token usage and throughput
    - Exact / fuzzy / keyword evaluation
    - Semantic evaluation
    - LLM-as-a-Judge
    - Cost
    - Retry / fallback reliability
    - Category-level performance
    """

    # =========================================================
    # UTILITY METHODS
    # =========================================================

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """Safely convert a value to float."""

        try:
            if value is None:
                return default

            return float(value)

        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        """Safely convert a value to int."""

        try:
            if value is None:
                return default

            return int(value)

        except (TypeError, ValueError):
            return default

    @staticmethod
    def _average(values: list[float]) -> float:
        """Calculate average safely."""

        if not values:
            return 0.0

        return sum(values) / len(values)

    @staticmethod
    def _percentile(
        values: list[float],
        percentile: float,
    ) -> float:
        """
        Calculate percentile using linear interpolation.
        """

        if not values:
            return 0.0

        sorted_values = sorted(values)

        if len(sorted_values) == 1:
            return sorted_values[0]

        position = (
            len(sorted_values) - 1
        ) * percentile

        lower_index = int(position)

        upper_index = min(
            lower_index + 1,
            len(sorted_values) - 1,
        )

        fraction = (
            position - lower_index
        )

        return (
            sorted_values[lower_index]
            + (
                sorted_values[upper_index]
                - sorted_values[lower_index]
            )
            * fraction
        )

    # =========================================================
    # BASIC METRICS
    # =========================================================

    @staticmethod
    def calculate_basic_metrics(
        results: list[dict[str, Any]],
    ) -> dict[str, Any]:

        total = len(results)

        successful_results = [
            result
            for result in results
            if result.get("status") == "success"
        ]

        failed_results = [
            result
            for result in results
            if result.get("status") != "success"
        ]

        successful = len(
            successful_results
        )

        failed = len(
            failed_results
        )

        correct = 0

        for result in successful_results:

            evaluation = result.get(
                "evaluation",
                {},
            )

            if (
                evaluation.get(
                    "exact_match"
                )
                == 1.0
            ):
                correct += 1

        accuracy = (
            correct / successful
            if successful > 0
            else 0.0
        )

        error_rate = (
            failed / total
            if total > 0
            else 0.0
        )

        return {
            "total": total,
            "successful": successful,
            "failed": failed,
            "error_rate": round(
                error_rate,
                4,
            ),
            "correct": correct,
            "accuracy": round(
                accuracy,
                4,
            ),
        }

    # =========================================================
    # LATENCY METRICS
    # =========================================================

    @staticmethod
    def calculate_latency_metrics(
        results: list[dict[str, Any]],
    ) -> dict[str, float]:

        latency_values: list[float] = []
        provider_latency_values: list[float] = []

        for result in results:

            if result.get("status") != "success":
                continue

            latency = result.get(
                "latency"
            )

            if latency is not None:

                latency_values.append(
                    BenchmarkMetrics._safe_float(
                        latency
                    )
                )

            provider_latency = result.get(
                "provider_latency"
            )

            if provider_latency is not None:

                provider_latency_values.append(
                    BenchmarkMetrics._safe_float(
                        provider_latency
                    )
                )

        if latency_values:

            average_latency = (
                BenchmarkMetrics._average(
                    latency_values
                )
            )

            p50_latency = median(
                latency_values
            )

            p95_latency = (
                BenchmarkMetrics._percentile(
                    latency_values,
                    0.95,
                )
            )

        else:

            average_latency = 0.0
            p50_latency = 0.0
            p95_latency = 0.0

        average_provider_latency = (
            BenchmarkMetrics._average(
                provider_latency_values
            )
            if provider_latency_values
            else 0.0
        )

        return {
            "average_latency": round(
                average_latency,
                4,
            ),
            "p50_latency": round(
                p50_latency,
                4,
            ),
            "p95_latency": round(
                p95_latency,
                4,
            ),
            "average_provider_latency": round(
                average_provider_latency,
                4,
            ),
        }

    # =========================================================
    # TOKEN METRICS
    # =========================================================

    @staticmethod
    def calculate_token_metrics(
        results: list[dict[str, Any]],
    ) -> dict[str, Any]:

        input_tokens = 0
        output_tokens = 0
        total_tokens = 0

        latency_values: list[float] = []

        for result in results:

            if result.get("status") != "success":
                continue

            tokens = result.get(
                "tokens",
                {},
            )

            if not isinstance(tokens, dict):
                tokens = {}

            input_tokens += (
                BenchmarkMetrics._safe_int(
                    tokens.get("input", 0)
                )
            )

            output_tokens += (
                BenchmarkMetrics._safe_int(
                    tokens.get("output", 0)
                )
            )

            total_tokens += (
                BenchmarkMetrics._safe_int(
                    tokens.get("total", 0)
                )
            )

            latency = result.get(
                "latency"
            )

            if latency is not None:

                latency_values.append(
                    BenchmarkMetrics._safe_float(
                        latency
                    )
                )

        total_latency = sum(
            latency_values
        )

        tokens_per_second = (
            total_tokens / total_latency
            if total_latency > 0
            else 0.0
        )

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "tokens_per_second": round(
                tokens_per_second,
                4,
            ),
        }

    # =========================================================
    # EVALUATION METRICS
    # =========================================================

    @staticmethod
    def calculate_evaluation_metrics(
        results: list[dict[str, Any]],
    ) -> dict[str, float]:

        successful_results = [
            result
            for result in results
            if result.get("status") == "success"
        ]

        exact_values: list[float] = []
        fuzzy_values: list[float] = []
        fuzzy_similarity_values: list[float] = []
        keyword_values: list[float] = []
        semantic_match_values: list[float] = []
        semantic_similarity_values: list[float] = []

        for result in successful_results:

            evaluation = result.get(
                "evaluation",
                {},
            )

            if not isinstance(
                evaluation,
                dict,
            ):
                evaluation = {}

            # Exact match
            exact_match = evaluation.get(
                "exact_match"
            )

            if exact_match is not None:

                exact_values.append(
                    BenchmarkMetrics._safe_float(
                        exact_match
                    )
                )

            # Fuzzy match
            fuzzy_match = evaluation.get(
                "fuzzy_match"
            )

            if fuzzy_match is not None:

                fuzzy_values.append(
                    BenchmarkMetrics._safe_float(
                        fuzzy_match
                    )
                )

            # Fuzzy similarity
            fuzzy_similarity = evaluation.get(
                "fuzzy_similarity"
            )

            if fuzzy_similarity is not None:

                fuzzy_similarity_values.append(
                    BenchmarkMetrics._safe_float(
                        fuzzy_similarity
                    )
                )

            # Keyword match
            keyword_match = evaluation.get(
                "keyword_match"
            )

            if keyword_match is not None:

                keyword_values.append(
                    BenchmarkMetrics._safe_float(
                        keyword_match
                    )
                )

            # Semantic evaluation
            semantic = result.get(
                "semantic_evaluation"
            )

            if not isinstance(
                semantic,
                dict,
            ):
                continue

            semantic_match = semantic.get(
                "semantic_match"
            )

            if semantic_match is not None:

                semantic_match_values.append(
                    BenchmarkMetrics._safe_float(
                        semantic_match
                    )
                )

            semantic_similarity = semantic.get(
                "semantic_similarity"
            )

            if semantic_similarity is not None:

                semantic_similarity_values.append(
                    BenchmarkMetrics._safe_float(
                        semantic_similarity
                    )
                )

        return {
            "exact_match": round(
                BenchmarkMetrics._average(
                    exact_values
                ),
                4,
            ),
            "fuzzy_match": round(
                BenchmarkMetrics._average(
                    fuzzy_values
                ),
                4,
            ),
            "average_fuzzy_similarity": round(
                BenchmarkMetrics._average(
                    fuzzy_similarity_values
                ),
                4,
            ),
            "keyword_match": round(
                BenchmarkMetrics._average(
                    keyword_values
                ),
                4,
            ),
            "semantic_match": round(
                BenchmarkMetrics._average(
                    semantic_match_values
                ),
                4,
            ),
            "average_semantic_similarity": round(
                BenchmarkMetrics._average(
                    semantic_similarity_values
                ),
                4,
            ),
        }

    # =========================================================
    # LLM-AS-A-JUDGE METRICS
    # =========================================================

    @staticmethod
    def calculate_judge_metrics(
        results: list[dict[str, Any]],
    ) -> dict[str, Any]:

        judge_results: list[
            dict[str, Any]
        ] = []

        successful_results = 0

        for result in results:

            if result.get(
                "status"
            ) != "success":

                continue

            successful_results += 1

            judge = result.get(
                "llm_judge"
            )

            if not isinstance(
                judge,
                dict,
            ):
                continue

            if judge.get(
                "status"
            ) == "error":

                continue

            if judge.get(
                "correctness"
            ) is None:

                continue

            judge_results.append(
                judge
            )

        if not judge_results:

            return {
                "judge_enabled": False,
                "judge_success_rate": 0.0,
                "average_correctness": 0.0,
                "average_relevance": 0.0,
                "average_reasoning_quality": 0.0,
                "average_confidence": 0.0,
            }

        def average_judge_score(
            field: str,
        ) -> float:

            values: list[float] = []

            for judge in judge_results:

                value = judge.get(
                    field
                )

                if value is None:
                    continue

                values.append(
                    BenchmarkMetrics._safe_float(
                        value
                    )
                )

            return BenchmarkMetrics._average(
                values
            )

        judge_success_rate = (
            len(judge_results)
            / successful_results
            if successful_results > 0
            else 0.0
        )

        return {
            "judge_enabled": True,
            "judge_success_rate": round(
                judge_success_rate,
                4,
            ),
            "average_correctness": round(
                average_judge_score(
                    "correctness"
                ),
                4,
            ),
            "average_relevance": round(
                average_judge_score(
                    "relevance"
                ),
                4,
            ),
            "average_reasoning_quality": round(
                average_judge_score(
                    "reasoning_quality"
                ),
                4,
            ),
            "average_confidence": round(
                average_judge_score(
                    "confidence"
                ),
                4,
            ),
        }

    # =========================================================
    # COST METRICS
    # =========================================================

    @staticmethod
    def calculate_cost_metrics(
        results: list[dict[str, Any]],
    ) -> dict[str, float]:

        input_cost = 0.0
        output_cost = 0.0
        total_cost = 0.0

        for result in results:

            cost = result.get(
                "cost",
                {},
            )

            if not isinstance(
                cost,
                dict,
            ):
                continue

            input_cost += (
                BenchmarkMetrics._safe_float(
                    cost.get(
                        "input_cost_usd",
                        0.0,
                    )
                )
            )

            output_cost += (
                BenchmarkMetrics._safe_float(
                    cost.get(
                        "output_cost_usd",
                        0.0,
                    )
                )
            )

            total_cost += (
                BenchmarkMetrics._safe_float(
                    cost.get(
                        "total_cost_usd",
                        0.0,
                    )
                )
            )

        return {
            "input_cost_usd": round(
                input_cost,
                8,
            ),
            "output_cost_usd": round(
                output_cost,
                8,
            ),
            "total_cost_usd": round(
                total_cost,
                8,
            ),
        }

    # =========================================================
    # RETRY + FALLBACK METRICS
    # =========================================================

    @staticmethod
    def calculate_retry_metrics(
        results: list[dict[str, Any]],
    ) -> dict[str, Any]:

        total = len(results)

        if total == 0:

            return {
                "retry_rate": 0.0,
                "fallback_rate": 0.0,
                "fallback_success_rate": 0.0,
                "average_attempts": 0.0,
                "average_retries": 0.0,
                "total_retries": 0,
                "total_attempts": 0,
            }

        retry_count_results = 0
        fallback_count = 0
        fallback_success_count = 0

        total_attempts = 0
        total_retries = 0

        results_with_retry_data = 0

        for result in results:

            retry_info = result.get(
                "retry"
            )

            if not isinstance(
                retry_info,
                dict,
            ):
                continue

            results_with_retry_data += 1

            attempts = BenchmarkMetrics._safe_int(
                retry_info.get(
                    "attempts",
                    0,
                )
            )

            retries = BenchmarkMetrics._safe_int(
                retry_info.get(
                    "retries",
                    0,
                )
            )

            fallback_used = bool(
                retry_info.get(
                    "fallback_used",
                    False,
                )
            )

            total_attempts += attempts
            total_retries += retries

            if retries > 0:
                retry_count_results += 1

            if fallback_used:

                fallback_count += 1

                if result.get(
                    "status"
                ) == "success":

                    fallback_success_count += 1

        retry_rate = (
            retry_count_results
            / total
            if total > 0
            else 0.0
        )

        fallback_rate = (
            fallback_count
            / total
            if total > 0
            else 0.0
        )

        fallback_success_rate = (
            fallback_success_count
            / fallback_count
            if fallback_count > 0
            else 0.0
        )

        average_attempts = (
            total_attempts
            / results_with_retry_data
            if results_with_retry_data > 0
            else 0.0
        )

        average_retries = (
            total_retries
            / results_with_retry_data
            if results_with_retry_data > 0
            else 0.0
        )

        return {
            "retry_rate": round(
                retry_rate,
                4,
            ),
            "fallback_rate": round(
                fallback_rate,
                4,
            ),
            "fallback_success_rate": round(
                fallback_success_rate,
                4,
            ),
            "average_attempts": round(
                average_attempts,
                4,
            ),
            "average_retries": round(
                average_retries,
                4,
            ),
            "total_retries": total_retries,
            "total_attempts": total_attempts,
        }

    # =========================================================
    # CATEGORY METRICS
    # =========================================================

    @staticmethod
    def calculate_category_metrics(
        results: list[dict[str, Any]],
    ) -> dict[str, Any]:

        categories: dict[
            str,
            dict[str, Any],
        ] = {}

        for result in results:

            category = result.get(
                "category",
                "unknown",
            )

            if category not in categories:

                categories[category] = {
                    "total": 0,
                    "successful": 0,
                    "failed": 0,
                    "correct": 0,
                    "accuracy": 0.0,
                }

            category_metrics = categories[
                category
            ]

            category_metrics[
                "total"
            ] += 1

            if result.get(
                "status"
            ) == "success":

                category_metrics[
                    "successful"
                ] += 1

                evaluation = result.get(
                    "evaluation",
                    {},
                )

                if (
                    isinstance(
                        evaluation,
                        dict,
                    )
                    and evaluation.get(
                        "exact_match"
                    )
                    == 1.0
                ):

                    category_metrics[
                        "correct"
                    ] += 1

            else:

                category_metrics[
                    "failed"
                ] += 1

        for metrics in categories.values():

            successful = metrics[
                "successful"
            ]

            metrics[
                "accuracy"
            ] = round(
                (
                    metrics["correct"]
                    / successful
                )
                if successful > 0
                else 0.0,
                4,
            )

        return categories

    # =========================================================
    # MAIN CALCULATION
    # =========================================================

    @staticmethod
    def calculate(
        results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Calculate all benchmark metrics.
        """

        basic = (
            BenchmarkMetrics.calculate_basic_metrics(
                results
            )
        )

        latency = (
            BenchmarkMetrics.calculate_latency_metrics(
                results
            )
        )

        tokens = (
            BenchmarkMetrics.calculate_token_metrics(
                results
            )
        )

        evaluation = (
            BenchmarkMetrics.calculate_evaluation_metrics(
                results
            )
        )

        judge = (
            BenchmarkMetrics.calculate_judge_metrics(
                results
            )
        )

        cost = (
            BenchmarkMetrics.calculate_cost_metrics(
                results
            )
        )

        retry = (
            BenchmarkMetrics.calculate_retry_metrics(
                results
            )
        )

        categories = (
            BenchmarkMetrics.calculate_category_metrics(
                results
            )
        )

        return {

            # -------------------------------------------------
            # BASIC
            # -------------------------------------------------

            "total": basic["total"],
            "successful": basic["successful"],
            "failed": basic["failed"],
            "error_rate": basic["error_rate"],
            "correct": basic["correct"],
            "accuracy": basic["accuracy"],

            # -------------------------------------------------
            # LATENCY
            # -------------------------------------------------

            "average_latency": latency[
                "average_latency"
            ],
            "p50_latency": latency[
                "p50_latency"
            ],
            "p95_latency": latency[
                "p95_latency"
            ],
            "average_provider_latency": latency[
                "average_provider_latency"
            ],

            # -------------------------------------------------
            # TOKENS
            # -------------------------------------------------

            "input_tokens": tokens[
                "input_tokens"
            ],
            "output_tokens": tokens[
                "output_tokens"
            ],
            "total_tokens": tokens[
                "total_tokens"
            ],
            "tokens_per_second": tokens[
                "tokens_per_second"
            ],

            # -------------------------------------------------
            # EVALUATION
            # -------------------------------------------------

            "exact_match": evaluation[
                "exact_match"
            ],
            "fuzzy_match": evaluation[
                "fuzzy_match"
            ],
            "average_fuzzy_similarity": evaluation[
                "average_fuzzy_similarity"
            ],
            "keyword_match": evaluation[
                "keyword_match"
            ],
            "semantic_match": evaluation[
                "semantic_match"
            ],
            "average_semantic_similarity": evaluation[
                "average_semantic_similarity"
            ],

            # -------------------------------------------------
            # LLM JUDGE
            # -------------------------------------------------

            "judge": judge,

            # -------------------------------------------------
            # COST
            # -------------------------------------------------

            "cost": cost,

            # -------------------------------------------------
            # RETRY / FALLBACK
            # -------------------------------------------------

            "retry": retry,

            # -------------------------------------------------
            # CATEGORY
            # -------------------------------------------------

            "categories": categories,
        }
    