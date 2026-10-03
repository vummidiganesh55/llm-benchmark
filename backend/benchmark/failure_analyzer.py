from typing import Any


class FailureAnalyzer:

    FAILURE_TYPES = {
        "provider_error",
        "timeout",
        "rate_limit",
        "evaluation_failure",
        "factual_error",
        "reasoning_error",
        "coding_error",
        "formatting_error",
        "instruction_following_error",
        "unknown"
    }

    @staticmethod
    def _classify_provider_error(
        error_type: str | None,
        error_message: str | None
    ) -> str:

        error_type = (
            error_type or ""
        ).lower()

        error_message = (
            error_message or ""
        ).lower()

        # -----------------------------------------
        # Timeout
        # -----------------------------------------

        timeout_keywords = [
            "timeout",
            "timed out",
            "time out",
            "read timeout",
            "connect timeout"
        ]

        if (
            "timeout" in error_type
            or any(
                keyword in error_message
                for keyword in timeout_keywords
            )
        ):
            return "timeout"

        # -----------------------------------------
        # Rate limit
        # -----------------------------------------

        rate_limit_keywords = [
            "429",
            "rate limit",
            "rate_limit",
            "resource_exhausted",
            "too many requests",
            "quota exceeded",
            "exceeded your quota",
        ]

        if any(
            keyword in error_message
            for keyword in rate_limit_keywords
        ):
            return "rate_limit"

        # -----------------------------------------
        # Other provider errors
        # -----------------------------------------

        return "provider_error"

    @staticmethod
    def _classify_evaluation_failure(
        result: dict[str, Any]
    ) -> str:

        evaluation = result.get(
            "evaluation",
            {}
        )

        category = (
            result.get(
                "category",
                ""
            )
            .lower()
            .strip()
        )

        exact_match = evaluation.get(
            "exact_match",
            0.0
        )

        fuzzy_match = evaluation.get(
            "fuzzy_match",
            0.0
        )

        keyword_match = evaluation.get(
            "keyword_match",
            0.0
        )

        semantic_match = evaluation.get(
            "semantic_match",
            0.0
        )

        # -----------------------------------------
        # Correct response
        # -----------------------------------------

        if exact_match == 1.0:
            return None

        # -----------------------------------------
        # Coding failure
        # -----------------------------------------

        if category == "coding":
            return "coding_error"

        # -----------------------------------------
        # Reasoning failure
        # -----------------------------------------

        if category == "reasoning":
            return "reasoning_error"

        # -----------------------------------------
        # Formatting failure
        # -----------------------------------------

        if (
            fuzzy_match == 1.0
            and exact_match == 0.0
        ):
            return "formatting_error"

        # -----------------------------------------
        # Semantic/keyword disagreement
        # -----------------------------------------

        if (
            keyword_match > 0.0
            or semantic_match == 1.0
        ):
            return "factual_error"

        # -----------------------------------------
        # Default model-answer failure
        # -----------------------------------------

        return "factual_error"

    @staticmethod
    def analyze(
        result: dict[str, Any]
    ) -> dict[str, Any]:

        status = (
            result.get(
                "status",
                "unknown"
            )
        )

        # -----------------------------------------
        # Provider/request failure
        # -----------------------------------------

        if status == "error":

            failure_type = (
                FailureAnalyzer
                ._classify_provider_error(
                    error_type=result.get(
                        "error_type"
                    ),
                    error_message=result.get(
                        "error_message"
                    )
                )
            )

            return {
                "is_failure": True,
                "failure_type": failure_type,
                "category": result.get(
                    "category"
                ),
                "error_type": result.get(
                    "error_type"
                ),
                "error_message": result.get(
                    "error_message"
                ),
                "analysis_source": "provider"
            }

        # -----------------------------------------
        # Successful request but incorrect answer
        # -----------------------------------------

        if status == "success":

            failure_type = (
                FailureAnalyzer
                ._classify_evaluation_failure(
                    result
                )
            )

            if failure_type is None:

                return {
                    "is_failure": False,
                    "failure_type": None,
                    "category": result.get(
                        "category"
                    ),
                    "error_type": None,
                    "error_message": None,
                    "analysis_source": "evaluation"
                }

            return {
                "is_failure": True,
                "failure_type": failure_type,
                "category": result.get(
                    "category"
                ),
                "error_type": None,
                "error_message": None,
                "analysis_source": "evaluation"
            }

        # -----------------------------------------
        # Unknown result state
        # -----------------------------------------

        return {
            "is_failure": True,
            "failure_type": "unknown",
            "category": result.get(
                "category"
            ),
            "error_type": result.get(
                "error_type"
            ),
            "error_message": result.get(
                "error_message"
            ),
            "analysis_source": "unknown"
        }

    @staticmethod
    def analyze_results(
        results: list[dict[str, Any]]
    ) -> dict[str, Any]:

        analyzed_results = []

        failure_counts = {}

        category_failures = {}

        total_failures = 0

        for result in results:

            analysis = FailureAnalyzer.analyze(
                result
            )

            analyzed_results.append({
                **result,
                "failure_analysis": analysis
            })

            if analysis["is_failure"]:

                total_failures += 1

                failure_type = analysis[
                    "failure_type"
                ]

                failure_counts[
                    failure_type
                ] = (
                    failure_counts.get(
                        failure_type,
                        0
                    ) + 1
                )

                category = analysis.get(
                    "category",
                    "unknown"
                )

                if category not in category_failures:
                    category_failures[
                        category
                    ] = {}

                category_failures[
                    category
                ][failure_type] = (
                    category_failures[
                        category
                    ].get(
                        failure_type,
                        0
                    ) + 1
                )

        total_results = len(
            results
        )

        failure_rate = (
            total_failures / total_results
            if total_results
            else 0.0
        )

        return {
            "total_results": total_results,

            "total_failures": total_failures,

            "failure_rate": round(
                failure_rate,
                4
            ),

            "failure_counts": failure_counts,

            "category_failures": category_failures,

            "results": analyzed_results
        }