from typing import Any

from backend.benchmark.retry_handler import RetryHandler
from backend.benchmark.evaluator import Evaluator
from backend.benchmark.config import BenchmarkConfig

from backend.observability.logger import get_logger


class BenchmarkRunner:
    def __init__(
        self,
        provider,
        semantic_evaluator=None,
        llm_judge=None,
        fallback_provider=None,
        retry_handler=None,
        config: BenchmarkConfig | None = None,
    ):
        self.provider = provider
        self.semantic_evaluator = semantic_evaluator
        self.llm_judge = llm_judge
        self.fallback_provider = fallback_provider
        self.config = config

        self.retry_handler = retry_handler or RetryHandler(
            max_retries=2,
            retry_delay=1.0,
            backoff_factor=2.0,
        )

        self.logger = get_logger(__name__)

        self.logger.info(
            "BenchmarkRunner initialized | "
            "provider=%s | fallback=%s | config=%s",
            type(self.provider).__name__,
            type(self.fallback_provider).__name__
            if self.fallback_provider is not None
            else None,
            self.config.to_dict() if self.config else None,
        )

    # ---------------------------------------------------------
    # Provider execution
    # ---------------------------------------------------------

    def _generate_with_provider(
        self,
        provider,
        prompt: str,
    ) -> dict[str, Any]:

        provider_name = type(provider).__name__

        self.logger.info(
            "Provider request started | provider=%s",
            provider_name,
        )

        try:
            output = provider.generate(prompt)

            self.logger.info(
                "Provider request completed | provider=%s",
                provider_name,
            )

            return output

        except Exception as error:
            self.logger.error(
                "Provider request failed | provider=%s | "
                "error_type=%s | error=%s",
                provider_name,
                type(error).__name__,
                str(error),
            )
            raise

    # ---------------------------------------------------------
    # Retry + fallback
    # ---------------------------------------------------------

    def _generate_with_retry_and_fallback(
        self,
        prompt: str,
    ) -> dict[str, Any]:

        self.logger.info(
            "Primary provider execution started"
        )

        primary_result = self.retry_handler.execute(
            lambda: self._generate_with_provider(
                self.provider,
                prompt,
            )
        )

        # -----------------------------------------------------
        # Primary provider succeeded
        # -----------------------------------------------------

        if primary_result["success"]:

            self.logger.info(
                "Primary provider succeeded | attempts=%s | retries=%s",
                primary_result["attempts"],
                primary_result["retries"],
            )

            return {
                "success": True,
                "output": primary_result["result"],
                "retry": {
                    "attempts": primary_result["attempts"],
                    "retries": primary_result["retries"],
                    "fallback_used": False,
                    "source": "primary",
                    "primary_errors": primary_result["errors"],
                    "fallback_errors": [],
                },
            }

        self.logger.warning(
            "Primary provider exhausted | attempts=%s | retries=%s",
            primary_result["attempts"],
            primary_result["retries"],
        )

        # -----------------------------------------------------
        # No fallback provider configured
        # -----------------------------------------------------

        if self.fallback_provider is None:

            self.logger.warning(
                "No fallback provider configured"
            )

            return {
                "success": False,
                "output": None,
                "retry": {
                    "attempts": primary_result["attempts"],
                    "retries": primary_result["retries"],
                    "fallback_used": False,
                    "source": "primary",
                    "primary_errors": primary_result["errors"],
                    "fallback_errors": [],
                },
            }

        # -----------------------------------------------------
        # Primary exhausted → fallback provider
        # -----------------------------------------------------

        self.logger.info(
            "Fallback provider execution started"
        )

        fallback_result = self.retry_handler.execute(
            lambda: self._generate_with_provider(
                self.fallback_provider,
                prompt,
            )
        )

        total_attempts = (
            primary_result["attempts"]
            + fallback_result["attempts"]
        )

        total_retries = (
            primary_result["retries"]
            + fallback_result["retries"]
        )

        # -----------------------------------------------------
        # Fallback succeeded
        # -----------------------------------------------------

        if fallback_result["success"]:

            self.logger.info(
                "Fallback provider succeeded | "
                "total_attempts=%s | total_retries=%s",
                total_attempts,
                total_retries,
            )

            return {
                "success": True,
                "output": fallback_result["result"],
                "retry": {
                    "attempts": total_attempts,
                    "retries": total_retries,
                    "fallback_used": True,
                    "source": "fallback",
                    "primary_errors": primary_result["errors"],
                    "fallback_errors": fallback_result["errors"],
                },
            }

        # -----------------------------------------------------
        # Both providers failed
        # -----------------------------------------------------

        self.logger.error(
            "Primary and fallback providers failed | "
            "total_attempts=%s | total_retries=%s",
            total_attempts,
            total_retries,
        )

        return {
            "success": False,
            "output": None,
            "retry": {
                "attempts": total_attempts,
                "retries": total_retries,
                "fallback_used": True,
                "source": "fallback",
                "primary_errors": primary_result["errors"],
                "fallback_errors": fallback_result["errors"],
            },
        }

    # ---------------------------------------------------------
    # Semantic evaluation
    # ---------------------------------------------------------

    def _evaluate_semantic(
        self,
        response: str,
        expected: str,
    ) -> dict[str, Any] | None:

        if self.semantic_evaluator is None:
            return None

        try:

            self.logger.info(
                "Semantic evaluation started"
            )

            result = self.semantic_evaluator.evaluate(
                response=response,
                expected=expected,
            )

            self.logger.info(
                "Semantic evaluation completed"
            )

            return result

        except Exception as error:

            self.logger.error(
                "Semantic evaluation failed | "
                "error_type=%s | error=%s",
                type(error).__name__,
                str(error),
            )

            return {
                "status": "error",
                "error_type": type(error).__name__,
                "error_message": str(error),
            }

    # ---------------------------------------------------------
    # LLM-as-a-Judge evaluation
    # ---------------------------------------------------------

    def _evaluate_judge(
        self,
        question: str,
        expected: str,
        response: str,
    ) -> dict[str, Any] | None:

        if self.llm_judge is None:
            return None

        try:

            self.logger.info(
                "LLM judge evaluation started"
            )

            result = self.llm_judge.evaluate(
                question=question,
                expected=expected,
                response=response,
            )

            self.logger.info(
                "LLM judge evaluation completed"
            )

            return result

        except Exception as error:

            self.logger.error(
                "LLM judge evaluation failed | "
                "error_type=%s | error=%s",
                type(error).__name__,
                str(error),
            )

            return {
                "status": "error",
                "error_type": type(error).__name__,
                "error_message": str(error),
            }

    # ---------------------------------------------------------
    # Benchmark execution
    # ---------------------------------------------------------

    def run(
        self,
        questions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        results = []

        # -----------------------------------------------------
        # Apply configuration question limit
        #
        # Important:
        # We do NOT modify the original questions list.
        # -----------------------------------------------------

        questions_to_run = questions

        if (
            self.config is not None
            and self.config.question_limit is not None
        ):
            questions_to_run = questions[
                : self.config.question_limit
            ]

        self.logger.info(
            "Benchmark started | total_questions=%s | "
            "questions_to_run=%s",
            len(questions),
            len(questions_to_run),
        )

        # -----------------------------------------------------
        # Configuration metadata
        # -----------------------------------------------------

        config_metadata = (
            self.config.to_dict()
            if self.config is not None
            else None
        )

        # -----------------------------------------------------
        # Process questions
        # -----------------------------------------------------

        for question in questions_to_run:

            prompt = question["prompt"]
            expected = question["expected"]

            self.logger.info(
                "Benchmark question started | id=%s | category=%s",
                question.get("id"),
                question.get("category"),
            )

            result_base = {
                "id": question.get("id"),
                "category": question.get("category"),
                "prompt": prompt,
                "expected": expected,
                "config": config_metadata,
            }

            try:

                # -------------------------------------------------
                # Generate response with retry + fallback
                # -------------------------------------------------

                execution = (
                    self._generate_with_retry_and_fallback(
                        prompt
                    )
                )

                retry_info = execution["retry"]

                # -------------------------------------------------
                # Provider failure
                # -------------------------------------------------

                if not execution["success"]:

                    primary_errors = retry_info.get(
                        "primary_errors",
                        [],
                    )

                    fallback_errors = retry_info.get(
                        "fallback_errors",
                        [],
                    )

                    all_errors = (
                        primary_errors
                        + fallback_errors
                    )

                    last_error = (
                        all_errors[-1]
                        if all_errors
                        else {}
                    )

                    result = {
                        **result_base,
                        "status": "error",
                        "response": None,
                        "evaluation": None,
                        "semantic_evaluation": None,
                        "llm_judge": None,
                        "latency": None,
                        "provider_latency": None,
                        "tokens": {
                            "input": 0,
                            "output": 0,
                            "total": 0,
                        },
                        "cost": {
                            "input_cost_usd": 0.0,
                            "output_cost_usd": 0.0,
                            "total_cost_usd": 0.0,
                        },
                        "retry": retry_info,
                        "error": {
                            "type": last_error.get(
                                "error_type",
                                "ProviderError",
                            ),
                            "message": last_error.get(
                                "error_message",
                                "Provider request failed.",
                            ),
                        },
                    }

                    results.append(result)

                    self.logger.error(
                        "Benchmark question failed | id=%s | "
                        "error_type=%s",
                        question.get("id"),
                        result["error"]["type"],
                    )

                    continue

                # -------------------------------------------------
                # Successful provider execution
                # -------------------------------------------------

                output = execution["output"]

                response = output.get(
                    "response",
                    "",
                )

                # -------------------------------------------------
                # Deterministic evaluation
                # -------------------------------------------------

                evaluation = Evaluator.evaluate(
                    response=response,
                    expected=expected,
                )

                # -------------------------------------------------
                # Semantic evaluation
                # -------------------------------------------------

                semantic_evaluation = (
                    self._evaluate_semantic(
                        response=response,
                        expected=expected,
                    )
                )

                # -------------------------------------------------
                # LLM-as-a-Judge
                # -------------------------------------------------

                llm_judge_result = self._evaluate_judge(
                    question=prompt,
                    expected=expected,
                    response=response,
                )

                # -------------------------------------------------
                # Latency normalization
                #
                # Some providers return `provider_latency`
                # but do not return a separate `latency`.
                #
                # Metrics.py expects `latency`, so use
                # provider_latency as the fallback.
                # -------------------------------------------------

                provider_latency = output.get(
                    "provider_latency"
                )

                latency = output.get(
                    "latency"
                )

                if latency is None:
                    latency = provider_latency

                # -------------------------------------------------
                # Build successful result
                # -------------------------------------------------

                result = {
                    **result_base,
                    "status": "success",
                    "response": response,
                    "evaluation": evaluation,
                    "semantic_evaluation": semantic_evaluation,
                    "llm_judge": llm_judge_result,
                    "latency": latency,
                    "provider_latency": provider_latency,
                    "tokens": {
                        "input": output.get(
                            "input_tokens",
                            0,
                        ),
                        "output": output.get(
                            "output_tokens",
                            0,
                        ),
                        "total": output.get(
                            "total_tokens",
                            0,
                        ),
                    },
                    "cost": output.get(
                        "cost",
                        {
                            "input_cost_usd": 0.0,
                            "output_cost_usd": 0.0,
                            "total_cost_usd": 0.0,
                        },
                    ),
                    "retry": retry_info,
                }

                results.append(result)

                self.logger.info(
                    "Benchmark question completed | "
                    "id=%s | status=success | "
                    "exact_match=%s | latency=%s",
                    question.get("id"),
                    evaluation.get("exact_match"),
                    latency,
                )

            # -----------------------------------------------------
            # Unexpected question-level failure
            # -----------------------------------------------------

            except Exception as error:

                self.logger.error(
                    "Benchmark question crashed | "
                    "id=%s | error_type=%s | error=%s",
                    question.get("id"),
                    type(error).__name__,
                    str(error),
                )

                results.append(
                    {
                        **result_base,
                        "status": "error",
                        "response": None,
                        "evaluation": None,
                        "semantic_evaluation": None,
                        "llm_judge": None,
                        "latency": None,
                        "provider_latency": None,
                        "tokens": {
                            "input": 0,
                            "output": 0,
                            "total": 0,
                        },
                        "cost": {
                            "input_cost_usd": 0.0,
                            "output_cost_usd": 0.0,
                            "total_cost_usd": 0.0,
                        },
                        "retry": {
                            "attempts": 0,
                            "retries": 0,
                            "fallback_used": False,
                            "source": "none",
                            "primary_errors": [],
                            "fallback_errors": [],
                        },
                        "error": {
                            "type": type(error).__name__,
                            "message": str(error),
                        },
                    }
                )

        # ---------------------------------------------------------
        # Benchmark completed
        # ---------------------------------------------------------

        self.logger.info(
            "Benchmark completed | questions=%s | results=%s",
            len(questions_to_run),
            len(results),
        )

        return results