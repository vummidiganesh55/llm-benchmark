import time
from typing import Any, Callable


class RetryHandler:
    """
    Handles retry attempts and optional fallback execution
    for LLM benchmark requests.
    """

    def __init__(
        self,
        max_retries: int = 2,
        retry_delay: float = 2.0,
        backoff_factor: float = 2.0,
    ):
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")

        if retry_delay < 0:
            raise ValueError("retry_delay must be >= 0")

        if backoff_factor < 1:
            raise ValueError("backoff_factor must be >= 1")

        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.backoff_factor = backoff_factor

    @staticmethod
    def _is_retryable_error(error: Exception) -> bool:
        """
        Determine whether an exception is potentially transient.
        """

        error_text = str(error).lower()
        error_type = type(error).__name__.lower()

        retryable_keywords = [
            "timeout",
            "timed out",
            "429",
            "rate limit",
            "rate_limit",
            "resource exhausted",
            "quota exceeded",
            "temporarily unavailable",
            "temporary failure",
            "connection reset",
            "connection error",
            "service unavailable",
            "503",
            "502",
            "504",
        ]

        return (
            any(keyword in error_text for keyword in retryable_keywords)
            or "timeout" in error_type
            or "ratelimit" in error_type
            or "connection" in error_type
        )

    def _calculate_delay(self, retry_number: int) -> float:
        """
        Exponential backoff.

        retry_number:
            1 -> retry_delay
            2 -> retry_delay * backoff_factor
            3 -> retry_delay * backoff_factor^2
        """

        return self.retry_delay * (
            self.backoff_factor ** (retry_number - 1)
        )

    def execute(
        self,
        function: Callable[[], Any],
    ) -> dict[str, Any]:
        """
        Execute a provider function with retries.

        Returns structured execution information.
        """

        attempts = 0
        errors = []

        total_attempts = self.max_retries + 1

        for attempt in range(1, total_attempts + 1):
            attempts += 1

            try:
                result = function()

                return {
                    "success": True,
                    "result": result,
                    "attempts": attempts,
                    "retries": attempts - 1,
                    "errors": errors,
                }

            except Exception as error:
                retryable = self._is_retryable_error(error)

                error_info = {
                    "attempt": attempt,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "retryable": retryable,
                }

                errors.append(error_info)

                # Do not retry non-transient errors.
                if not retryable:
                    break

                # No retry remains.
                if attempt >= total_attempts:
                    break

                delay = self._calculate_delay(attempt)

                error_info["retry_delay"] = delay

                if delay > 0:
                    time.sleep(delay)

        return {
            "success": False,
            "result": None,
            "attempts": attempts,
            "retries": max(0, attempts - 1),
            "errors": errors,
        }

    def execute_with_fallback(
        self,
        primary_function: Callable[[], Any],
        fallback_function: Callable[[], Any] | None = None,
    ) -> dict[str, Any]:
        """
        Execute the primary provider with retries.

        If all retry attempts fail, optionally execute the fallback.
        """

        primary_result = self.execute(primary_function)

        if primary_result["success"]:
            return {
                "success": True,
                "source": "primary",
                "result": primary_result["result"],
                "attempts": primary_result["attempts"],
                "retries": primary_result["retries"],
                "fallback_used": False,
                "primary": primary_result,
                "fallback": None,
            }

        if fallback_function is None:
            return {
                "success": False,
                "source": "primary",
                "result": None,
                "attempts": primary_result["attempts"],
                "retries": primary_result["retries"],
                "fallback_used": False,
                "primary": primary_result,
                "fallback": None,
            }

        fallback_result = self.execute(fallback_function)

        if fallback_result["success"]:
            return {
                "success": True,
                "source": "fallback",
                "result": fallback_result["result"],
                "attempts": (
                    primary_result["attempts"]
                    + fallback_result["attempts"]
                ),
                "retries": (
                    primary_result["retries"]
                    + fallback_result["retries"]
                ),
                "fallback_used": True,
                "primary": primary_result,
                "fallback": fallback_result,
            }

        return {
            "success": False,
            "source": "fallback",
            "result": None,
            "attempts": (
                primary_result["attempts"]
                + fallback_result["attempts"]
            ),
            "retries": (
                primary_result["retries"]
                + fallback_result["retries"]
            ),
            "fallback_used": True,
            "primary": primary_result,
            "fallback": fallback_result,
        }