from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from time import perf_counter
from typing import Any, Callable


class ConcurrencyRunner:
    """
    Executes benchmark requests concurrently and reports
    throughput, latency, and success/failure statistics.
    """

    def __init__(
        self,
        worker: Callable[[Any], Any],
        max_workers: int = 4,
    ):
        if max_workers <= 0:
            raise ValueError(
                "max_workers must be greater than 0."
            )

        self.worker = worker
        self.max_workers = max_workers

    def run(
        self,
        inputs: list[Any],
    ) -> dict[str, Any]:
        """
        Execute the supplied worker concurrently.

        Each input is passed to the worker independently.
        """

        if not inputs:
            return {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "success_rate": 0.0,
                "error_rate": 0.0,
                "total_time": 0.0,
                "average_latency": 0.0,
                "throughput": 0.0,
                "max_workers": self.max_workers,
                "results": [],
            }

        start_time = perf_counter()

        results: list[dict[str, Any]] = []

        with ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:

            future_map = {
                executor.submit(
                    self._execute,
                    index,
                    item,
                ): index
                for index, item in enumerate(inputs)
            }

            for future in as_completed(
                future_map
            ):
                index = future_map[future]

                try:
                    result = future.result()

                except Exception as exc:
                    result = {
                        "index": index,
                        "success": False,
                        "latency": 0.0,
                        "result": None,
                        "error": str(exc),
                    }

                results.append(result)

        total_time = perf_counter() - start_time

        # Restore original input ordering.
        results.sort(
            key=lambda item: item["index"]
        )

        total_requests = len(results)

        successful_requests = sum(
            1
            for result in results
            if result["success"]
        )

        failed_requests = (
            total_requests
            - successful_requests
        )

        success_rate = (
            successful_requests
            / total_requests
            if total_requests
            else 0.0
        )

        error_rate = (
            failed_requests
            / total_requests
            if total_requests
            else 0.0
        )

        successful_latencies = [
            result["latency"]
            for result in results
            if result["success"]
        ]

        average_latency = (
            sum(successful_latencies)
            / len(successful_latencies)
            if successful_latencies
            else 0.0
        )

        throughput = (
            successful_requests / total_time
            if total_time > 0
            else 0.0
        )

        return {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "success_rate": success_rate,
            "error_rate": error_rate,
            "total_time": total_time,
            "average_latency": average_latency,
            "throughput": throughput,
            "max_workers": self.max_workers,
            "results": results,
        }

    def _execute(
        self,
        index: int,
        item: Any,
    ) -> dict[str, Any]:
        """
        Execute one worker request and measure latency.
        """

        start_time = perf_counter()

        try:

            result = self.worker(item)

            latency = (
                perf_counter() - start_time
            )

            return {
                "index": index,
                "success": True,
                "latency": latency,
                "result": result,
                "error": None,
            }

        except Exception as exc:

            latency = (
                perf_counter() - start_time
            )

            return {
                "index": index,
                "success": False,
                "latency": latency,
                "result": None,
                "error": str(exc),
            }