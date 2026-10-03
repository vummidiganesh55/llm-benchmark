import json
import os
from datetime import datetime, timezone
from typing import Any


RESULTS_FILE = "results/benchmark_results.json"


class BenchmarkStorage:

    @staticmethod
    def save(
        model: str,
        metrics: dict[str, Any],
        results: list[dict[str, Any]],
        dataset: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> str:

        os.makedirs(
            os.path.dirname(RESULTS_FILE),
            exist_ok=True,
        )

        # -------------------------------------------------
        # Generate unique experiment ID
        # -------------------------------------------------

        now = datetime.now(timezone.utc)

        experiment_id = now.strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        timestamp = now.isoformat()

        # -------------------------------------------------
        # Experiment record
        # -------------------------------------------------

        experiment = {
            "experiment_id": experiment_id,
            "timestamp": timestamp,

            "model": model,

            "config": config,

            "dataset": dataset,

            "metrics": metrics,

            "results": results,
        }

        # -------------------------------------------------
        # Load existing experiments
        # -------------------------------------------------

        if os.path.exists(RESULTS_FILE):

            with open(
                RESULTS_FILE,
                "r",
                encoding="utf-8",
            ) as file:

                experiments = json.load(file)

        else:

            experiments = []

        # -------------------------------------------------
        # Safety check
        # -------------------------------------------------

        if not isinstance(
            experiments,
            list,
        ):
            experiments = []

        # -------------------------------------------------
        # Append experiment
        # -------------------------------------------------

        experiments.append(
            experiment
        )

        # -------------------------------------------------
        # Persist
        # -------------------------------------------------

        with open(
            RESULTS_FILE,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                experiments,
                file,
                indent=2,
                ensure_ascii=False,
            )

        return experiment_id

    # -----------------------------------------------------
    # Load all experiments
    # -----------------------------------------------------

    @staticmethod
    def load_all() -> list[dict[str, Any]]:

        if not os.path.exists(
            RESULTS_FILE
        ):
            return []

        with open(
            RESULTS_FILE,
            "r",
            encoding="utf-8",
        ) as file:

            experiments = json.load(file)

        if not isinstance(
            experiments,
            list,
        ):
            return []

        return experiments

    # -----------------------------------------------------
    # Get one experiment
    # -----------------------------------------------------

    @staticmethod
    def get_by_id(
        experiment_id: str,
    ) -> dict[str, Any] | None:

        experiments = (
            BenchmarkStorage.load_all()
        )

        for experiment in experiments:

            if (
                experiment.get("experiment_id")
                == experiment_id
            ):
                return experiment

        return None