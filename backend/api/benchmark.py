import json
import requests

from fastapi import APIRouter, HTTPException

from backend.llm.provider_factory import ProviderFactory

from backend.benchmark.config import BenchmarkConfig
from backend.benchmark.runner import BenchmarkRunner
from backend.benchmark.metrics import BenchmarkMetrics
from backend.benchmark.storage import BenchmarkStorage
from backend.benchmark.dataset_version import DatasetVersion
from backend.benchmark.regression_service import RegressionService
from backend.benchmark.experiment_report import ExperimentReportGenerator
from backend.benchmark.concurrency_runner import ConcurrencyRunner
from backend.benchmark.leaderboard import ModelLeaderboard

from backend.schemas.benchmark import (
    BenchmarkRequest,
    RegressionRequest,
    ConcurrencyRequest,
    LeaderboardRequest,
)

from backend.llm.model_registry import (
    get_model_config,
    get_available_models,
)


router = APIRouter()


# ============================================================
# GET AVAILABLE MODELS
# ============================================================

@router.get("/models")
def get_models():

    try:

        return {
            "models": get_available_models()
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Unable to load model registry: {str(e)}",
        )


# ============================================================
# RUN BENCHMARK
# ============================================================

@router.post("/run")
def run_benchmark(
    request: BenchmarkRequest,
):

    # --------------------------------------------------------
    # 1. Validate model
    # --------------------------------------------------------

    try:

        model_config = get_model_config(
            request.model
        )

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e),
        )


    # --------------------------------------------------------
    # 2. Load benchmark dataset
    # --------------------------------------------------------

    try:

        with open(
            "datasets/benchmark.json",
            "r",
            encoding="utf-8",
        ) as file:

            questions = json.load(file)

    except FileNotFoundError:

        raise HTTPException(
            status_code=500,
            detail="Benchmark dataset not found.",
        )

    except json.JSONDecodeError:

        raise HTTPException(
            status_code=500,
            detail="Benchmark dataset contains invalid JSON.",
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Unable to load benchmark dataset: "
                f"{str(e)}"
            ),
        )


    # --------------------------------------------------------
    # 3. Validate dataset
    # --------------------------------------------------------

    if not questions:

        raise HTTPException(
            status_code=500,
            detail="Benchmark dataset is empty.",
        )


    # --------------------------------------------------------
    # 4. Get dataset version information
    # --------------------------------------------------------

    try:

        dataset_version = DatasetVersion()

        dataset_info = dataset_version.get_info()

    except FileNotFoundError as e:

        raise HTTPException(
            status_code=500,
            detail=f"Dataset metadata error: {str(e)}",
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Unable to calculate dataset version: "
                f"{str(e)}"
            ),
        )


    # Preserve original dataset size.
    total_dataset_questions = len(questions)


    # --------------------------------------------------------
    # 5. Validate benchmark limit
    # --------------------------------------------------------

    if request.limit is not None:

        if request.limit <= 0:

            raise HTTPException(
                status_code=400,
                detail="limit must be greater than 0.",
            )

        if request.limit > total_dataset_questions:

            raise HTTPException(
                status_code=400,
                detail=(
                    f"limit cannot exceed the number of "
                    f"questions in the dataset "
                    f"({total_dataset_questions})."
                ),
            )

        questions = questions[:request.limit]


    questions_run = len(questions)


    # --------------------------------------------------------
    # 6. Create Benchmark Configuration
    # --------------------------------------------------------

    benchmark_config = BenchmarkConfig(
        model=provider_model_name(
            model_config,
            request.model,
        ),
        provider=model_provider_name(
            model_config,
        ),
        dataset_path="datasets/benchmark.json",
        dataset_metadata_path="datasets/metadata.json",
        question_limit=request.limit,
    )


    # --------------------------------------------------------
    # 7. Create LLM provider
    # --------------------------------------------------------

    try:

        provider = ProviderFactory.create(
            request.model
        )

    except NotImplementedError as e:

        raise HTTPException(
            status_code=501,
            detail=str(e),
        )

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Unable to create LLM provider: "
                f"{str(e)}"
            ),
        )


    # --------------------------------------------------------
    # 8. Run benchmark
    # --------------------------------------------------------

    try:

        runner = BenchmarkRunner(
            provider=provider,
            config=benchmark_config,
        )

        results = runner.run(
            questions
        )

    except requests.exceptions.ConnectionError:

        raise HTTPException(
            status_code=503,
            detail=(
                "Unable to connect to Ollama. "
                "Make sure Ollama is running on "
                "http://localhost:11434."
            ),
        )

    except requests.exceptions.Timeout:

        raise HTTPException(
            status_code=504,
            detail=(
                "Ollama request timed out. "
                "The model took too long to respond."
            ),
        )

    except requests.exceptions.HTTPError as e:

        raise HTTPException(
            status_code=502,
            detail=(
                f"Ollama returned an HTTP error: "
                f"{str(e)}"
            ),
        )

    except requests.exceptions.RequestException as e:

        raise HTTPException(
            status_code=502,
            detail=(
                f"Ollama request failed: "
                f"{str(e)}"
            ),
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Benchmark execution failed: "
                f"{str(e)}"
            ),
        )


    # --------------------------------------------------------
    # 9. Validate benchmark results
    # --------------------------------------------------------

    if not results:

        raise HTTPException(
            status_code=500,
            detail=(
                "Benchmark completed but "
                "returned no results."
            ),
        )


    # --------------------------------------------------------
    # 10. Calculate metrics
    # --------------------------------------------------------

    try:

        metrics = BenchmarkMetrics.calculate(
            results
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Unable to calculate benchmark metrics: "
                f"{str(e)}"
            ),
        )


    # --------------------------------------------------------
    # 11. Save reproducible experiment
    # --------------------------------------------------------

    try:

        experiment_id = BenchmarkStorage.save(
            model=provider.model,
            metrics=metrics,
            results=results,
            dataset=dataset_info,
            config=benchmark_config.to_dict(),
        )

    except TypeError:

        # Compatibility fallback for older storage versions.
        experiment_id = BenchmarkStorage.save(
            model=provider.model,
            metrics=metrics,
            results=results,
            dataset=dataset_info,
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Unable to save benchmark results: "
                f"{str(e)}"
            ),
        )


    # --------------------------------------------------------
    # 12. Return successful response
    # --------------------------------------------------------

    return {

        "status": "success",

        "experiment_id": experiment_id,

        "model": provider.model,

        "benchmark": {

            "dataset_questions": (
                total_dataset_questions
            ),

            "questions_run": questions_run,

            "limit": request.limit,

        },

        "dataset": dataset_info,

        "config": benchmark_config.to_dict(),

        "metrics": metrics,

        "results": results,

    }


# ============================================================
# REGRESSION DETECTION
# ============================================================

@router.post("/regression")
def detect_regression(
    request: RegressionRequest,
):

    try:

        service = RegressionService()

        result = service.compare_experiments(
            baseline_experiment_id=(
                request.baseline_experiment_id
            ),
            current_experiment_id=(
                request.current_experiment_id
            ),
        )

        return {
            "status": "success",
            "regression": result,
        }

    except ValueError as e:

        raise HTTPException(
            status_code=404,
            detail=str(e),
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Unable to detect regression: "
                f"{str(e)}"
            ),
        )


# ============================================================
# EXPERIMENT REPORT
# ============================================================

@router.get("/report")
def get_experiment_report(
    experiment_id: str,
):

    try:

        storage = BenchmarkStorage()

        experiment = storage.get_by_id(
            experiment_id
        )

        if experiment is None:

            raise HTTPException(
                status_code=404,
                detail=(
                    f"Experiment not found: "
                    f"{experiment_id}"
                ),
            )

        generator = ExperimentReportGenerator(experiment)

        report = generator.generate_report()

        text_report = generator.generate_text_report()

        return {
            "status": "success",
            "report": report,
            "text_report": text_report,
        }

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Unable to generate experiment report: "
                f"{str(e)}"
            ),
        )


# ============================================================
# CONCURRENCY BENCHMARK
# ============================================================

@router.post("/concurrency")
def run_concurrency_benchmark(
    request: ConcurrencyRequest,
):

    # --------------------------------------------------------
    # 1. Validate request
    # --------------------------------------------------------

    if request.limit <= 0:

        raise HTTPException(
            status_code=400,
            detail="limit must be greater than 0.",
        )

    if request.concurrency <= 0:

        raise HTTPException(
            status_code=400,
            detail="concurrency must be greater than 0.",
        )


    # --------------------------------------------------------
    # 2. Validate model
    # --------------------------------------------------------

    try:

        model_config = get_model_config(
            request.model
        )

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e),
        )


    # --------------------------------------------------------
    # 3. Load dataset
    # --------------------------------------------------------

    try:

        with open(
            "datasets/benchmark.json",
            "r",
            encoding="utf-8",
        ) as file:

            questions = json.load(file)

    except FileNotFoundError:

        raise HTTPException(
            status_code=500,
            detail="Benchmark dataset not found.",
        )

    except json.JSONDecodeError:

        raise HTTPException(
            status_code=500,
            detail="Benchmark dataset contains invalid JSON.",
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Unable to load benchmark dataset: "
                f"{str(e)}"
            ),
        )


    # --------------------------------------------------------
    # 4. Validate limit
    # --------------------------------------------------------

    if not questions:

        raise HTTPException(
            status_code=500,
            detail="Benchmark dataset is empty.",
        )

    if request.limit > len(questions):

        raise HTTPException(
            status_code=400,
            detail=(
                f"limit cannot exceed the number of "
                f"questions in the dataset "
                f"({len(questions)})."
            ),
        )

    questions = questions[:request.limit]


    # --------------------------------------------------------
    # 5. Create configuration
    # --------------------------------------------------------

    benchmark_config = BenchmarkConfig(
        model=provider_model_name(
            model_config,
            request.model,
        ),
        provider=model_provider_name(
            model_config,
        ),
        dataset_path="datasets/benchmark.json",
        dataset_metadata_path="datasets/metadata.json",
        question_limit=request.limit,
    )


    # --------------------------------------------------------
    # 6. Create provider
    # --------------------------------------------------------

    try:

        provider = ProviderFactory.create(
            request.model
        )

    except NotImplementedError as e:

        raise HTTPException(
            status_code=501,
            detail=str(e),
        )

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Unable to create LLM provider: "
                f"{str(e)}"
            ),
        )


    # --------------------------------------------------------
    # 7. Execute individual questions concurrently
    # --------------------------------------------------------

    def execute_question(question):

        runner = BenchmarkRunner(
            provider=provider,
            config=benchmark_config,
        )

        result = runner.run(
            [question]
        )

        if not result:

            raise RuntimeError(
                "Benchmark runner returned no result."
            )

        return result[0]


    try:

        concurrency_runner = ConcurrencyRunner(
            worker=execute_question,
            max_workers=request.concurrency,
        )

        concurrency_result = concurrency_runner.run(
            questions
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Concurrency benchmark failed: "
                f"{str(e)}"
            ),
        )


    # --------------------------------------------------------
    # 8. Return concurrency metrics
    # --------------------------------------------------------

    return {

        "status": "success",

        "model": provider.model,

        "benchmark": {

            "questions_run": len(questions),

            "concurrency": request.concurrency,

        },

        "metrics": {

            "total_requests": (
                concurrency_result[
                    "total_requests"
                ]
            ),

            "successful_requests": (
                concurrency_result[
                    "successful_requests"
                ]
            ),

            "failed_requests": (
                concurrency_result[
                    "failed_requests"
                ]
            ),

            "success_rate": (
                concurrency_result[
                    "success_rate"
                ]
            ),

            "error_rate": (
                concurrency_result[
                    "error_rate"
                ]
            ),

            "total_time": (
                concurrency_result[
                    "total_time"
                ]
            ),

            "average_latency": (
                concurrency_result[
                    "average_latency"
                ]
            ),

            "throughput": (
                concurrency_result[
                    "throughput"
                ]
            ),

            "max_workers": (
                concurrency_result[
                    "max_workers"
                ]
            ),

        },

        "results": (
            concurrency_result[
                "results"
            ]
        ),

    }


# ============================================================
# MODEL LEADERBOARD
# ============================================================

@router.get("/leaderboard")
def get_leaderboard(
    sort_by: str = "accuracy",
    descending: bool = True,
    latest_only: bool = False,
):

    """
    Build a model leaderboard from stored benchmark
    experiments.

    The endpoint exposes measurable benchmark metrics
    without assigning an overall winner.
    """

    try:

        # ----------------------------------------------------
        # Load persisted experiments
        # ----------------------------------------------------

        experiments = (
            BenchmarkStorage.get_all()
        )

        if experiments is None:

            experiments = []


        # ----------------------------------------------------
        # Build leaderboard
        # ----------------------------------------------------

        leaderboard = ModelLeaderboard(
            experiments
        )


        # ----------------------------------------------------
        # Select all experiments or latest per model
        # ----------------------------------------------------

        if latest_only:

            entries = leaderboard.build_latest(
                sort_by=sort_by,
                descending=descending,
            )

        else:

            entries = leaderboard.build(
                sort_by=sort_by,
                descending=descending,
            )


        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        summary = leaderboard.summary()


        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return {

            "status": "success",

            "sort_by": sort_by,

            "descending": descending,

            "latest_only": latest_only,

            "summary": summary,

            "leaderboard": entries,

        }


    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Unable to build model leaderboard: "
                f"{str(e)}"
            ),
        )


# ============================================================
# CONFIGURATION HELPERS
# ============================================================

def provider_model_name(
    model_config,
    requested_model: str,
) -> str:

    """
    Resolve the configured model name.

    Falls back to the model supplied by the API request.
    """

    if isinstance(model_config, dict):

        model_name = (
            model_config.get("model")
            or model_config.get("model_name")
        )

        if model_name:

            return str(model_name)

    return requested_model


def model_provider_name(
    model_config,
) -> str | None:

    """
    Resolve the provider name from the model registry.
    """

    if isinstance(model_config, dict):

        provider = model_config.get(
            "provider"
        )

        if provider:

            return str(provider)

    return None