import json
from pathlib import Path

from backend.llm.ollama_provider import OllamaProvider
from backend.benchmark.runner import BenchmarkRunner
from backend.benchmark.metrics import BenchmarkMetrics


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = "qwen2.5:3b"

DATASET_PATH = Path("datasets/benchmark.json")

RESULTS_DIR = Path("results")

OUTPUT_PATH = RESULTS_DIR / "final_50q_qwen2.5_3b.json"


# ============================================================
# LOAD DATASET
# ============================================================

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Benchmark dataset not found: {DATASET_PATH}"
    )


with DATASET_PATH.open(
    "r",
    encoding="utf-8",
) as file:
    questions = json.load(file)


if not isinstance(questions, list):
    raise ValueError(
        "Benchmark dataset must contain a JSON list of questions."
    )


if len(questions) != 50:
    raise ValueError(
        f"Expected exactly 50 benchmark questions, "
        f"but found {len(questions)}."
    )


# ============================================================
# VALIDATE DATASET
# ============================================================

required_fields = {
    "id",
    "category",
    "prompt",
    "expected",
}


for index, question in enumerate(questions, start=1):

    missing_fields = required_fields - set(question.keys())

    if missing_fields:
        raise ValueError(
            f"Question {index} is missing fields: "
            f"{sorted(missing_fields)}"
        )


# ============================================================
# DISPLAY BENCHMARK CONFIGURATION
# ============================================================

print("\n==============================")
print("FINAL 50Q BENCHMARK")
print("==============================")

print("Model:", MODEL)
print("Dataset:", DATASET_PATH)
print("Questions:", len(questions))


# ============================================================
# LLM PROVIDER
# ============================================================

provider = OllamaProvider(
    model=MODEL
)


# ============================================================
# BENCHMARK RUNNER
# ============================================================

runner = BenchmarkRunner(provider)


# ============================================================
# RUN FINAL BENCHMARK
# ============================================================

print("\nStarting benchmark...\n")

results = runner.run(questions)


# ============================================================
# CALCULATE METRICS
# ============================================================

metrics = BenchmarkMetrics.calculate(results)


# ============================================================
# DISPLAY INDIVIDUAL RESULTS
# ============================================================

for result in results:

    print("\n-----------------------------")

    print("ID:", result.get("id"))

    print("Category:", result.get("category"))

    print("Question:", result.get("prompt"))

    print("Expected:", result.get("expected"))

    print("Response:", result.get("response"))

    print(
        "Latency:",
        result.get("latency"),
        "seconds",
    )

    print(
        "Provider Latency:",
        result.get("provider_latency"),
        "seconds",
    )

    print(
        "Tokens:",
        result.get("tokens"),
    )

    print(
        "Evaluation:",
        result.get("evaluation"),
    )

    print(
        "Retry:",
        result.get("retry"),
    )


# ============================================================
# BENCHMARK SUMMARY
# ============================================================

print("\n==============================")
print("FINAL BENCHMARK SUMMARY")
print("==============================")


print(
    json.dumps(
        metrics,
        indent=4,
        ensure_ascii=False,
        default=str,
    )
)


# ============================================================
# BUILD FINAL RESULT ARTIFACT
# ============================================================

final_output = {
    "benchmark": {
        "name": "LLM Benchmarking Platform Final 50Q Benchmark",
        "model": MODEL,
        "dataset": str(DATASET_PATH),
        "question_count": len(questions),
    },
    "metrics": metrics,
    "results": results,
}


# ============================================================
# SAVE FINAL RESULT
# ============================================================

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


with OUTPUT_PATH.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        final_output,
        file,
        indent=4,
        ensure_ascii=False,
        default=str,
    )


print("\n==============================")
print("FINAL RESULT SAVED")
print("==============================")

print(
    "File:",
    OUTPUT_PATH,
)