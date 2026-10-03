import json

from backend.llm.ollama_provider import OllamaProvider
from backend.benchmark.runner import BenchmarkRunner

from backend.benchmark.metrics import BenchmarkMetrics
provider = OllamaProvider(
    model="qwen2.5:3b"
)

runner = BenchmarkRunner(provider)


questions = [

    {
        "id": 1,
        "category": "knowledge",
        "prompt": "What is the capital of France?",
        "expected": "Paris"
    },

    {
        "id": 2,
        "category": "math",
        "prompt": "What is 2 + 2?",
        "expected": "4"
    }

]


results = runner.run(questions)
metrics = BenchmarkMetrics.calculate(
    results
)

with open(
    "datasets/benchmark.json",
    "r",
    encoding="utf-8"
) as file:

    questions = json.load(file)


for result in results:

    print("\n-----------------------------")

    print("ID:", result["id"])
    print("Category:", result["category"])
    print("Question:", result["prompt"])
    print("Expected:", result["expected"])
    print("Response:", result["response"])
    print("Latency:", result["latency"], "seconds")
    print("Tokens:", result["tokens"])
    print("Score:", result["score"])

print("\n==============================")
print("BENCHMARK SUMMARY")
print("==============================")

print(
    "Total Questions:",
    metrics["total_questions"]
)

print(
    "Correct Answers:",
    metrics["correct_answers"]
)

print(
    "Accuracy:",
    metrics["accuracy"] * 100,
    "%"
)

print(
    "Average Latency:",
    metrics["average_latency"],
    "seconds"
)

print(
    "Total Tokens:",
    metrics["total_tokens"]
)

print(
    "Tokens/Second:",
    metrics["tokens_per_second"]
)