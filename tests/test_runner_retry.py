from backend.benchmark.runner import BenchmarkRunner


class FailingProvider:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        raise TimeoutError("Primary provider timed out")


class SuccessfulProvider:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1

        return {
            "model": "fallback-model",
            "provider": "test",
            "response": "Paris",
            "input_tokens": 5,
            "output_tokens": 2,
            "total_tokens": 7,
            "latency": 0.01,
            "provider_latency": 0.005,
        }


def test_runner_uses_fallback_after_primary_failure():

    primary = FailingProvider()
    fallback = SuccessfulProvider()

    runner = BenchmarkRunner(
        provider=primary,
        fallback_provider=fallback,
    )

    questions = [
        {
            "id": 1,
            "category": "knowledge",
            "prompt": "What is the capital of France?",
            "expected": "Paris",
        }
    ]

    results = runner.run(questions)

    assert len(results) == 1

    result = results[0]

    assert result["status"] == "success"
    assert result["response"] == "Paris"

    assert primary.calls == 3
    assert fallback.calls == 1

    assert result["retry"]["fallback_used"] is True
    assert result["retry"]["source"] == "fallback"

    assert result["retry"]["attempts"] == 4
    assert result["retry"]["retries"] == 2

    assert result["evaluation"]["exact_match"] == 1.0