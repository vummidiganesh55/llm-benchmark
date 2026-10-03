from backend.benchmark.cost_calculator import CostCalculator


def test_local_ollama_model_is_zero_cost():

    result = CostCalculator.calculate(
        model="qwen2.5:3b",
        input_tokens=1000,
        output_tokens=1000,
    )

    assert result["input_cost_usd"] == 0.0
    assert result["output_cost_usd"] == 0.0
    assert result["total_cost_usd"] == 0.0


def test_groq_gpt_oss_20b_pricing():

    result = CostCalculator.calculate(
        model="openai/gpt-oss-20b",
        input_tokens=4084,
        output_tokens=5276,
    )

    assert result["input_cost_usd"] == 0.0003063
    assert result["output_cost_usd"] == 0.0015828
    assert result["total_cost_usd"] == 0.0018891


def test_groq_pricing_lookup():

    pricing = CostCalculator.get_pricing(
        "openai/gpt-oss-20b"
    )

    assert pricing["input_per_1k"] == 0.000075
    assert pricing["output_per_1k"] == 0.000300


def test_unknown_model_does_not_invent_price():

    result = CostCalculator.calculate(
        model="unknown-model",
        input_tokens=1000,
        output_tokens=1000,
    )

    assert result["total_cost_usd"] == 0.0


def test_experiment_cost():

    results = [
        {
            "tokens": {
                "input": 1000,
                "output": 2000,
            }
        },
        {
            "tokens": {
                "input": 2000,
                "output": 3000,
            },
        },
    ]

    result = CostCalculator.calculate_experiment(
        model="openai/gpt-oss-20b",
        results=results,
    )

    assert result["input_tokens"] == 3000
    assert result["output_tokens"] == 5000

    assert result["input_cost_usd"] == 0.000225
    assert result["output_cost_usd"] == 0.0015
    assert result["total_cost_usd"] == 0.001725