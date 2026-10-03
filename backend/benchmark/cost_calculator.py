from typing import Any


class CostCalculator:
    """
    Calculate estimated LLM API costs from token usage.

    Pricing is expressed as USD per 1,000 tokens.

    Local Ollama models have zero API cost.

    Cloud pricing must be kept explicit so that benchmark
    experiments remain reproducible.
    """

    PRICING: dict[str, dict[str, float]] = {

        # =====================================================
        # LOCAL OLLAMA MODELS
        # =====================================================

        "qwen2.5:3b": {
            "input_per_1k": 0.0,
            "output_per_1k": 0.0,
        },

        "llama3.2:latest": {
            "input_per_1k": 0.0,
            "output_per_1k": 0.0,
        },

        "qwen2.5-coder:3b": {
            "input_per_1k": 0.0,
            "output_per_1k": 0.0,
        },

        "mistral:latest": {
            "input_per_1k": 0.0,
            "output_per_1k": 0.0,
        },

        # =====================================================
        # GROQ
        # =====================================================

        # Groq:
        # openai/gpt-oss-20b
        #
        # Official pricing:
        # Input  = $0.075 / 1M tokens
        # Output = $0.30  / 1M tokens
        #
        # Converted to per 1K tokens:
        # Input  = 0.075 / 1000
        # Output = 0.30  / 1000

        "openai/gpt-oss-20b": {
            "input_per_1k": 0.000075,
            "output_per_1k": 0.000300,
        },

        # =====================================================
        # OTHER CLOUD MODELS
        # =====================================================

        # Keep these explicit.
        # Do not assign pricing unless verified.

        "gpt-5.6-luna": {
            "input_per_1k": 0.0,
            "output_per_1k": 0.0,
        },

        "claude-sonnet-4-5": {
            "input_per_1k": 0.0,
            "output_per_1k": 0.0,
        },

        "gemini-3.6-flash": {
            "input_per_1k": 0.0,
            "output_per_1k": 0.0,
        },

        "deepseek-flash": {
            "input_per_1k": 0.0,
            "output_per_1k": 0.0,
        },

        "mistral-api": {
            "input_per_1k": 0.0,
            "output_per_1k": 0.0,
        },

        "kimi-k2.5": {
            "input_per_1k": 0.0,
            "output_per_1k": 0.0,
        },

        "grok-4.7": {
            "input_per_1k": 0.0,
            "output_per_1k": 0.0,
        },

        "command": {
            "input_per_1k": 0.0,
            "output_per_1k": 0.0,
        },
    }

    # =========================================================
    # PRICING LOOKUP
    # =========================================================

    @classmethod
    def get_pricing(
        cls,
        model: str,
    ) -> dict[str, float]:

        if model in cls.PRICING:

            return cls.PRICING[model].copy()

        return {
            "input_per_1k": 0.0,
            "output_per_1k": 0.0,
        }

    # =========================================================
    # COST CALCULATION
    # =========================================================

    @classmethod
    def calculate(
        cls,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> dict[str, float]:

        pricing = cls.get_pricing(model)

        input_tokens = max(
            0,
            int(input_tokens or 0),
        )

        output_tokens = max(
            0,
            int(output_tokens or 0),
        )

        input_cost = (
            input_tokens / 1000
        ) * pricing["input_per_1k"]

        output_cost = (
            output_tokens / 1000
        ) * pricing["output_per_1k"]

        total_cost = (
            input_cost
            + output_cost
        )

        return {
            "input_cost_usd": round(
                input_cost,
                8,
            ),
            "output_cost_usd": round(
                output_cost,
                8,
            ),
            "total_cost_usd": round(
                total_cost,
                8,
            ),
        }

    # =========================================================
    # RESULT-LEVEL COST
    # =========================================================

    @classmethod
    def calculate_from_result(
        cls,
        model: str,
        result: dict[str, Any],
    ) -> dict[str, float]:

        tokens = result.get(
            "tokens",
            {},
        )

        if not isinstance(
            tokens,
            dict,
        ):
            tokens = {}

        input_tokens = tokens.get(
            "input",
            0,
        )

        output_tokens = tokens.get(
            "output",
            0,
        )

        return cls.calculate(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    # =========================================================
    # EXPERIMENT-LEVEL COST
    # =========================================================

    @classmethod
    def calculate_experiment(
        cls,
        model: str,
        results: list[dict[str, Any]],
    ) -> dict[str, float]:

        total_input_cost = 0.0
        total_output_cost = 0.0
        total_cost = 0.0

        total_input_tokens = 0
        total_output_tokens = 0

        for result in results:

            cost = cls.calculate_from_result(
                model=model,
                result=result,
            )

            total_input_cost += cost[
                "input_cost_usd"
            ]

            total_output_cost += cost[
                "output_cost_usd"
            ]

            total_cost += cost[
                "total_cost_usd"
            ]

            tokens = result.get(
                "tokens",
                {},
            )

            if isinstance(
                tokens,
                dict,
            ):

                try:
                    total_input_tokens += int(
                        tokens.get(
                            "input",
                            0,
                        )
                        or 0
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    pass

                try:
                    total_output_tokens += int(
                        tokens.get(
                            "output",
                            0,
                        )
                        or 0
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    pass

        return {
            "input_cost_usd": round(
                total_input_cost,
                8,
            ),
            "output_cost_usd": round(
                total_output_cost,
                8,
            ),
            "total_cost_usd": round(
                total_cost,
                8,
            ),
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
        }