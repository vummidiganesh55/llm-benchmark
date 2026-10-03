import requests

from .base import LLMProvider


class OllamaProvider(LLMProvider):

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434"
    ):
        super().__init__(
            model=model,
            provider_name="ollama"
        )

        self.base_url = base_url

    def generate(self, prompt: str) -> dict:

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        response.raise_for_status()

        data = response.json()

        input_tokens = data.get(
            "prompt_eval_count"
        )

        output_tokens = data.get(
            "eval_count"
        )

        total_tokens = None

        if (
            input_tokens is not None
            and output_tokens is not None
        ):
            total_tokens = (
                input_tokens + output_tokens
            )

        total_duration = data.get(
            "total_duration"
        )

        provider_latency = None

        if total_duration is not None:
            provider_latency = (
                total_duration / 1_000_000_000
            )

        return {
            "model": self.model,
            "provider": self.provider_name,
            "response": data.get(
                "response",
                ""
            ),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "provider_latency": provider_latency,
            "finish_reason": data.get(
                "done_reason"
            )
        }