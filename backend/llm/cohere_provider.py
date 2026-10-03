import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from .base import LLMProvider


load_dotenv()


class CohereProvider(LLMProvider):

    def __init__(self, model: str):

        super().__init__(
            model=model,
            provider_name="cohere"
        )

        api_key = os.getenv(
            "COHERE_API_KEY"
        )

        if not api_key:
            raise ValueError(
                "COHERE_API_KEY is not configured."
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.cohere.ai/compatibility/v1"
        )

    def generate(self, prompt: str) -> dict:

        start_time = time.perf_counter()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        latency = (
            time.perf_counter() - start_time
        )

        choice = response.choices[0]

        usage = response.usage

        input_tokens = getattr(
            usage,
            "prompt_tokens",
            None
        ) if usage else None

        output_tokens = getattr(
            usage,
            "completion_tokens",
            None
        ) if usage else None

        total_tokens = getattr(
            usage,
            "total_tokens",
            None
        ) if usage else None

        return {
            "model": self.model,
            "provider": self.provider_name,
            "response": choice.message.content or "",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "provider_latency": latency,
            "finish_reason": choice.finish_reason
        }