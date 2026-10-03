import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from .base import LLMProvider


load_dotenv()


class OpenAIProvider(LLMProvider):

    def __init__(
        self,
        model: str
    ):
        super().__init__(
            model=model,
            provider_name="openai"
        )

        api_key = os.getenv(
            "OPENAI_API_KEY"
        )

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not configured."
            )

        self.client = OpenAI(
            api_key=api_key
        )

    def generate(
        self,
        prompt: str
    ) -> dict:

        start_time = time.perf_counter()

        response = self.client.responses.create(
            model=self.model,
            input=prompt
        )

        end_time = time.perf_counter()

        latency = (
            end_time - start_time
        )

        usage = response.usage

        input_tokens = None
        output_tokens = None
        total_tokens = None

        if usage:

            input_tokens = getattr(
                usage,
                "input_tokens",
                None
            )

            output_tokens = getattr(
                usage,
                "output_tokens",
                None
            )

            total_tokens = getattr(
                usage,
                "total_tokens",
                None
            )

        return {
            "model": self.model,
            "provider": self.provider_name,
            "response": response.output_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "provider_latency": latency,
            "finish_reason": None
        }