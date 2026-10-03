import os
import time

from dotenv import load_dotenv
from mistralai.client import Mistral

from .base import LLMProvider


load_dotenv()


class MistralProvider(LLMProvider):

    def __init__(self, model: str):

        super().__init__(
            model=model,
            provider_name="mistral"
        )

        api_key = os.getenv(
            "MISTRAL_API_KEY"
        )

        if not api_key:
            raise ValueError(
                "MISTRAL_API_KEY is not configured."
            )

        self.client = Mistral(
            api_key=api_key
        )

    def generate(self, prompt: str) -> dict:

        start_time = time.perf_counter()

        response = self.client.chat.complete(
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

        message = response.choices[0].message

        response_text = (
            message.content or ""
        )

        input_tokens = None
        output_tokens = None
        total_tokens = None

        usage = getattr(
            response,
            "usage",
            None
        )

        if usage:

            input_tokens = getattr(
                usage,
                "prompt_tokens",
                None
            )

            output_tokens = getattr(
                usage,
                "completion_tokens",
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
            "response": response_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "provider_latency": latency,
            "finish_reason": getattr(
                response.choices[0],
                "finish_reason",
                None
            )
        }