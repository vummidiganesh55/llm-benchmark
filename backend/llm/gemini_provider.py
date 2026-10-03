import os
import time

from dotenv import load_dotenv
from google import genai

from .base import LLMProvider


load_dotenv()


class GeminiProvider(LLMProvider):

    def __init__(
        self,
        model: str
    ):
        super().__init__(
            model=model,
            provider_name="google"
        )

        api_key = os.getenv(
            "GEMINI_API_KEY"
        )

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not configured."
            )

        self.client = genai.Client(
            api_key=api_key
        )

    def generate(
        self,
        prompt: str
    ) -> dict:

        start_time = time.perf_counter()

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )

        latency = (
            time.perf_counter() - start_time
        )

        response_text = response.text or ""

        input_tokens = None
        output_tokens = None
        total_tokens = None

        usage = getattr(
            response,
            "usage_metadata",
            None
        )

        if usage:

            input_tokens = getattr(
                usage,
                "prompt_token_count",
                None
            )

            output_tokens = getattr(
                usage,
                "candidates_token_count",
                None
            )

            total_tokens = getattr(
                usage,
                "total_token_count",
                None
            )

        finish_reason = None

        try:
            finish_reason = (
                response.candidates[0]
                .finish_reason
            )
        except (
            AttributeError,
            IndexError,
            TypeError
        ):
            pass

        return {
            "model": self.model,
            "provider": self.provider_name,
            "response": response_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "provider_latency": latency,
            "finish_reason": finish_reason
        }