import os
import time

from dotenv import load_dotenv
from anthropic import Anthropic

from .base import LLMProvider


load_dotenv()


class AnthropicProvider(LLMProvider):

    def __init__(
        self,
        model: str
    ):
        super().__init__(
            model=model,
            provider_name="anthropic"
        )

        api_key = os.getenv(
            "ANTHROPIC_API_KEY"
        )

        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not configured."
            )

        self.client = Anthropic(
            api_key=api_key
        )

    def generate(
        self,
        prompt: str
    ) -> dict:

        start_time = time.perf_counter()

        response = self.client.messages.create(
            model=self.model,
            max_tokens=512,
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

        input_tokens = None
        output_tokens = None
        total_tokens = None

        if response.usage:

            input_tokens = getattr(
                response.usage,
                "input_tokens",
                None
            )

            output_tokens = getattr(
                response.usage,
                "output_tokens",
                None
            )

            if (
                input_tokens is not None
                and output_tokens is not None
            ):
                total_tokens = (
                    input_tokens +
                    output_tokens
                )

        response_text = ""

        for content in response.content:

            if getattr(
                content,
                "type",
                None
            ) == "text":

                response_text += (
                    content.text
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
                response,
                "stop_reason",
                None
            )
        }