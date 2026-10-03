import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from .base import LLMProvider


load_dotenv()


class DeepSeekProvider(LLMProvider):

    def __init__(
        self,
        model: str
    ):
        super().__init__(
            model=model,
            provider_name="deepseek"
        )

        api_key = os.getenv(
            "DEEPSEEK_API_KEY"
        )

        if not api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY is not configured."
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )

    def generate(
        self,
        prompt: str
    ) -> dict:

        start_time = time.perf_counter()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            stream=False
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

        if response.usage:

            input_tokens = getattr(
                response.usage,
                "prompt_tokens",
                None
            )

            output_tokens = getattr(
                response.usage,
                "completion_tokens",
                None
            )

            total_tokens = getattr(
                response.usage,
                "total_tokens",
                None
            )

        finish_reason = None

        if response.choices:

            finish_reason = (
                response.choices[0].finish_reason
            )

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