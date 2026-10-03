from abc import ABC, abstractmethod
from typing import Any


class LLMProvider(ABC):
    """
    Base interface for all LLM providers.

    Every provider must implement generate()
    and return the same standardized response structure.
    """

    def __init__(
        self,
        model: str,
        provider_name: str
    ):
        self.model = model
        self.provider_name = provider_name

    @abstractmethod
    def generate(self, prompt: str) -> dict[str, Any]:
        """
        Generate a response from an LLM.

        Expected return structure:

        {
            "model": str,
            "provider": str,
            "response": str,
            "input_tokens": int | None,
            "output_tokens": int | None,
            "total_tokens": int | None,
            "provider_latency": float | None,
            "finish_reason": str | None
        }
        """

        raise NotImplementedError

    def get_model_name(self) -> str:
        return self.model

    def get_provider_name(self) -> str:
        return self.provider_name