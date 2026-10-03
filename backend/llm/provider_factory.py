from .model_registry import get_model_config

from .ollama_provider import OllamaProvider
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .gemini_provider import GeminiProvider
from .deepseek_provider import DeepSeekProvider
from .mistral_provider import MistralProvider
from .kimi_provider import KimiProvider
from .grok_provider import GrokProvider
from .cohere_provider import CohereProvider
from .groq_provider import GroqProvider


class ProviderFactory:

    @staticmethod
    def create(model_name: str):

        config = get_model_config(
            model_name
        )

        provider = config["provider"]

        if provider == "ollama":
            return OllamaProvider(
                model=config["model"]
            )

        if provider == "openai":
            return OpenAIProvider(
                model=config["model"]
            )

        if provider == "anthropic":
            return AnthropicProvider(
                model=config["model"]
            )

        if provider == "google":
            return GeminiProvider(
                model=config["model"]
            )

        if provider == "deepseek":
            return DeepSeekProvider(
                model=config["model"]
            )

        if provider == "mistral":
            return MistralProvider(
                model=config["model"]
            )

        if provider == "kimi":
            return KimiProvider(
                model=config["model"]
            )

        if provider == "xai":
            return GrokProvider(
                model=config["model"]
            )

        if provider == "cohere":
            return CohereProvider(
                model=config["model"]
            )

        if provider == "groq":
            return GroqProvider(
                model=config["model"]
            )

        raise NotImplementedError(
            f"Provider '{provider}' "
            f"is registered but not implemented yet."
        )