def test_model_registry_contains_multiple_providers():

    from backend.llm.model_registry import MODELS

    providers = {
        config["provider"]
        for config in MODELS.values()
    }

    assert "ollama" in providers
    assert "openai" in providers
    assert "anthropic" in providers
    assert "google" in providers
    assert "deepseek" in providers
    assert "mistral" in providers
    assert "kimi" in providers
    assert "xai" in providers
    assert "cohere" in providers
    assert "groq" in providers


def test_model_config_contains_required_fields():

    from backend.llm.model_registry import MODELS

    required_fields = {
        "provider",
        "model",
        "display_name",
        "supports_streaming",
        "supports_tools",
    }

    for model_name, config in MODELS.items():

        assert required_fields.issubset(
            config.keys()
        ), f"Missing fields for {model_name}"


def test_get_provider():

    from backend.llm.model_registry import get_provider

    assert (
        get_provider("qwen2.5:3b")
        == "ollama"
    )


def test_get_models_by_provider():

    from backend.llm.model_registry import (
        get_models_by_provider
    )

    ollama_models = get_models_by_provider(
        "ollama"
    )

    assert "qwen2.5:3b" in ollama_models
    assert "llama3.2:latest" in ollama_models
    assert "mistral:latest" in ollama_models

def test_provider_factory_ollama():

    from backend.llm.provider_factory import (
        ProviderFactory
    )

    provider = ProviderFactory.create(
        "qwen2.5:3b"
    )

    assert type(provider).__name__ == "OllamaProvider"

    assert provider.model == "qwen2.5:3b"

    assert provider.provider_name == "ollama"
def test_openai_provider_requires_api_key(
    monkeypatch
):

    from backend.llm.openai_provider import (
        OpenAIProvider
    )

    monkeypatch.delenv(
        "OPENAI_API_KEY",
        raising=False
    )

    import pytest

    with pytest.raises(
        ValueError,
        match="OPENAI_API_KEY"
    ):

        OpenAIProvider(
            model="gpt-5.6-luna"
        )
def test_anthropic_provider_requires_api_key(
    monkeypatch
):

    import pytest

    from backend.llm.anthropic_provider import (
        AnthropicProvider
    )

    monkeypatch.delenv(
        "ANTHROPIC_API_KEY",
        raising=False
    )

    with pytest.raises(
        ValueError,
        match="ANTHROPIC_API_KEY"
    ):

        AnthropicProvider(
            model="claude-sonnet-4-5"
        )
def test_gemini_provider_requires_api_key(
    monkeypatch
):

    import pytest

    from backend.llm.gemini_provider import (
        GeminiProvider
    )

    monkeypatch.delenv(
        "GEMINI_API_KEY",
        raising=False
    )

    with pytest.raises(
        ValueError,
        match="GEMINI_API_KEY"
    ):

        GeminiProvider(
            model="gemini-3.6-flash"
        )

def test_deepseek_provider_requires_api_key(
    monkeypatch
):

    import pytest

    from backend.llm.deepseek_provider import (
        DeepSeekProvider
    )

    monkeypatch.delenv(
        "DEEPSEEK_API_KEY",
        raising=False
    )

    with pytest.raises(
        ValueError,
        match="DEEPSEEK_API_KEY"
    ):

        DeepSeekProvider(
            model="deepseek-flash"
        )
def test_mistral_provider_requires_api_key(
    monkeypatch
):

    import pytest

    from backend.llm.mistral_provider import (
        MistralProvider
    )

    monkeypatch.delenv(
        "MISTRAL_API_KEY",
        raising=False
    )

    with pytest.raises(
        ValueError,
        match="MISTRAL_API_KEY"
    ):

        MistralProvider(
            model="mistral-large-latest"
        )


def test_kimi_provider_requires_api_key(
    monkeypatch
):

    import pytest

    from backend.llm.kimi_provider import (
        KimiProvider
    )

    monkeypatch.delenv(
        "MOONSHOT_API_KEY",
        raising=False
    )

    with pytest.raises(
        ValueError,
        match="MOONSHOT_API_KEY"
    ):

        KimiProvider(
            model="kimi-k2.5"
        )


def test_grok_provider_requires_api_key(
    monkeypatch
):

    import pytest

    from backend.llm.grok_provider import (
        GrokProvider
    )

    monkeypatch.delenv(
        "XAI_API_KEY",
        raising=False
    )

    with pytest.raises(
        ValueError,
        match="XAI_API_KEY"
    ):

        GrokProvider(
            model="grok-4.7"
        )


def test_cohere_provider_requires_api_key(
    monkeypatch
):

    import pytest

    from backend.llm.cohere_provider import (
        CohereProvider
    )

    monkeypatch.delenv(
        "COHERE_API_KEY",
        raising=False
    )

    with pytest.raises(
        ValueError,
        match="COHERE_API_KEY"
    ):

        CohereProvider(
            model="command-a-plus-05-2026"
        )


def test_groq_provider_requires_api_key(
    monkeypatch
):

    import pytest

    from backend.llm.groq_provider import (
        GroqProvider
    )

    monkeypatch.delenv(
        "GROQ_API_KEY",
        raising=False
    )

    with pytest.raises(
        ValueError,
        match="GROQ_API_KEY"
    ):

        GroqProvider(
            model="openai/gpt-oss-20b"
        )