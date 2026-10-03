from backend.llm.model_registry import (
    MODELS,
    get_available_models,
    get_model_config,
    get_provider,
    get_models_by_provider,
)


def test_models_registry_is_not_empty():
    assert MODELS
    assert isinstance(MODELS, dict)


def test_get_available_models_returns_all_registered_models():
    models = get_available_models()

    assert isinstance(models, list)
    assert len(models) == len(MODELS)

    for model_name in MODELS:
        assert model_name in models


def test_expected_ollama_models_are_registered():
    expected_models = {
        "qwen2.5:3b",
        "llama3.2:latest",
        "qwen2.5-coder:3b",
        "mistral:latest",
    }

    models = get_available_models()

    assert expected_models.issubset(set(models))


def test_expected_cloud_models_are_registered():
    expected_models = {
        "openai:gpt-5.6-luna",
        "anthropic:claude-sonnet",
        "google:gemini-3.6-flash",
        "deepseek:flash",
        "mistral:api",
        "kimi:k2.5",
        "xai:grok-4.7",
        "cohere:command",
        "groq:gpt-oss",
    }

    models = get_available_models()

    assert expected_models.issubset(set(models))


def test_get_model_config_returns_correct_qwen_config():
    config = get_model_config("qwen2.5:3b")

    assert config["provider"] == "ollama"
    assert config["model"] == "qwen2.5:3b"
    assert config["display_name"] == "Qwen 2.5 3B"
    assert config["supports_streaming"] is True
    assert config["supports_tools"] is False


def test_get_model_config_returns_correct_gemini_config():
    config = get_model_config("google:gemini-3.6-flash")

    assert config["provider"] == "google"
    assert config["model"] == "gemini-3.6-flash"
    assert config["display_name"] == "Gemini 3.6 Flash"
    assert config["supports_streaming"] is True
    assert config["supports_tools"] is True


def test_get_model_config_returns_correct_groq_config():
    config = get_model_config("groq:gpt-oss")

    assert config["provider"] == "groq"
    assert config["model"] == "openai/gpt-oss-20b"
    assert config["display_name"] == "GPT-OSS 20B via Groq"
    assert config["supports_streaming"] is True
    assert config["supports_tools"] is True


def test_get_model_config_unknown_model_raises_value_error():
    try:
        get_model_config("unknown:model")
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert str(exc) == "Unsupported model: unknown:model"


def test_get_provider_for_ollama_model():
    assert get_provider("qwen2.5:3b") == "ollama"
    assert get_provider("llama3.2:latest") == "ollama"
    assert get_provider("qwen2.5-coder:3b") == "ollama"
    assert get_provider("mistral:latest") == "ollama"


def test_get_provider_for_cloud_models():
    assert get_provider("openai:gpt-5.6-luna") == "openai"
    assert get_provider("anthropic:claude-sonnet") == "anthropic"
    assert get_provider("google:gemini-3.6-flash") == "google"
    assert get_provider("deepseek:flash") == "deepseek"
    assert get_provider("mistral:api") == "mistral"
    assert get_provider("kimi:k2.5") == "kimi"
    assert get_provider("xai:grok-4.7") == "xai"
    assert get_provider("cohere:command") == "cohere"
    assert get_provider("groq:gpt-oss") == "groq"


def test_get_provider_unknown_model_raises_value_error():
    try:
        get_provider("unknown:model")
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert str(exc) == "Unsupported model: unknown:model"


def test_get_models_by_provider_ollama():
    models = get_models_by_provider("ollama")

    expected = [
        "qwen2.5:3b",
        "llama3.2:latest",
        "qwen2.5-coder:3b",
        "mistral:latest",
    ]

    assert models == expected


def test_get_models_by_provider_google():
    models = get_models_by_provider("google")

    assert models == [
        "google:gemini-3.6-flash"
    ]


def test_get_models_by_provider_groq():
    models = get_models_by_provider("groq")

    assert models == [
        "groq:gpt-oss"
    ]


def test_get_models_by_provider_unknown_returns_empty_list():
    models = get_models_by_provider("unknown")

    assert models == []


def test_every_model_has_required_configuration():
    required_fields = {
        "provider",
        "model",
        "display_name",
        "supports_streaming",
        "supports_tools",
    }

    for model_name, config in MODELS.items():
        assert required_fields.issubset(config.keys()), (
            f"Missing required fields for {model_name}"
        )


def test_every_model_has_matching_provider_information():
    for model_name, config in MODELS.items():
        provider = get_provider(model_name)

        assert provider == config["provider"]


def test_every_registered_model_is_returned_by_get_available_models():
    models = get_available_models()

    for model_name in MODELS:
        config = get_model_config(model_name)

        assert model_name in models
        assert isinstance(config, dict)
        assert isinstance(config["provider"], str)
        assert isinstance(config["model"], str)
        assert isinstance(config["display_name"], str)