from typing import Any


MODELS: dict[str, dict[str, Any]] = {

    # ============================================================
    # OLLAMA
    # ============================================================

    "qwen2.5:3b": {
        "provider": "ollama",
        "model": "qwen2.5:3b",
        "display_name": "Qwen 2.5 3B",
        "supports_streaming": True,
        "supports_tools": False,
    },

    "llama3.2:latest": {
        "provider": "ollama",
        "model": "llama3.2:latest",
        "display_name": "Llama 3.2",
        "supports_streaming": True,
        "supports_tools": False,
    },

    "qwen2.5-coder:3b": {
        "provider": "ollama",
        "model": "qwen2.5-coder:3b",
        "display_name": "Qwen 2.5 Coder 3B",
        "supports_streaming": True,
        "supports_tools": False,
    },

    "mistral:latest": {
        "provider": "ollama",
        "model": "mistral:latest",
        "display_name": "Mistral",
        "supports_streaming": True,
        "supports_tools": False,
    },


    # ============================================================
    # OPENAI
    # ============================================================

    "openai:gpt-5.6-luna": {
    "provider": "openai",
    "model": "gpt-5.6-luna",
    "display_name": "GPT-5.6 Luna",
    "supports_streaming": True,
    "supports_tools": True,
    },


    # ============================================================
    # ANTHROPIC
    # ============================================================

    "anthropic:claude-sonnet": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-5",
        "display_name": "Claude Sonnet 4.5",
        "supports_streaming": True,
        "supports_tools": True,
    },


    # ============================================================
    # GOOGLE GEMINI
    # ============================================================

    "google:gemini-3.6-flash": {
    "provider": "google",
    "model": "gemini-3.6-flash",
    "display_name": "Gemini 3.6 Flash",
    "supports_streaming": True,
    "supports_tools": True,
    },


    # ============================================================
    # DEEPSEEK
    # ============================================================

    "deepseek:flash": {
    "provider": "deepseek",
    "model": "deepseek-flash",
    "display_name": "DeepSeek Flash",
    "supports_streaming": True,
    "supports_tools": True,
    },


   # ============================================================
# MISTRAL API
# ============================================================

    "mistral:api": {
    "provider": "mistral",
    "model": "mistral-large-latest",
    "display_name": "Mistral Large",
    "supports_streaming": True,
    "supports_tools": True,
    },


# ============================================================
# KIMI / MOONSHOT
# ============================================================

    "kimi:k2.5": {
    "provider": "kimi",
    "model": "kimi-k2.5",
    "display_name": "Kimi K2.5",
    "supports_streaming": True,
    "supports_tools": True,
    },


# ============================================================
# GROK / XAI
# ============================================================

    "xai:grok-4.7": {
    "provider": "xai",
    "model": "grok-4.7",
    "display_name": "Grok 4.7",
    "supports_streaming": True,
    "supports_tools": True,
    },


# ============================================================
# COHERE
# ============================================================

    "cohere:command": {
    "provider": "cohere",
    "model": "command-a-plus-05-2026",
    "display_name": "Cohere Command A+",
    "supports_streaming": True,
    "supports_tools": True,
    },


# ============================================================
# GROQ
# ============================================================

    "groq:gpt-oss": {
    "provider": "groq",
    "model": "openai/gpt-oss-20b",
    "display_name": "GPT-OSS 20B via Groq",
    "supports_streaming": True,
    "supports_tools": True,
    },
}


def get_available_models() -> list[str]:
    """
    Return all registered model IDs.
    """

    return list(MODELS.keys())


def get_model_config(
    model_name: str
) -> dict[str, Any]:
    """
    Return configuration for a registered model.
    """

    if model_name not in MODELS:
        raise ValueError(
            f"Unsupported model: {model_name}"
        )

    return MODELS[model_name]


def get_provider(
    model_name: str
) -> str:
    """
    Return the provider associated with a model.
    """

    config = get_model_config(model_name)

    return config["provider"]


def get_models_by_provider(
    provider: str
) -> list[str]:
    """
    Return all models belonging to a provider.
    """

    return [
        model_name
        for model_name, config in MODELS.items()
        if config["provider"] == provider
    ]