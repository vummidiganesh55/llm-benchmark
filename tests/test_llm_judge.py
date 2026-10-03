from unittest.mock import MagicMock, patch

import pytest

from backend.benchmark.llm_judge import LLMJudge


def test_llm_judge_initialization():
    with patch(
        "backend.benchmark.llm_judge.OllamaProvider"
    ) as mock_provider:

        judge = LLMJudge(model="qwen2.5:3b")

        mock_provider.assert_called_once_with(
            model="qwen2.5:3b"
        )

        assert judge.provider is mock_provider.return_value


def test_llm_judge_build_prompt():
    with patch(
        "backend.benchmark.llm_judge.OllamaProvider"
    ):
        judge = LLMJudge(model="qwen2.5:3b")

        prompt = judge._build_prompt(
            question="What is the capital of France?",
            response="Paris",
            expected="Paris",
        )

        assert isinstance(prompt, str)
        assert "What is the capital of France?" in prompt
        assert "Paris" in prompt
        assert "correctness" in prompt
        assert "relevance" in prompt
        assert "reasoning_quality" in prompt
        assert "confidence" in prompt


def test_llm_judge_success():
    mock_response = {
        "response": (
            '{"correctness": 1.0, '
            '"relevance": 1.0, '
            '"reasoning_quality": 0.9, '
            '"confidence": 0.95}'
        )
    }

    with patch(
        "backend.benchmark.llm_judge.OllamaProvider"
    ) as mock_provider:

        mock_provider.return_value.generate.return_value = mock_response

        judge = LLMJudge(model="qwen2.5:3b")

        result = judge.evaluate(
            question="What is the capital of France?",
            response="Paris",
            expected="Paris",
        )

        assert result["correctness"] == 1.0
        assert result["relevance"] == 1.0
        assert result["reasoning_quality"] == 0.9
        assert result["confidence"] == 0.95


def test_llm_judge_scores_are_bounded():
    mock_response = {
        "response": (
            '{"correctness": 0.8, '
            '"relevance": 0.7, '
            '"reasoning_quality": 0.6, '
            '"confidence": 0.5}'
        )
    }

    with patch(
        "backend.benchmark.llm_judge.OllamaProvider"
    ) as mock_provider:

        mock_provider.return_value.generate.return_value = mock_response

        judge = LLMJudge(model="qwen2.5:3b")

        result = judge.evaluate(
            question="What is 2 + 2?",
            response="4",
            expected="4",
        )

        for key in (
            "correctness",
            "relevance",
            "reasoning_quality",
            "confidence",
        ):
            assert 0.0 <= result[key] <= 1.0


def test_llm_judge_invalid_score_raises_error():
    mock_response = {
        "response": (
            '{"correctness": 2.0, '
            '"relevance": 1.0, '
            '"reasoning_quality": 1.0, '
            '"confidence": 1.0}'
        )
    }

    with patch(
        "backend.benchmark.llm_judge.OllamaProvider"
    ) as mock_provider:

        mock_provider.return_value.generate.return_value = mock_response

        judge = LLMJudge(model="qwen2.5:3b")

        with pytest.raises(ValueError):
            judge.evaluate(
                question="What is 2 + 2?",
                response="4",
                expected="4",
            )


def test_llm_judge_missing_score_raises_error():
    mock_response = {
        "response": (
            '{"correctness": 1.0, '
            '"relevance": 1.0}'
        )
    }

    with patch(
        "backend.benchmark.llm_judge.OllamaProvider"
    ) as mock_provider:

        mock_provider.return_value.generate.return_value = mock_response

        judge = LLMJudge(model="qwen2.5:3b")

        with pytest.raises((ValueError, KeyError)):
            judge.evaluate(
                question="What is 2 + 2?",
                response="4",
                expected="4",
            )


def test_llm_judge_invalid_json_raises_error():
    mock_response = {
        "response": "This is not valid JSON."
    }

    with patch(
        "backend.benchmark.llm_judge.OllamaProvider"
    ) as mock_provider:

        mock_provider.return_value.generate.return_value = mock_response

        judge = LLMJudge(model="qwen2.5:3b")

        with pytest.raises((ValueError, TypeError)):
            judge.evaluate(
                question="What is 2 + 2?",
                response="4",
                expected="4",
            )