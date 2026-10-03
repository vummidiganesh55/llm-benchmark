import json
import re
from typing import Any

from backend.llm.ollama_provider import OllamaProvider


class LLMJudge:

    def __init__(
        self,
        model: str = "qwen2.5:3b"
    ):
        self.provider = OllamaProvider(
            model=model
        )

    def _build_prompt(
        self,
        question: str,
        expected: str,
        response: str
    ) -> str:

        return f"""
You are an evaluation system.

Compare the MODEL ANSWER with the EXPECTED ANSWER.

QUESTION:
{question}

EXPECTED ANSWER:
{expected}

MODEL ANSWER:
{response}

Scoring:

CORRECTNESS:
1.0 = the model answer is correct
0.5 = partially correct
0.0 = incorrect

RELEVANCE:
1.0 = directly answers the question
0.5 = partially relevant
0.0 = irrelevant

REASONING_QUALITY:
1.0 = reasoning is clear and logically correct
0.5 = reasoning is partially correct or incomplete
0.0 = reasoning is incorrect or missing when reasoning is required

CONFIDENCE:
A number from 0.0 to 1.0 representing your confidence in your evaluation.

For this example:
If the model answer contains the expected correct answer, mark CORRECTNESS as 1.0.

Return ONLY this JSON object.
Do not write explanations.
Do not use Markdown.
Do not add text before or after the JSON.

{{
  "correctness": 1.0,
  "relevance": 1.0,
  "reasoning_quality": 1.0,
  "confidence": 1.0
}}
""".strip()

    def _extract_json(
        self,
        text: str
    ) -> dict[str, Any]:

        text = text.strip()

        # ---------------------------------------------
        # Direct JSON
        # ---------------------------------------------

        try:

            parsed = json.loads(text)

            if isinstance(parsed, dict):
                return parsed

        except json.JSONDecodeError:
            pass

        # ---------------------------------------------
        # JSON inside response
        # ---------------------------------------------

        match = re.search(
            r"\{[\s\S]*\}",
            text
        )

        if not match:

            raise ValueError(
                "Judge did not return a JSON object."
            )

        try:

            parsed = json.loads(
                match.group(0)
            )

        except json.JSONDecodeError as e:

            raise ValueError(
                f"Invalid judge JSON: {e}"
            )

        if not isinstance(parsed, dict):

            raise ValueError(
                "Judge JSON must be an object."
            )

        return parsed

    @staticmethod
    def _validate_score(
        value: Any,
        field_name: str
    ) -> float:

        if value is None:

            raise ValueError(
                f"Judge response missing '{field_name}'."
            )

        try:

            value = float(value)

        except (TypeError, ValueError):

            raise ValueError(
                f"Invalid value for '{field_name}': {value}"
            )

        if not 0.0 <= value <= 1.0:

            raise ValueError(
                f"'{field_name}' must be between 0.0 and 1.0."
            )

        return round(
            value,
            4
        )

    def evaluate(
        self,
        question: str,
        expected: str,
        response: str
    ) -> dict[str, Any]:

        prompt = self._build_prompt(
            question=question,
            expected=expected,
            response=response
        )

        output = self.provider.generate(
            prompt
        )

        raw_response = output.get(
            "response",
            ""
        )

        if not raw_response:

            raise ValueError(
                "Judge returned an empty response."
            )

        judge_result = self._extract_json(
            raw_response
        )

        return {
            "correctness": self._validate_score(
                judge_result.get("correctness"),
                "correctness"
            ),

            "relevance": self._validate_score(
                judge_result.get("relevance"),
                "relevance"
            ),

            "reasoning_quality": self._validate_score(
                judge_result.get("reasoning_quality"),
                "reasoning_quality"
            ),

            "confidence": self._validate_score(
                judge_result.get("confidence"),
                "confidence"
            ),

            "judge_model": output.get(
                "model"
            ),

            "judge_provider": output.get(
                "provider"
            )
        }