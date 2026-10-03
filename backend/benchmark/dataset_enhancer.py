import json
from pathlib import Path
from typing import Any


class DatasetEnhancer:

    def __init__(
        self,
        dataset_path: str = "datasets/benchmark.json",
    ):
        self.dataset_path = Path(dataset_path)

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    def load(self) -> list[dict[str, Any]]:

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.dataset_path}"
            )

        with open(
            self.dataset_path,
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        if not isinstance(data, list):
            raise ValueError(
                "Benchmark dataset must contain a JSON list."
            )

        return data

    # --------------------------------------------------------
    # VALIDATE ORIGINAL SCHEMA
    # --------------------------------------------------------

    @staticmethod
    def validate_question(
        question: dict[str, Any],
    ) -> None:

        required_fields = {
            "id",
            "category",
            "prompt",
            "expected",
        }

        missing_fields = (
            required_fields
            - question.keys()
        )

        if missing_fields:

            raise ValueError(
                f"Question {question.get('id')} "
                f"is missing fields: "
                f"{sorted(missing_fields)}"
            )

    # --------------------------------------------------------
    # DIFFICULTY CLASSIFICATION
    # --------------------------------------------------------

    @staticmethod
    def infer_difficulty(
        question: dict[str, Any],
    ) -> str:

        question_id = int(
            question["id"]
        )

        # ----------------------------------------------------
        # Math: Questions 1-10
        # Basic arithmetic operations.
        # ----------------------------------------------------

        if 1 <= question_id <= 10:
            return "easy"

        # ----------------------------------------------------
        # Knowledge: Questions 11-20
        # Basic factual knowledge.
        # ----------------------------------------------------

        if 11 <= question_id <= 20:
            return "easy"

        # ----------------------------------------------------
        # Reasoning: Questions 21-30
        # Mostly simple reasoning, with two slightly
        # more involved questions.
        # ----------------------------------------------------

        if question_id in {22, 30}:
            return "medium"

        if 21 <= question_id <= 30:
            return "easy"

        # ----------------------------------------------------
        # Coding: Questions 31-40
        # Basic Python knowledge.
        # ----------------------------------------------------

        if 31 <= question_id <= 40:
            return "easy"

        # ----------------------------------------------------
        # General: Questions 41-50
        # Basic technical terminology.
        # ----------------------------------------------------

        if 41 <= question_id <= 50:
            return "easy"

        raise ValueError(
            f"Unable to classify difficulty for "
            f"question {question_id}."
        )

    # --------------------------------------------------------
    # EVALUATION TYPE
    # --------------------------------------------------------

    @staticmethod
    def infer_evaluation_type(
        question: dict[str, Any],
    ) -> str:

        category = str(
            question["category"]
        ).lower()

        if category == "math":
            return "exact"

        if category == "coding":
            return "exact"

        if category == "knowledge":
            return "exact"

        if category == "reasoning":
            return "exact"

        if category == "general":
            return "exact"

        raise ValueError(
            f"Unknown category: {category}"
        )

    # --------------------------------------------------------
    # TAG CLASSIFICATION
    # --------------------------------------------------------

    @staticmethod
    def infer_tags(
        question: dict[str, Any],
    ) -> list[str]:

        question_id = int(
            question["id"]
        )

        category = str(
            question["category"]
        ).lower()

        prompt = str(
            question["prompt"]
        ).lower()

        tags: list[str] = [
            category
        ]

        # ----------------------------------------------------
        # MATH
        # ----------------------------------------------------

        if category == "math":

            if any(
                word in prompt
                for word in [
                    "multiply",
                    "multiplied",
                    "times",
                ]
            ):
                tags.append("multiplication")

            elif any(
                word in prompt
                for word in [
                    "divide",
                    "divided",
                ]
            ):
                tags.append("division")

            elif any(
                word in prompt
                for word in [
                    "plus",
                    "add",
                    "sum",
                ]
            ):
                tags.append("addition")

            elif any(
                word in prompt
                for word in [
                    "minus",
                    "subtract",
                    "difference",
                ]
            ):
                tags.append("subtraction")

            else:
                tags.append("arithmetic")

        # ----------------------------------------------------
        # KNOWLEDGE
        # ----------------------------------------------------

        elif category == "knowledge":

            if question_id == 11:
                tags.append("geography")

            elif question_id == 12:
                tags.append("astronomy")

            elif question_id == 13:
                tags.append("geography")

            elif question_id == 14:
                tags.append("geography")

            elif question_id == 15:
                tags.append("astronomy")

            elif question_id == 16:
                tags.append("chemistry")

            elif question_id == 17:
                tags.append("geography")

            elif question_id == 18:
                tags.append("calendar")

            elif question_id == 19:
                tags.append("biology")

            elif question_id == 20:
                tags.append("physics")

            else:
                tags.append("factual_knowledge")

        # ----------------------------------------------------
        # REASONING
        # ----------------------------------------------------

        elif category == "reasoning":

            if question_id in {22, 25}:
                tags.append("comparison")

            elif question_id in {23, 28}:
                tags.append("temporal_reasoning")

            elif question_id in {24, 27, 29}:
                tags.append("counting")

            elif question_id == 26:
                tags.append("time_calculation")

            elif question_id == 30:
                tags.append("rate_reasoning")

            else:
                tags.append("logical_reasoning")

        # ----------------------------------------------------
        # CODING
        # ----------------------------------------------------

        elif category == "coding":

            if question_id in {31, 32}:
                tags.append("functions")

            elif question_id == 33:
                tags.append("data_structures")

            elif question_id == 34:
                tags.append("syntax")

            elif question_id == 35:
                tags.append("iteration")

            elif question_id == 36:
                tags.append("data_structures")

            elif question_id == 37:
                tags.append("exception_handling")

            elif question_id == 38:
                tags.append("built_in_functions")

            elif question_id == 39:
                tags.append("operators")

            elif question_id == 40:
                tags.append("functions")

            else:
                tags.append("python")

        # ----------------------------------------------------
        # GENERAL
        # ----------------------------------------------------

        elif category == "general":

            if question_id in {
                41,
                42,
                43,
                44,
                45,
                46,
                47,
                49,
                50,
            }:
                tags.append("acronym")

            elif question_id == 48:
                tags.append("operating_system")

            else:
                tags.append("technical_knowledge")

        return tags

    # --------------------------------------------------------
    # ENHANCE DATASET
    # --------------------------------------------------------

    def enhance(
        self,
    ) -> list[dict[str, Any]]:

        questions = self.load()

        enhanced_questions = []

        for question in questions:

            self.validate_question(
                question
            )

            enhanced_question = dict(
                question
            )

            enhanced_question[
                "difficulty"
            ] = self.infer_difficulty(
                question
            )

            enhanced_question[
                "evaluation_type"
            ] = self.infer_evaluation_type(
                question
            )

            enhanced_question[
                "tags"
            ] = self.infer_tags(
                question
            )

            enhanced_questions.append(
                enhanced_question
            )

        return enhanced_questions

    # --------------------------------------------------------
    # SAVE ENHANCED DATASET
    # --------------------------------------------------------

    def save(
        self,
        output_path: str = (
            "datasets/benchmark_enhanced.json"
        ),
    ) -> str:

        enhanced_questions = self.enhance()

        output_file = Path(
            output_path
        )

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            output_file,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                enhanced_questions,
                file,
                indent=2,
                ensure_ascii=False,
            )

        return str(output_file)