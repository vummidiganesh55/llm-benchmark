import re
from difflib import SequenceMatcher


class Evaluator:

    @staticmethod
    def normalize(text: str) -> str:
        """
        Normalize text for comparison.

        Operations:
        - Convert to lowercase
        - Convert common number words to digits
        - Remove punctuation
        - Normalize whitespace
        """

        text = str(text).lower().strip()

        number_words = {
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10"
        }

        for word, digit in number_words.items():
            text = re.sub(
                rf"\b{word}\b",
                digit,
                text
            )

        text = re.sub(
            r"[^\w\s]",
            "",
            text
        )

        text = " ".join(
            text.split()
        )

        return text


    @staticmethod
    def exact_match(
        response: str,
        expected: str
    ) -> float:
        """
        Exact/containment match.

        Returns:
            1.0 -> expected answer is present
            0.0 -> expected answer is absent
        """

        response_normalized = Evaluator.normalize(
            response
        )

        expected_normalized = Evaluator.normalize(
            expected
        )

        if not expected_normalized:
            return 0.0

        return (
            1.0
            if expected_normalized in response_normalized
            else 0.0
        )


    @staticmethod
    def fuzzy_similarity(
        response: str,
        expected: str
    ) -> float:
        """
        Calculate the best fuzzy similarity between
        the expected answer and relevant parts of
        the model response.

        Returns:
            Score between 0.0 and 1.0.
        """

        response_normalized = Evaluator.normalize(
            response
        )

        expected_normalized = Evaluator.normalize(
            expected
        )

        if not response_normalized:
            return 0.0

        if not expected_normalized:
            return 0.0

        # ---------------------------------------------
        # Exact containment
        # ---------------------------------------------

        if expected_normalized in response_normalized:
            return 1.0


        # ---------------------------------------------
        # Token-based comparison
        # ---------------------------------------------

        response_words = (
            response_normalized.split()
        )

        expected_words = (
            expected_normalized.split()
        )

        best_score = 0.0


        # ---------------------------------------------
        # Single-word expected answer
        # ---------------------------------------------

        if len(expected_words) == 1:

            for word in response_words:

                similarity = SequenceMatcher(
                    None,
                    word,
                    expected_normalized
                ).ratio()

                best_score = max(
                    best_score,
                    similarity
                )


        # ---------------------------------------------
        # Multi-word expected answer
        # ---------------------------------------------

        else:

            expected_length = len(
                expected_words
            )

            for i in range(
                len(response_words)
            ):

                window_words = response_words[
                    i:i + expected_length
                ]

                window = " ".join(
                    window_words
                )

                similarity = SequenceMatcher(
                    None,
                    window,
                    expected_normalized
                ).ratio()

                best_score = max(
                    best_score,
                    similarity
                )


        return round(
            best_score,
            4
        )


    @staticmethod
    def fuzzy_match(
        response: str,
        expected: str,
        threshold: float = 0.80
    ) -> float:
        """
        Convert fuzzy similarity into a binary score.

        Returns:
            1.0 -> similarity >= threshold
            0.0 -> similarity < threshold
        """

        similarity = Evaluator.fuzzy_similarity(
            response,
            expected
        )

        return (
            1.0
            if similarity >= threshold
            else 0.0
        )


    @staticmethod
    def keyword_match(
        response: str,
        expected: str
    ) -> float:
        """
        Calculate keyword overlap between the
        expected answer and model response.

        Returns:
            Score between 0.0 and 1.0.
        """

        response_normalized = Evaluator.normalize(
            response
        )

        expected_normalized = Evaluator.normalize(
            expected
        )

        if not expected_normalized:
            return 0.0

        expected_keywords = set(
            expected_normalized.split()
        )

        response_keywords = set(
            response_normalized.split()
        )

        if not expected_keywords:
            return 0.0

        matched_keywords = (
            expected_keywords
            & response_keywords
        )

        return round(
            len(matched_keywords)
            / len(expected_keywords),
            4
        )


    @staticmethod
    def evaluate(
        response: str,
        expected: str
    ) -> dict:
        """
        Run all current evaluation methods.

        Semantic similarity will be added
        in a later stage.
        """

        exact = Evaluator.exact_match(
            response,
            expected
        )

        fuzzy_similarity = (
            Evaluator.fuzzy_similarity(
                response,
                expected
            )
        )

        fuzzy = (
            1.0
            if fuzzy_similarity >= 0.80
            else 0.0
        )

        keyword = Evaluator.keyword_match(
            response,
            expected
        )

        return {
            "exact_match": exact,
            "fuzzy_match": fuzzy,
            "fuzzy_similarity": fuzzy_similarity,
            "keyword_match": keyword
        }