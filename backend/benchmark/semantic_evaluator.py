import math
from typing import Any, Callable

import requests

from backend.benchmark.embedding_cache import EmbeddingCache


class SemanticEvaluator:
    """
    Semantic similarity evaluator using Ollama embeddings.

    Embeddings are cached so repeated evaluation of the same
    model + text does not require another Ollama embedding request.
    """

    def __init__(
        self,
        model: str = "nomic-embed-text:latest",
        base_url: str = "http://localhost:11434",
        cache: EmbeddingCache | None = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.cache = cache or EmbeddingCache()

    def _request_embedding(self, text: str) -> list[float]:
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={
                "model": self.model,
                "prompt": text,
            },
            timeout=120,
        )

        response.raise_for_status()

        data: dict[str, Any] = response.json()
        embedding = data.get("embedding")

        if not embedding:
            raise ValueError("Ollama returned an empty embedding.")

        return [float(value) for value in embedding]

    def get_embedding(self, text: str) -> list[float]:
        """
        Get an embedding from cache or Ollama.
        """
        return self.cache.get_or_compute(
            model=self.model,
            text=text,
            compute_function=lambda: self._request_embedding(text),
        )

    @staticmethod
    def cosine_similarity(
        vector_a: list[float],
        vector_b: list[float],
    ) -> float:
        """
        Calculate cosine similarity between two vectors.
        """

        if len(vector_a) != len(vector_b):
            raise ValueError(
                "Embedding vectors must have the same dimension."
            )

        if not vector_a or not vector_b:
            return 0.0

        dot_product = sum(
            a * b
            for a, b in zip(vector_a, vector_b)
        )

        magnitude_a = math.sqrt(
            sum(a * a for a in vector_a)
        )

        magnitude_b = math.sqrt(
            sum(b * b for b in vector_b)
        )

        if magnitude_a == 0.0 or magnitude_b == 0.0:
            return 0.0

        similarity = dot_product / (magnitude_a * magnitude_b)

        return round(similarity, 4)

    def evaluate(
        self,
        response: str,
        expected: str,
    ) -> dict[str, Any]:
        """
        Compare model response with expected answer.
        """

        response_embedding = self.get_embedding(response)
        expected_embedding = self.get_embedding(expected)

        similarity = self.cosine_similarity(
            response_embedding,
            expected_embedding,
        )

        return {
            "semantic_similarity": similarity,
            "embedding_model": self.model,
            "cache": self.cache.get_stats(),
        }

    def evaluate_with_threshold(
        self,
        response: str,
        expected: str,
        threshold: float = 0.75,
    ) -> dict[str, Any]:
        """
        Evaluate semantic similarity and determine whether
        the similarity exceeds the configured threshold.
        """

        result = self.evaluate(
            response=response,
            expected=expected,
        )

        similarity = result["semantic_similarity"]

        result["semantic_match"] = (
            1.0 if similarity >= threshold else 0.0
        )

        result["threshold"] = threshold

        return result

    def get_cache_stats(self) -> dict[str, Any]:
        """
        Return embedding cache statistics.
        """

        return self.cache.get_stats()

    def clear_cache(self) -> None:
        """
        Clear the embedding cache.
        """

        self.cache.clear()