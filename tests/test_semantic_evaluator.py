import pytest

from backend.benchmark.semantic_evaluator import SemanticEvaluator


def test_cosine_similarity_identical_vectors():
    result = SemanticEvaluator.cosine_similarity(
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    )

    assert result == 1.0


def test_cosine_similarity_orthogonal_vectors():
    result = SemanticEvaluator.cosine_similarity(
        [1.0, 0.0],
        [0.0, 1.0],
    )

    assert result == 0.0


def test_cosine_similarity_opposite_vectors():
    result = SemanticEvaluator.cosine_similarity(
        [1.0, 0.0],
        [-1.0, 0.0],
    )

    assert result == -1.0


def test_cosine_similarity_different_dimensions():
    with pytest.raises(ValueError):
        SemanticEvaluator.cosine_similarity(
            [1.0, 0.0],
            [1.0, 0.0, 0.0],
        )


def test_cosine_similarity_empty_vectors():
    result = SemanticEvaluator.cosine_similarity([], [])

    assert result == 0.0


def test_cosine_similarity_zero_vector():
    result = SemanticEvaluator.cosine_similarity(
        [0.0, 0.0],
        [1.0, 2.0],
    )

    assert result == 0.0


def test_evaluate_with_threshold_match(monkeypatch):
    evaluator = SemanticEvaluator()

    monkeypatch.setattr(
        evaluator,
        "evaluate",
        lambda response, expected: {
            "semantic_similarity": 0.90,
            "embedding_model": "nomic-embed-text:latest",
            "cache": {},
        },
    )

    result = evaluator.evaluate_with_threshold(
        response="Paris",
        expected="Paris",
        threshold=0.75,
    )

    assert result["semantic_similarity"] == 0.90
    assert result["semantic_match"] == 1.0
    assert result["threshold"] == 0.75


def test_evaluate_with_threshold_no_match(monkeypatch):
    evaluator = SemanticEvaluator()

    monkeypatch.setattr(
        evaluator,
        "evaluate",
        lambda response, expected: {
            "semantic_similarity": 0.60,
            "embedding_model": "nomic-embed-text:latest",
            "cache": {},
        },
    )

    result = evaluator.evaluate_with_threshold(
        response="Berlin",
        expected="Paris",
        threshold=0.75,
    )

    assert result["semantic_similarity"] == 0.60
    assert result["semantic_match"] == 0.0
    assert result["threshold"] == 0.75


def test_get_embedding_uses_cache(tmp_path, monkeypatch):
    from backend.benchmark.embedding_cache import EmbeddingCache

    cache = EmbeddingCache(
        cache_path=tmp_path / "embedding_cache.json"
    )

    evaluator = SemanticEvaluator(cache=cache)

    calls = {"count": 0}

    def fake_request_embedding(text):
        calls["count"] += 1
        return [1.0, 2.0, 3.0]

    monkeypatch.setattr(
        evaluator,
        "_request_embedding",
        fake_request_embedding,
    )

    first = evaluator.get_embedding(
        "Paris is the capital of France."
    )

    second = evaluator.get_embedding(
        "Paris is the capital of France."
    )

    assert first == [1.0, 2.0, 3.0]
    assert second == [1.0, 2.0, 3.0]

    # First request computes the embedding.
    # Second request retrieves it from cache.
    assert calls["count"] == 1

    stats = evaluator.get_cache_stats()

    assert stats["entries"] == 1
    assert stats["hits"] == 1
    assert stats["misses"] == 1


def test_clear_cache(tmp_path):
    from backend.benchmark.embedding_cache import EmbeddingCache

    cache = EmbeddingCache(
        cache_path=tmp_path / "embedding_cache.json"
    )

    evaluator = SemanticEvaluator(cache=cache)

    evaluator.cache.set(
        model="test-model",
        text="hello",
        embedding=[1.0, 2.0, 3.0],
    )

    assert evaluator.cache.get_stats()["entries"] == 1

    evaluator.clear_cache()

    assert evaluator.cache.get_stats()["entries"] == 0
    assert evaluator.cache.get_stats()["hits"] == 0
    assert evaluator.cache.get_stats()["misses"] == 0