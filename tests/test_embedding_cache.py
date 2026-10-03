from pathlib import Path

from backend.benchmark.embedding_cache import (
    EmbeddingCache,
)


def test_cache_miss(tmp_path: Path):

    cache = EmbeddingCache(
        cache_path=str(
            tmp_path / "embedding_cache.json"
        )
    )

    result = cache.get(
        model="nomic-embed-text:latest",
        text="Hello world",
    )

    assert result is None

    stats = cache.get_stats()

    assert stats["hits"] == 0
    assert stats["misses"] == 1
    assert stats["entries"] == 0


def test_cache_set_and_get(tmp_path: Path):

    cache = EmbeddingCache(
        cache_path=str(
            tmp_path / "embedding_cache.json"
        )
    )

    embedding = [
        0.1,
        0.2,
        0.3,
    ]

    cache.set(
        model="nomic-embed-text:latest",
        text="Hello world",
        embedding=embedding,
    )

    result = cache.get(
        model="nomic-embed-text:latest",
        text="Hello world",
    )

    assert result == embedding

    stats = cache.get_stats()

    assert stats["hits"] == 1
    assert stats["misses"] == 0
    assert stats["entries"] == 1


def test_cache_persists_to_disk(tmp_path: Path):

    cache_path = tmp_path / "embedding_cache.json"

    cache1 = EmbeddingCache(
        cache_path=str(cache_path)
    )

    embedding = [
        0.5,
        0.6,
        0.7,
    ]

    cache1.set(
        model="nomic-embed-text:latest",
        text="Paris is the capital of France.",
        embedding=embedding,
    )

    cache2 = EmbeddingCache(
        cache_path=str(cache_path)
    )

    result = cache2.get(
        model="nomic-embed-text:latest",
        text="Paris is the capital of France.",
    )

    assert result == embedding


def test_different_models_do_not_share_cache(
    tmp_path: Path,
):

    cache = EmbeddingCache(
        cache_path=str(
            tmp_path / "embedding_cache.json"
        )
    )

    embedding = [
        0.1,
        0.2,
    ]

    cache.set(
        model="model-a",
        text="same text",
        embedding=embedding,
    )

    result = cache.get(
        model="model-b",
        text="same text",
    )

    assert result is None


def test_get_or_compute_only_computes_once(
    tmp_path: Path,
):

    cache = EmbeddingCache(
        cache_path=str(
            tmp_path / "embedding_cache.json"
        )
    )

    calls = {
        "count": 0
    }

    def compute_embedding():

        calls["count"] += 1

        return [
            0.1,
            0.2,
            0.3,
        ]

    first = cache.get_or_compute(
        model="nomic-embed-text:latest",
        text="test text",
        compute_function=compute_embedding,
    )

    second = cache.get_or_compute(
        model="nomic-embed-text:latest",
        text="test text",
        compute_function=compute_embedding,
    )

    assert first == second

    assert calls["count"] == 1

    stats = cache.get_stats()

    assert stats["misses"] == 1
    assert stats["hits"] == 1
    assert stats["entries"] == 1


def test_cache_clear(tmp_path: Path):

    cache_path = tmp_path / "embedding_cache.json"

    cache = EmbeddingCache(
        cache_path=str(cache_path)
    )

    cache.set(
        model="nomic-embed-text:latest",
        text="test",
        embedding=[0.1, 0.2],
    )

    assert cache_path.exists()

    cache.clear()

    assert not cache_path.exists()

    stats = cache.get_stats()

    assert stats["entries"] == 0
    assert stats["hits"] == 0
    assert stats["misses"] == 0