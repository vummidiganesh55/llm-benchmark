import hashlib
import json

import pytest

from backend.benchmark.dataset_version import DatasetVersion


def test_calculate_hash(tmp_path):
    dataset_file = tmp_path / "benchmark.json"
    metadata_file = tmp_path / "metadata.json"

    dataset_content = '[{"id": 1, "question": "What is 2+2?"}]'
    dataset_file.write_text(
        dataset_content,
        encoding="utf-8",
    )

    metadata_file.write_text(
        json.dumps(
            {
                "name": "Test Dataset",
                "version": "1.0.0",
                "description": "Test dataset",
                "file": "benchmark.json",
                "question_count": 1,
                "categories": {
                    "math": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    version = DatasetVersion(
        dataset_path=dataset_file,
        metadata_path=metadata_file,
    )

    actual_hash = version.calculate_hash()

    expected_hash = hashlib.sha256(
        dataset_content.encode("utf-8")
    ).hexdigest()

    assert actual_hash == expected_hash


def test_load_metadata(tmp_path):
    dataset_file = tmp_path / "benchmark.json"
    metadata_file = tmp_path / "metadata.json"

    dataset_file.write_text(
        "[]",
        encoding="utf-8",
    )

    metadata = {
        "name": "Test Dataset",
        "version": "1.0.0",
        "description": "Test description",
        "file": "benchmark.json",
        "question_count": 10,
        "categories": {
            "math": 5,
            "coding": 5,
        },
    }

    metadata_file.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    version = DatasetVersion(
        dataset_path=dataset_file,
        metadata_path=metadata_file,
    )

    result = version.load_metadata()

    assert result == metadata


def test_get_info(tmp_path):
    dataset_file = tmp_path / "benchmark.json"
    metadata_file = tmp_path / "metadata.json"

    dataset_content = '[{"id": 1}]'

    dataset_file.write_text(
        dataset_content,
        encoding="utf-8",
    )

    metadata = {
        "name": "Test Dataset",
        "version": "1.0.0",
        "description": "Benchmark dataset",
        "file": "benchmark.json",
        "question_count": 1,
        "categories": {
            "general": 1,
        },
    }

    metadata_file.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    version = DatasetVersion(
        dataset_path=dataset_file,
        metadata_path=metadata_file,
    )

    info = version.get_info()

    expected_hash = hashlib.sha256(
        dataset_content.encode("utf-8")
    ).hexdigest()

    assert info["name"] == "Test Dataset"
    assert info["version"] == "1.0.0"
    assert info["description"] == "Benchmark dataset"
    assert info["file"] == "benchmark.json"
    assert info["question_count"] == 1
    assert info["categories"] == {"general": 1}
    assert info["sha256"] == expected_hash


def test_missing_dataset_raises_error(tmp_path):
    dataset_file = tmp_path / "missing.json"
    metadata_file = tmp_path / "metadata.json"

    metadata_file.write_text(
        json.dumps(
            {
                "name": "Test Dataset",
                "version": "1.0.0",
            }
        ),
        encoding="utf-8",
    )

    version = DatasetVersion(
        dataset_path=dataset_file,
        metadata_path=metadata_file,
    )

    with pytest.raises(FileNotFoundError):
        version.calculate_hash()


def test_missing_metadata_raises_error(tmp_path):
    dataset_file = tmp_path / "benchmark.json"
    metadata_file = tmp_path / "missing.json"

    dataset_file.write_text(
        "[]",
        encoding="utf-8",
    )

    version = DatasetVersion(
        dataset_path=dataset_file,
        metadata_path=metadata_file,
    )

    with pytest.raises(FileNotFoundError):
        version.load_metadata()


def test_hash_changes_when_dataset_changes(tmp_path):
    dataset_file = tmp_path / "benchmark.json"
    metadata_file = tmp_path / "metadata.json"

    dataset_file.write_text(
        '[{"id": 1}]',
        encoding="utf-8",
    )

    metadata_file.write_text(
        json.dumps(
            {
                "name": "Test",
                "version": "1.0.0",
            }
        ),
        encoding="utf-8",
    )

    version = DatasetVersion(
        dataset_path=dataset_file,
        metadata_path=metadata_file,
    )

    first_hash = version.calculate_hash()

    dataset_file.write_text(
        '[{"id": 1}, {"id": 2}]',
        encoding="utf-8",
    )

    second_hash = version.calculate_hash()

    assert first_hash != second_hash
    