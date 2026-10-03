import json

from backend.benchmark.storage import BenchmarkStorage


def test_save_creates_experiment(tmp_path, monkeypatch):
    results_file = tmp_path / "benchmark_results.json"

    monkeypatch.setattr(
        "backend.benchmark.storage.RESULTS_FILE",
        str(results_file),
    )

    experiment_id = BenchmarkStorage.save(
        model="test-model",
        metrics={
            "accuracy": 0.95,
            "average_latency": 2.5,
        },
        results=[
            {
                "status": "success",
                "response": "Paris",
            }
        ],
    )

    assert experiment_id is not None
    assert len(experiment_id) > 0
    assert results_file.exists()


def test_save_persists_experiment(tmp_path, monkeypatch):
    results_file = tmp_path / "benchmark_results.json"

    monkeypatch.setattr(
        "backend.benchmark.storage.RESULTS_FILE",
        str(results_file),
    )

    experiment_id = BenchmarkStorage.save(
        model="qwen2.5:3b",
        metrics={
            "accuracy": 0.94,
        },
        results=[
            {
                "status": "success",
                "response": "Paris",
            }
        ],
        dataset={
            "version": "1.0.0",
            "question_count": 50,
        },
    )

    with open(results_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    assert len(data) == 1

    experiment = data[0]

    assert experiment["experiment_id"] == experiment_id
    assert experiment["model"] == "qwen2.5:3b"
    assert experiment["metrics"]["accuracy"] == 0.94
    assert experiment["dataset"]["version"] == "1.0.0"
    assert experiment["dataset"]["question_count"] == 50
    assert len(experiment["results"]) == 1


def test_load_all_when_file_does_not_exist(
    tmp_path,
    monkeypatch,
):
    results_file = tmp_path / "missing.json"

    monkeypatch.setattr(
        "backend.benchmark.storage.RESULTS_FILE",
        str(results_file),
    )

    experiments = BenchmarkStorage.load_all()

    assert experiments == []


def test_load_all_returns_saved_experiments(
    tmp_path,
    monkeypatch,
):
    results_file = tmp_path / "benchmark_results.json"

    monkeypatch.setattr(
        "backend.benchmark.storage.RESULTS_FILE",
        str(results_file),
    )

    BenchmarkStorage.save(
        model="model-a",
        metrics={"accuracy": 0.90},
        results=[],
    )

    BenchmarkStorage.save(
        model="model-b",
        metrics={"accuracy": 0.95},
        results=[],
    )

    experiments = BenchmarkStorage.load_all()

    assert len(experiments) == 2
    assert experiments[0]["model"] == "model-a"
    assert experiments[1]["model"] == "model-b"


def test_save_multiple_experiments(
    tmp_path,
    monkeypatch,
):
    results_file = tmp_path / "benchmark_results.json"

    monkeypatch.setattr(
        "backend.benchmark.storage.RESULTS_FILE",
        str(results_file),
    )

    first_id = BenchmarkStorage.save(
        model="model-a",
        metrics={"accuracy": 0.90},
        results=[],
    )

    second_id = BenchmarkStorage.save(
        model="model-b",
        metrics={"accuracy": 0.95},
        results=[],
    )

    experiments = BenchmarkStorage.load_all()

    assert len(experiments) == 2
    assert first_id == experiments[0]["experiment_id"]
    assert second_id == experiments[1]["experiment_id"]


def test_save_preserves_empty_dataset(
    tmp_path,
    monkeypatch,
):
    results_file = tmp_path / "benchmark_results.json"

    monkeypatch.setattr(
        "backend.benchmark.storage.RESULTS_FILE",
        str(results_file),
    )

    BenchmarkStorage.save(
        model="test-model",
        metrics={},
        results=[],
    )

    experiments = BenchmarkStorage.load_all()

    assert len(experiments) == 1
    assert experiments[0]["dataset"] is None


def test_save_preserves_results(
    tmp_path,
    monkeypatch,
):
    results_file = tmp_path / "benchmark_results.json"

    monkeypatch.setattr(
        "backend.benchmark.storage.RESULTS_FILE",
        str(results_file),
    )

    benchmark_results = [
        {
            "status": "success",
            "response": "Paris",
            "latency": 2.1,
        },
        {
            "status": "error",
            "response": None,
            "latency": 0.0,
        },
    ]

    BenchmarkStorage.save(
        model="test-model",
        metrics={"accuracy": 0.5},
        results=benchmark_results,
    )

    experiments = BenchmarkStorage.load_all()

    assert experiments[0]["results"] == benchmark_results