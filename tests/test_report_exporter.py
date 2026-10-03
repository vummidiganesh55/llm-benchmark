import json

from backend.benchmark.report_exporter import (
    ReportExporter,
)


def sample_report():
    return {
        "report_type": "LLM Benchmark Experiment Report",
        "experiment": {
            "experiment_id": "export_test_001",
            "model": "qwen2.5:3b",
            "timestamp": "2026-10-03T15:00:00",
        },
        "summary": {
            "total_questions": 10,
            "successful_questions": 10,
            "failed_questions": 0,
            "accuracy": 0.9,
            "error_rate": 0.0,
        },
        "quality": {
            "accuracy": 0.9,
            "exact_match": 0.8,
            "fuzzy_match": 0.9,
        },
        "reliability": {
            "error_rate": 0.0,
            "retry_rate": 0.1,
        },
        "performance": {
            "average_latency": 2.5,
            "p95_latency": 4.2,
            "tokens_per_second": 20.0,
        },
        "cost": {
            "total_cost_usd": 0.003,
            "cost_per_question_usd": 0.0003,
        },
        "categories": {
            "math": {
                "total": 2,
                "successful": 2,
                "failed": 0,
                "correct": 2,
                "accuracy": 1.0,
            },
            "coding": {
                "total": 2,
                "successful": 2,
                "failed": 0,
                "correct": 1,
                "accuracy": 0.5,
            },
        },
    }


def test_export_json(tmp_path):
    exporter = ReportExporter(
        output_dir=str(tmp_path)
    )

    report = sample_report()

    path = exporter.export_json(
        report,
        "export_test_001",
    )

    file_path = tmp_path / "export_test_001.json"

    assert path == str(file_path)
    assert file_path.exists()

    with file_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        loaded = json.load(file)

    assert loaded == report


def test_export_csv(tmp_path):
    exporter = ReportExporter(
        output_dir=str(tmp_path)
    )

    report = sample_report()

    path = exporter.export_csv(
        report,
        "export_test_001",
    )

    file_path = tmp_path / "export_test_001.csv"

    assert path == str(file_path)
    assert file_path.exists()

    content = file_path.read_text(
        encoding="utf-8"
    )

    assert "metric,value" in content
    assert "summary.accuracy" in content
    assert "0.9" in content
    assert "performance.p95_latency" in content


def test_export_html(tmp_path):
    exporter = ReportExporter(
        output_dir=str(tmp_path)
    )

    report = sample_report()

    path = exporter.export_html(
        report,
        "export_test_001",
    )

    file_path = tmp_path / "export_test_001.html"

    assert path == str(file_path)
    assert file_path.exists()

    content = file_path.read_text(
        encoding="utf-8"
    )

    assert "<!DOCTYPE html>" in content
    assert "LLM Benchmark Experiment Report" in content
    assert "qwen2.5:3b" in content
    assert "Summary" in content
    assert "Quality" in content
    assert "Performance" in content
    assert "Cost" in content
    assert "Category Performance" in content
    assert "math" in content
    assert "coding" in content


def test_export_all(tmp_path):
    exporter = ReportExporter(
        output_dir=str(tmp_path)
    )

    report = sample_report()

    paths = exporter.export_all(
        report,
        "export_test_001",
    )

    assert set(paths.keys()) == {
        "json",
        "csv",
        "html",
    }

    assert (
        tmp_path / "export_test_001.json"
    ).exists()

    assert (
        tmp_path / "export_test_001.csv"
    ).exists()

    assert (
        tmp_path / "export_test_001.html"
    ).exists()


def test_export_uses_report_experiment_id(tmp_path):
    exporter = ReportExporter(
        output_dir=str(tmp_path)
    )

    report = sample_report()

    paths = exporter.export_all(report)

    assert (
        tmp_path / "export_test_001.json"
    ).exists()

    assert (
        tmp_path / "export_test_001.csv"
    ).exists()

    assert (
        tmp_path / "export_test_001.html"
    ).exists()

    assert paths["json"].endswith(
        "export_test_001.json"
    )


def test_html_escapes_values(tmp_path):
    exporter = ReportExporter(
        output_dir=str(tmp_path)
    )

    report = sample_report()

    report["experiment"]["model"] = (
        "<script>alert('test')</script>"
    )

    path = exporter.export_html(
        report,
        "escape_test",
    )

    content = (
        tmp_path / "escape_test.html"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "<script>alert('test')</script>"
        not in content
    )

    assert "&lt;script&gt;" in content