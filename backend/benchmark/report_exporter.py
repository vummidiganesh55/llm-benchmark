import csv
import html
import json
from pathlib import Path
from typing import Any


class ReportExporter:
    """
    Exports benchmark experiment reports to JSON, CSV,
    and HTML formats.
    """

    def __init__(
        self,
        output_dir: str = "results/reports",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _safe_filename(
        self,
        experiment_id: str | None,
    ) -> str:
        return (
            experiment_id
            if experiment_id
            else "benchmark_report"
        )

    def export_json(
        self,
        report: dict[str, Any],
        experiment_id: str | None = None,
    ) -> str:

        filename = (
            self._safe_filename(experiment_id)
            + ".json"
        )

        path = self.output_dir / filename

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                report,
                file,
                indent=4,
                ensure_ascii=False,
            )

        return str(path)

    def export_csv(
        self,
        report: dict[str, Any],
        experiment_id: str | None = None,
    ) -> str:

        filename = (
            self._safe_filename(experiment_id)
            + ".csv"
        )

        path = self.output_dir / filename

        rows: list[dict[str, Any]] = []

        self._flatten_dict(
            report,
            rows,
        )

        if not rows:
            rows = [
                {
                    "metric": "report",
                    "value": "",
                }
            ]

        fieldnames = [
            "metric",
            "value",
        ]

        with path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for row in rows:
                writer.writerow(row)

        return str(path)

    def export_html(
        self,
        report: dict[str, Any],
        experiment_id: str | None = None,
    ) -> str:

        filename = (
            self._safe_filename(experiment_id)
            + ".html"
        )

        path = self.output_dir / filename

        experiment = report.get(
            "experiment",
            {},
        )

        summary = report.get(
            "summary",
            {},
        )

        quality = report.get(
            "quality",
            {},
        )

        reliability = report.get(
            "reliability",
            {},
        )

        performance = report.get(
            "performance",
            {},
        )

        cost = report.get(
            "cost",
            {},
        )

        categories = report.get(
            "categories",
            {},
        )

        html_content = self._build_html(
            experiment=experiment,
            summary=summary,
            quality=quality,
            reliability=reliability,
            performance=performance,
            cost=cost,
            categories=categories,
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:

            file.write(html_content)

        return str(path)

    def export_all(
        self,
        report: dict[str, Any],
        experiment_id: str | None = None,
    ) -> dict[str, str]:

        resolved_id = (
            experiment_id
            or report.get(
                "experiment",
                {},
            ).get("experiment_id")
            or "benchmark_report"
        )

        return {
            "json": self.export_json(
                report,
                resolved_id,
            ),
            "csv": self.export_csv(
                report,
                resolved_id,
            ),
            "html": self.export_html(
                report,
                resolved_id,
            ),
        }

    def _flatten_dict(
        self,
        data: Any,
        rows: list[dict[str, Any]],
        prefix: str = "",
    ) -> None:

        if isinstance(data, dict):

            for key, value in data.items():

                current_key = (
                    f"{prefix}.{key}"
                    if prefix
                    else str(key)
                )

                self._flatten_dict(
                    value,
                    rows,
                    current_key,
                )

            return

        if isinstance(data, list):

            for index, value in enumerate(data):

                current_key = (
                    f"{prefix}.{index}"
                    if prefix
                    else str(index)
                )

                self._flatten_dict(
                    value,
                    rows,
                    current_key,
                )

            return

        rows.append(
            {
                "metric": prefix,
                "value": data,
            }
        )

    def _build_html(
        self,
        experiment: dict[str, Any],
        summary: dict[str, Any],
        quality: dict[str, Any],
        reliability: dict[str, Any],
        performance: dict[str, Any],
        cost: dict[str, Any],
        categories: dict[str, Any],
    ) -> str:

        def esc(value: Any) -> str:
            return html.escape(
                str(value)
            )

        def table(
            data: dict[str, Any],
        ) -> str:

            rows = []

            for key, value in data.items():

                rows.append(
                    f"""
                    <tr>
                        <td>{esc(key)}</td>
                        <td>{esc(value)}</td>
                    </tr>
                    """
                )

            return (
                "<table>"
                "<thead>"
                "<tr>"
                "<th>Metric</th>"
                "<th>Value</th>"
                "</tr>"
                "</thead>"
                "<tbody>"
                + "".join(rows)
                + "</tbody>"
                "</table>"
            )

        category_rows = []

        for category, values in categories.items():

            if not isinstance(values, dict):
                continue

            category_rows.append(
                f"""
                <tr>
                    <td>{esc(category)}</td>
                    <td>{esc(values.get("total", 0))}</td>
                    <td>{esc(values.get("successful", 0))}</td>
                    <td>{esc(values.get("failed", 0))}</td>
                    <td>{esc(values.get("correct", 0))}</td>
                    <td>{esc(values.get("accuracy", 0))}</td>
                </tr>
                """
            )

        category_table = (
            "<table>"
            "<thead>"
            "<tr>"
            "<th>Category</th>"
            "<th>Total</th>"
            "<th>Successful</th>"
            "<th>Failed</th>"
            "<th>Correct</th>"
            "<th>Accuracy</th>"
            "</tr>"
            "</thead>"
            "<tbody>"
            + "".join(category_rows)
            + "</tbody>"
            "</table>"
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>LLM Benchmark Report</title>

<style>
body {{
    font-family: Arial, sans-serif;
    margin: 40px;
    line-height: 1.5;
}}

h1 {{
    margin-bottom: 5px;
}}

h2 {{
    margin-top: 35px;
}}

table {{
    border-collapse: collapse;
    width: 100%;
    margin-top: 15px;
}}

th,
td {{
    border: 1px solid #ddd;
    padding: 10px;
    text-align: left;
}}

th {{
    font-weight: bold;
}}

.section {{
    margin-bottom: 30px;
}}

.metadata {{
    padding: 15px;
    border: 1px solid #ddd;
}}
</style>
</head>

<body>

<h1>LLM Benchmark Experiment Report</h1>

<div class="metadata">
    <p>
        <strong>Experiment ID:</strong>
        {esc(experiment.get("experiment_id"))}
    </p>

    <p>
        <strong>Model:</strong>
        {esc(experiment.get("model"))}
    </p>

    <p>
        <strong>Timestamp:</strong>
        {esc(experiment.get("timestamp"))}
    </p>
</div>

<div class="section">

<h2>Summary</h2>

{table(summary)}

</div>

<div class="section">

<h2>Quality</h2>

{table(quality)}

</div>

<div class="section">

<h2>Reliability</h2>

{table(reliability)}

</div>

<div class="section">

<h2>Performance</h2>

{table(performance)}

</div>

<div class="section">

<h2>Cost</h2>

{table(cost)}

</div>

<div class="section">

<h2>Category Performance</h2>

{category_table}

</div>

</body>
</html>
"""