from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

RESULTS_FILE = Path("results/benchmark_results.json")
DATASET_FILE = Path("datasets/benchmark.json")
DATASET_METADATA_FILE = Path("datasets/metadata.json")
EMBEDDING_CACHE_FILE = Path("results/embedding_cache.json")


st.set_page_config(
    page_title="LLM Benchmarking Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .main {
            padding-top: 1rem;
        }

        .section-title {
            font-size: 1.4rem;
            font-weight: 700;
            margin-top: 1rem;
        }

        .small-text {
            font-size: 0.85rem;
            opacity: 0.75;
        }

        [data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.20);
            border-radius: 10px;
            padding: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_results() -> list[dict[str, Any]]:
    if not RESULTS_FILE.exists():
        return []

    try:
        with open(
            RESULTS_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        if isinstance(data, list):
            return data

    except (OSError, json.JSONDecodeError):
        pass

    return []


@st.cache_data
def load_dataset_metadata() -> dict[str, Any]:
    if not DATASET_METADATA_FILE.exists():
        return {}

    try:
        with open(
            DATASET_METADATA_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        if isinstance(data, dict):
            return data

    except (OSError, json.JSONDecodeError):
        pass

    return {}


@st.cache_data
def load_dataset() -> list[dict[str, Any]]:
    if not DATASET_FILE.exists():
        return []

    try:
        with open(
            DATASET_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        if isinstance(data, list):
            return data

    except (OSError, json.JSONDecodeError):
        pass

    return []


def load_embedding_cache_stats() -> dict[str, Any]:
    if not EMBEDDING_CACHE_FILE.exists():
        return {
            "entries": 0,
            "hits": 0,
            "misses": 0,
            "total_requests": 0,
            "hit_rate": 0.0,
        }

    try:
        with open(
            EMBEDDING_CACHE_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        embeddings = data.get(
            "embeddings",
            {},
        )

        if not isinstance(embeddings, dict):
            embeddings = {}

        return {
            "entries": len(embeddings),
            "hits": 0,
            "misses": 0,
            "total_requests": 0,
            "hit_rate": 0.0,
        }

    except (OSError, json.JSONDecodeError):
        return {
            "entries": 0,
            "hits": 0,
            "misses": 0,
            "total_requests": 0,
            "hit_rate": 0.0,
        }


experiments = load_results()
dataset_metadata = load_dataset_metadata()
dataset_questions = load_dataset()


# ============================================================
# SAFE HELPERS
# ============================================================

def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        if value is None:
            return default

        return float(value)

    except (TypeError, ValueError):
        return default


def safe_int(
    value: Any,
    default: int = 0,
) -> int:
    try:
        if value is None:
            return default

        return int(value)

    except (TypeError, ValueError):
        return default


def get_metric(
    metrics: Any,
    *keys: str,
    default: float = 0.0,
) -> float:

    if not isinstance(metrics, dict):
        return default

    for key in keys:

        if key in metrics:

            value = metrics.get(key)

            if value is not None:
                return safe_float(
                    value,
                    default,
                )

    return default


def calculate_average(
    values: list[float],
) -> float:

    if not values:
        return 0.0

    return sum(values) / len(values)


def format_percentage(
    value: Any,
) -> str:

    return f"{safe_float(value):.2%}"


def format_seconds(
    value: Any,
) -> str:

    return f"{safe_float(value):.2f}s"


def format_cost(
    value: Any,
) -> str:

    return f"${safe_float(value):.6f}"


# ============================================================
# TOKEN COMPATIBILITY
# ============================================================

def extract_tokens(
    result: dict[str, Any],
) -> tuple[int, int, int]:
    """
    Supports multiple historical benchmark schemas.

    Supported formats:

    1. tokens = {
           "input": 10,
           "output": 20,
           "total": 30
       }

    2. tokens = 30

    3. Direct fields:

       input_tokens
       output_tokens
       total_tokens
    """

    tokens = result.get(
        "tokens",
        {},
    )

    # --------------------------------------------------------
    # New schema
    # --------------------------------------------------------

    if isinstance(tokens, dict):

        input_tokens = safe_int(
            tokens.get("input")
        )

        output_tokens = safe_int(
            tokens.get("output")
        )

        total_tokens = safe_int(
            tokens.get("total")
        )

        return (
            input_tokens,
            output_tokens,
            total_tokens,
        )

    # --------------------------------------------------------
    # Historical schema where tokens is an integer
    # --------------------------------------------------------

    if isinstance(
        tokens,
        (int, float),
    ):

        input_tokens = safe_int(
            result.get(
                "input_tokens"
            )
        )

        output_tokens = safe_int(
            result.get(
                "output_tokens"
            )
        )

        total_tokens = safe_int(
            tokens
        )

        return (
            input_tokens,
            output_tokens,
            total_tokens,
        )

    # --------------------------------------------------------
    # Direct token fields
    # --------------------------------------------------------

    input_tokens = safe_int(
        result.get(
            "input_tokens"
        )
    )

    output_tokens = safe_int(
        result.get(
            "output_tokens"
        )
    )

    total_tokens = safe_int(
        result.get(
            "total_tokens"
        )
    )

    return (
        input_tokens,
        output_tokens,
        total_tokens,
    )


# ============================================================
# QUESTION COUNT
# ============================================================

def get_questions_run(
    experiment: dict[str, Any],
) -> int:

    metrics = experiment.get(
        "metrics",
        {},
    )

    if isinstance(metrics, dict):

        questions_run = metrics.get(
            "questions_run"
        )

        if questions_run is not None:
            return safe_int(
                questions_run
            )

    results = experiment.get(
        "results",
        [],
    )

    if isinstance(results, list):
        return len(results)

    return 0


# ============================================================
# VALID EXPERIMENTS
# ============================================================

def valid_experiments(
    data: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    valid = []

    for experiment in data:

        if not isinstance(
            experiment,
            dict,
        ):
            continue

        experiment_id = experiment.get(
            "experiment_id"
        )

        model = experiment.get(
            "model"
        )

        metrics = experiment.get(
            "metrics"
        )

        if not experiment_id:
            continue

        if not model:
            continue

        if not isinstance(
            metrics,
            dict,
        ):
            continue

        if not metrics:
            continue

        valid.append(
            experiment
        )

    return valid


valid_runs = valid_experiments(
    experiments
)


# ============================================================
# FAILURE EXTRACTION
# ============================================================

def extract_result_failures(
    experiment: dict[str, Any],
) -> list[dict[str, Any]]:

    failures = []

    results = experiment.get(
        "results",
        [],
    )

    if not isinstance(
        results,
        list,
    ):
        return failures

    for result in results:

        if not isinstance(
            result,
            dict,
        ):
            continue

        status = result.get(
            "status",
            "success",
        )

        if status == "success":
            continue

        failures.append(
            {
                "experiment_id": experiment.get(
                    "experiment_id"
                ),
                "model": experiment.get(
                    "model"
                ),
                "question_id": result.get(
                    "id"
                ),
                "category": result.get(
                    "category",
                    "unknown",
                ),
                "question": result.get(
                    "question",
                    result.get(
                        "prompt",
                        "",
                    ),
                ),
                "error": result.get(
                    "error"
                ),
                "failure_type": result.get(
                    "failure_type",
                    "unknown",
                ),
            }
        )

    return failures


# ============================================================
# HEADER
# ============================================================

st.title(
    "🤖 LLM Benchmarking Platform"
)

st.markdown(
    """
    **Multi-provider LLM evaluation platform**

    Evaluate LLMs across:

    **Quality • Latency • Throughput • Cost • Reliability •
    Semantic Similarity • LLM-as-a-Judge • Statistics •
    Failure Analysis**
    """
)


# ============================================================
# NO DATA
# ============================================================

if not valid_runs:

    st.warning(
        "No valid benchmark experiments found in "
        "`results/benchmark_results.json`."
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ Dashboard Controls"
)

models = sorted(
    {
        experiment.get(
            "model"
        )
        for experiment in valid_runs
        if experiment.get(
            "model"
        )
    }
)

selected_model = st.sidebar.selectbox(
    "Model",
    ["All Models"] + models,
)


if st.sidebar.button(
    "🔄 Refresh Results"
):

    st.cache_data.clear()

    st.rerun()


# ============================================================
# FILTER
# ============================================================

if selected_model == "All Models":

    filtered_runs = valid_runs

else:

    filtered_runs = [
        experiment
        for experiment in valid_runs
        if experiment.get(
            "model"
        )
        == selected_model
    ]


# ============================================================
# SIDEBAR DATA
# ============================================================

st.sidebar.divider()

st.sidebar.subheader(
    "🧠 Embedding Cache"
)

cache_stats = load_embedding_cache_stats()

st.sidebar.metric(
    "Cached Embeddings",
    cache_stats.get(
        "entries",
        0,
    ),
)

st.sidebar.caption(
    "Persistent cache: "
    "`results/embedding_cache.json`"
)


# ============================================================
# TABS
# ============================================================

(
    overview_tab,
    compare_tab,
    quality_tab,
    category_tab,
    performance_tab,
    cost_tab,
    reliability_tab,
    failure_tab,
    statistics_tab,
    experiments_tab,
    explorer_tab,
) = st.tabs(
    [
        "📊 Overview",
        "🏆 Model Comparison",
        "🎯 Quality",
        "📚 Categories",
        "⚡ Performance",
        "💰 Cost",
        "🔄 Reliability",
        "🚨 Failures",
        "📈 Statistics",
        "🧪 Experiments",
        "🔎 Question Explorer",
    ]
)


# ============================================================
# TAB 1 — OVERVIEW
# ============================================================

with overview_tab:

    st.header(
        "📊 Benchmark Overview"
    )

    total_experiments = len(
        filtered_runs
    )

    models_tested = len(
        {
            experiment.get(
                "model"
            )
            for experiment in filtered_runs
        }
    )

    total_questions = sum(
        get_questions_run(
            experiment
        )
        for experiment in filtered_runs
    )

    accuracies = [
        get_metric(
            experiment.get(
                "metrics",
                {},
            ),
            "accuracy",
        )
        for experiment in filtered_runs
    ]

    latencies = [
        get_metric(
            experiment.get(
                "metrics",
                {},
            ),
            "average_latency",
        )
        for experiment in filtered_runs
    ]

    total_cost = sum(
        get_metric(
            experiment.get(
                "metrics",
                {},
            ),
            "total_cost",
        )
        for experiment in filtered_runs
    )

    avg_accuracy = calculate_average(
        accuracies
    )

    avg_latency = calculate_average(
        latencies
    )

    col1, col2, col3, col4, col5, col6 = st.columns(
        6
    )

    col1.metric(
        "Experiments",
        total_experiments,
    )

    col2.metric(
        "Models",
        models_tested,
    )

    col3.metric(
        "Questions",
        total_questions,
    )

    col4.metric(
        "Avg Accuracy",
        format_percentage(
            avg_accuracy
        ),
    )

    col5.metric(
        "Avg Latency",
        format_seconds(
            avg_latency
        ),
    )

    col6.metric(
        "Recorded Cost",
        format_cost(
            total_cost
        ),
    )

    st.divider()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    st.subheader(
        "📚 Dataset"
    )

    dataset_col1, dataset_col2, dataset_col3, dataset_col4 = (
        st.columns(4)
    )

    dataset_col1.metric(
        "Dataset Version",
        dataset_metadata.get(
            "version",
            "Unknown",
        ),
    )

    dataset_col2.metric(
        "Questions",
        dataset_metadata.get(
            "question_count",
            len(dataset_questions),
        ),
    )

    dataset_col3.metric(
        "Categories",
        len(
            dataset_metadata.get(
                "categories",
                {},
            )
        ),
    )

    dataset_col4.metric(
        "Dataset",
        dataset_metadata.get(
            "name",
            "Unknown",
        ),
    )

    if dataset_metadata:

        with st.expander(
            "🔐 Dataset Metadata"
        ):

            st.json(
                dataset_metadata
            )

    st.divider()

    # --------------------------------------------------------
    # Latest experiments
    # --------------------------------------------------------

    st.subheader(
        "🧪 Latest Experiments"
    )

    latest_runs = sorted(
        filtered_runs,
        key=lambda item: item.get(
            "timestamp",
            "",
        ),
        reverse=True,
    )[:10]

    latest_rows = []

    for experiment in latest_runs:

        metrics = experiment.get(
            "metrics",
            {},
        )

        latest_rows.append(
            {
                "Experiment": experiment.get(
                    "experiment_id"
                ),
                "Model": experiment.get(
                    "model"
                ),
                "Questions": get_questions_run(
                    experiment
                ),
                "Accuracy": format_percentage(
                    metrics.get(
                        "accuracy"
                    )
                ),
                "Avg Latency": format_seconds(
                    metrics.get(
                        "average_latency"
                    )
                ),
                "Timestamp": experiment.get(
                    "timestamp"
                ),
            }
        )

    if latest_rows:

        st.dataframe(
            pd.DataFrame(
                latest_rows
            ),
            width="stretch",
            hide_index=True,
        )


# ============================================================
# TAB 2 — MODEL COMPARISON
# ============================================================

with compare_tab:

    st.header(
        "🏆 Model Comparison"
    )

    comparison_rows = []

    for experiment in filtered_runs:

        metrics = experiment.get(
            "metrics",
            {},
        )

        comparison_rows.append(
            {
                "Model": experiment.get(
                    "model"
                ),
                "Experiment": experiment.get(
                    "experiment_id"
                ),
                "Accuracy": get_metric(
                    metrics,
                    "accuracy",
                ),
                "Avg Latency (s)": get_metric(
                    metrics,
                    "average_latency",
                ),
                "P50 (s)": get_metric(
                    metrics,
                    "p50_latency",
                ),
                "P95 (s)": get_metric(
                    metrics,
                    "p95_latency",
                ),
                "Tokens/sec": get_metric(
                    metrics,
                    "tokens_per_second",
                ),
                "Cost ($)": get_metric(
                    metrics,
                    "total_cost",
                ),
                "Error Rate": get_metric(
                    metrics,
                    "error_rate",
                ),
            }
        )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    if comparison_df.empty:

        st.info(
            "No comparison data available."
        )

    else:

        st.dataframe(
            comparison_df,
            width="stretch",
            hide_index=True,
        )

        st.subheader(
            "Accuracy by Model"
        )

        accuracy_chart = (
            comparison_df
            .groupby("Model")[
                "Accuracy"
            ]
            .mean()
            .sort_values(
                ascending=False
            )
        )

        st.bar_chart(
            accuracy_chart
        )

        st.subheader(
            "Average Latency"
        )

        latency_chart = (
            comparison_df
            .groupby("Model")[
                "Avg Latency (s)"
            ]
            .mean()
            .sort_values()
        )

        st.bar_chart(
            latency_chart
        )

        st.subheader(
            "Throughput"
        )

        throughput_chart = (
            comparison_df
            .groupby("Model")[
                "Tokens/sec"
            ]
            .mean()
            .sort_values(
                ascending=False
            )
        )

        st.bar_chart(
            throughput_chart
        )


# ============================================================
# TAB 3 — QUALITY
# ============================================================

with quality_tab:

    st.header(
        "🎯 Quality Evaluation"
    )

    quality_rows = []

    for experiment in filtered_runs:

        metrics = experiment.get(
            "metrics",
            {},
        )

        quality_rows.append(
            {
                "Model": experiment.get(
                    "model"
                ),
                "Accuracy": get_metric(
                    metrics,
                    "accuracy",
                ),
                "Exact Match": get_metric(
                    metrics,
                    "exact_match",
                ),
                "Fuzzy Match": get_metric(
                    metrics,
                    "fuzzy_match",
                ),
                "Keyword Match": get_metric(
                    metrics,
                    "keyword_match",
                ),
                "Semantic Match": get_metric(
                    metrics,
                    "semantic_match",
                ),
                "Semantic Similarity": get_metric(
                    metrics,
                    "average_semantic_similarity",
                    "semantic_similarity",
                ),
                "Judge Correctness": get_metric(
                    metrics,
                    "average_judge_correctness",
                    "judge_correctness",
                ),
                "Judge Relevance": get_metric(
                    metrics,
                    "average_judge_relevance",
                    "judge_relevance",
                ),
                "Judge Reasoning": get_metric(
                    metrics,
                    "average_judge_reasoning_quality",
                    "judge_reasoning_quality",
                ),
            }
        )

    quality_df = pd.DataFrame(
        quality_rows
    )

    if quality_df.empty:

        st.info(
            "No quality metrics available."
        )

    else:

        st.dataframe(
            quality_df,
            width="stretch",
            hide_index=True,
        )

        st.subheader(
            "Traditional Evaluation Metrics"
        )

        quality_chart = (
            quality_df
            .groupby("Model")[
                [
                    "Exact Match",
                    "Fuzzy Match",
                    "Keyword Match",
                ]
            ]
            .mean()
        )

        st.bar_chart(
            quality_chart
        )

        st.subheader(
            "Semantic Evaluation"
        )

        semantic_chart = (
            quality_df
            .groupby("Model")[
                [
                    "Semantic Match",
                    "Semantic Similarity",
                ]
            ]
            .mean()
        )

        st.bar_chart(
            semantic_chart
        )

        st.info(
            "Semantic similarity is displayed using the "
            "recorded benchmark values. The configured "
            "semantic threshold is not treated as a "
            "universally calibrated threshold."
        )

        st.subheader(
            "LLM-as-a-Judge"
        )

        judge_columns = [
            "Judge Correctness",
            "Judge Relevance",
            "Judge Reasoning",
        ]

        judge_chart = (
            quality_df
            .groupby("Model")[
                judge_columns
            ]
            .mean()
        )

        st.bar_chart(
            judge_chart
        )


# ============================================================
# TAB 4 — CATEGORY ANALYSIS
# ============================================================

with category_tab:

    st.header(
        "📚 Category Performance"
    )

    category_rows = []

    for experiment in filtered_runs:

        results = experiment.get(
            "results",
            [],
        )

        if not isinstance(
            results,
            list,
        ):
            continue

        for result in results:

            if not isinstance(
                result,
                dict,
            ):
                continue

            category = result.get(
                "category",
                "unknown",
            )

            evaluation = result.get(
                "evaluation",
                {},
            )

            if not isinstance(
                evaluation,
                dict,
            ):
                evaluation = {}

            category_rows.append(
                {
                    "Model": experiment.get(
                        "model"
                    ),
                    "Category": category,
                    "Exact Match": safe_float(
                        evaluation.get(
                            "exact_match"
                        )
                    ),
                    "Fuzzy Match": safe_float(
                        evaluation.get(
                            "fuzzy_match"
                        )
                    ),
                    "Keyword Match": safe_float(
                        evaluation.get(
                            "keyword_match"
                        )
                    ),
                }
            )

    category_df = pd.DataFrame(
        category_rows
    )

    if category_df.empty:

        st.info(
            "Category-level result data is not "
            "available for the selected experiments."
        )

    else:

        category_summary = (
            category_df
            .groupby(
                [
                    "Model",
                    "Category",
                ]
            )
            .agg(
                {
                    "Exact Match": "mean",
                    "Fuzzy Match": "mean",
                    "Keyword Match": "mean",
                }
            )
            .reset_index()
        )

        st.dataframe(
            category_summary,
            width="stretch",
            hide_index=True,
        )

        st.subheader(
            "Accuracy by Category"
        )

        category_chart = (
            category_df
            .groupby(
                [
                    "Model",
                    "Category",
                ]
            )[
                "Exact Match"
            ]
            .mean()
            .unstack(
                fill_value=0
            )
        )

        st.bar_chart(
            category_chart
        )


# ============================================================
# TAB 5 — PERFORMANCE
# ============================================================

with performance_tab:

    st.header(
        "⚡ Performance Analysis"
    )

    performance_rows = []

    for experiment in filtered_runs:

        metrics = experiment.get(
            "metrics",
            {},
        )

        performance_rows.append(
            {
                "Model": experiment.get(
                    "model"
                ),
                "Average Latency": get_metric(
                    metrics,
                    "average_latency",
                ),
                "P50 Latency": get_metric(
                    metrics,
                    "p50_latency",
                ),
                "P95 Latency": get_metric(
                    metrics,
                    "p95_latency",
                ),
                "Provider Latency": get_metric(
                    metrics,
                    "average_provider_latency",
                    "provider_latency",
                ),
                "Tokens/sec": get_metric(
                    metrics,
                    "tokens_per_second",
                ),
            }
        )

    performance_df = pd.DataFrame(
        performance_rows
    )

    if performance_df.empty:

        st.info(
            "No performance data available."
        )

    else:

        st.dataframe(
            performance_df,
            width="stretch",
            hide_index=True,
        )

        st.subheader(
            "Average Latency"
        )

        st.bar_chart(
            performance_df
            .groupby("Model")[
                "Average Latency"
            ]
            .mean()
        )

        st.subheader(
            "P50 vs P95"
        )

        latency_percentiles = (
            performance_df
            .groupby("Model")[
                [
                    "P50 Latency",
                    "P95 Latency",
                ]
            ]
            .mean()
        )

        st.bar_chart(
            latency_percentiles
        )

        st.subheader(
            "Tokens per Second"
        )

        st.bar_chart(
            performance_df
            .groupby("Model")[
                "Tokens/sec"
            ]
            .mean()
        )


# ============================================================
# TAB 6 — COST
# ============================================================

with cost_tab:

    st.header(
        "💰 Cost Analysis"
    )

    cost_rows = []

    for experiment in filtered_runs:

        metrics = experiment.get(
            "metrics",
            {},
        )

        cost_rows.append(
            {
                "Model": experiment.get(
                    "model"
                ),
                "Input Tokens": safe_int(
                    metrics.get(
                        "input_tokens"
                    )
                ),
                "Output Tokens": safe_int(
                    metrics.get(
                        "output_tokens"
                    )
                ),
                "Total Tokens": safe_int(
                    metrics.get(
                        "total_tokens"
                    )
                ),
                "Estimated Cost ($)": get_metric(
                    metrics,
                    "total_cost",
                ),
            }
        )

    cost_df = pd.DataFrame(
        cost_rows
    )

    if cost_df.empty:

        st.info(
            "No cost data available."
        )

    else:

        st.dataframe(
            cost_df,
            width="stretch",
            hide_index=True,
        )

        st.subheader(
            "Recorded Estimated Cost"
        )

        cost_by_model = (
            cost_df
            .groupby("Model")[
                "Estimated Cost ($)"
            ]
            .sum()
        )

        st.bar_chart(
            cost_by_model
        )

        st.info(
            "Cost values are shown as recorded by the "
            "benchmark. A value of zero should not "
            "automatically be interpreted as verified "
            "commercial pricing."
        )


# ============================================================
# TAB 7 — RELIABILITY
# ============================================================

with reliability_tab:

    st.header(
        "🔄 Reliability Analysis"
    )

    reliability_rows = []

    for experiment in filtered_runs:

        metrics = experiment.get(
            "metrics",
            {},
        )

        reliability_rows.append(
            {
                "Model": experiment.get(
                    "model"
                ),
                "Success Rate": get_metric(
                    metrics,
                    "success_rate",
                ),
                "Error Rate": get_metric(
                    metrics,
                    "error_rate",
                ),
                "Retry Rate": get_metric(
                    metrics,
                    "retry_rate",
                ),
                "Fallback Rate": get_metric(
                    metrics,
                    "fallback_rate",
                ),
                "Fallback Success Rate": get_metric(
                    metrics,
                    "fallback_success_rate",
                ),
                "Average Attempts": get_metric(
                    metrics,
                    "average_attempts",
                ),
                "Average Retries": get_metric(
                    metrics,
                    "average_retries",
                ),
            }
        )

    reliability_df = pd.DataFrame(
        reliability_rows
    )

    if reliability_df.empty:

        st.info(
            "No reliability data available."
        )

    else:

        st.dataframe(
            reliability_df,
            width="stretch",
            hide_index=True,
        )

        st.subheader(
            "Error Rate"
        )

        st.bar_chart(
            reliability_df
            .groupby("Model")[
                "Error Rate"
            ]
            .mean()
        )

        st.subheader(
            "Retry Rate"
        )

        st.bar_chart(
            reliability_df
            .groupby("Model")[
                "Retry Rate"
            ]
            .mean()
        )

        st.subheader(
            "Fallback Rate"
        )

        st.bar_chart(
            reliability_df
            .groupby("Model")[
                "Fallback Rate"
            ]
            .mean()
        )


# ============================================================
# TAB 8 — FAILURE ANALYSIS
# ============================================================

with failure_tab:

    st.header(
        "🚨 Failure Analysis"
    )

    all_failures = []

    for experiment in filtered_runs:

        all_failures.extend(
            extract_result_failures(
                experiment
            )
        )

    failure_count = len(
        all_failures
    )

    st.metric(
        "Recorded Failed Questions",
        failure_count,
    )

    if not all_failures:

        st.success(
            "No question-level failures are recorded "
            "in the selected experiments."
        )

        st.info(
            "This does not mean that every historical "
            "experiment was analyzed by FailureAnalyzer. "
            "Only recorded failure information is shown."
        )

    else:

        failures_df = pd.DataFrame(
            all_failures
        )

        st.subheader(
            "Failure Distribution"
        )

        failure_distribution = (
            failures_df[
                "failure_type"
            ]
            .value_counts()
        )

        st.bar_chart(
            failure_distribution
        )

        st.subheader(
            "Failures by Model"
        )

        st.bar_chart(
            failures_df[
                "model"
            ].value_counts()
        )

        st.subheader(
            "Failure Details"
        )

        st.dataframe(
            failures_df,
            width="stretch",
            hide_index=True,
        )


# ============================================================
# TAB 9 — STATISTICS
# ============================================================

with statistics_tab:

    st.header(
        "📈 Statistical Analysis"
    )

    statistic_metric = st.selectbox(
        "Metric",
        [
            "accuracy",
            "average_latency",
            "p50_latency",
            "p95_latency",
            "tokens_per_second",
        ],
    )

    statistic_rows = []

    model_names = sorted(
        {
            experiment.get(
                "model"
            )
            for experiment in filtered_runs
        }
    )

    for model in model_names:

        values = []

        for experiment in filtered_runs:

            if experiment.get(
                "model"
            ) != model:
                continue

            metrics = experiment.get(
                "metrics",
                {},
            )

            if not isinstance(
                metrics,
                dict,
            ):
                continue

            value = metrics.get(
                statistic_metric
            )

            if value is None:
                continue

            values.append(
                safe_float(value)
            )

        if not values:
            continue

        values_series = pd.Series(
            values
        )

        count = len(
            values
        )

        mean = float(
            values_series.mean()
        )

        median = float(
            values_series.median()
        )

        if count > 1:

            std_dev = float(
                values_series.std(
                    ddof=1
                )
            )

            variance = float(
                values_series.var(
                    ddof=1
                )
            )

        else:

            std_dev = 0.0
            variance = 0.0

        statistic_rows.append(
            {
                "Model": model,
                "Runs": count,
                "Mean": mean,
                "Median": median,
                "Std Dev": std_dev,
                "Variance": variance,
                "Min": float(
                    values_series.min()
                ),
                "Max": float(
                    values_series.max()
                ),
                "Range": float(
                    values_series.max()
                    - values_series.min()
                ),
            }
        )

    statistics_df = pd.DataFrame(
        statistic_rows
    )

    if statistics_df.empty:

        st.info(
            "Not enough repeated experiment data "
            "for this metric."
        )

    else:

        st.dataframe(
            statistics_df,
            width="stretch",
            hide_index=True,
        )

        st.subheader(
            "Mean"
        )

        st.bar_chart(
            statistics_df.set_index(
                "Model"
            )["Mean"]
        )

        st.subheader(
            "Median"
        )

        st.bar_chart(
            statistics_df.set_index(
                "Model"
            )["Median"]
        )

        st.info(
            "The dashboard displays descriptive statistics "
            "from the stored experiments. For small samples, "
            "Student's t-distribution is generally preferable "
            "to a normal approximation for confidence intervals."
        )


# ============================================================
# TAB 10 — EXPERIMENT HISTORY
# ============================================================

with experiments_tab:

    st.header(
        "🧪 Experiment History"
    )

    history_rows = []

    for experiment in filtered_runs:

        dataset = experiment.get(
            "dataset"
        ) or {}

        metrics = experiment.get(
            "metrics",
            {},
        )

        history_rows.append(
            {
                "Experiment ID": experiment.get(
                    "experiment_id"
                ),
                "Model": experiment.get(
                    "model"
                ),
                "Timestamp": experiment.get(
                    "timestamp"
                ),
                "Dataset Version": dataset.get(
                    "version"
                ),
                "Dataset Questions": dataset.get(
                    "question_count"
                ),
                "Questions Run": get_questions_run(
                    experiment
                ),
                "Accuracy": format_percentage(
                    metrics.get(
                        "accuracy"
                    )
                ),
            }
        )

    history_df = pd.DataFrame(
        history_rows
    )

    if not history_df.empty:

        st.dataframe(
            history_df,
            width="stretch",
            hide_index=True,
        )

    st.subheader(
        "🔐 Dataset Reproducibility"
    )

    metadata_col1, metadata_col2 = st.columns(
        2
    )

    metadata_col1.metric(
        "Dataset Version",
        dataset_metadata.get(
            "version",
            "Unknown",
        ),
    )

    metadata_col2.write(
        "Dataset SHA-256"
    )

    # Calculate actual hash when the dataset exists.
    if DATASET_FILE.exists():

        try:

            dataset_hash = hashlib.sha256(
                DATASET_FILE.read_bytes()
            ).hexdigest()

        except OSError:

            dataset_hash = "Unable to calculate hash."

    else:

        dataset_hash = "Dataset file not found."

    metadata_col2.code(
        dataset_hash
    )

    if dataset_metadata.get(
        "sha256"
    ):

        st.caption(
            "Metadata SHA-256:"
        )

        st.code(
            dataset_metadata.get(
                "sha256"
            )
        )


# ============================================================
# TAB 11 — QUESTION EXPLORER
# ============================================================

with explorer_tab:

    st.header(
        "🔎 Question-Level Explorer"
    )

    experiment_options = [
        experiment.get(
            "experiment_id"
        )
        for experiment in filtered_runs
    ]

    selected_experiment_id = st.selectbox(
        "Experiment",
        experiment_options,
    )

    selected_experiment = next(
        (
            experiment
            for experiment in filtered_runs
            if experiment.get(
                "experiment_id"
            )
            == selected_experiment_id
        ),
        None,
    )

    if selected_experiment:

        results = selected_experiment.get(
            "results",
            [],
        )

        if not isinstance(
            results,
            list,
        ):

            results = []

        if not results:

            st.info(
                "Question-level results are not "
                "available for this experiment."
            )

        else:

            categories = sorted(
                {
                    result.get(
                        "category",
                        "unknown",
                    )
                    for result in results
                    if isinstance(
                        result,
                        dict,
                    )
                }
            )

            selected_category = st.selectbox(
                "Category",
                ["All"] + categories,
            )

            filtered_results = results

            if selected_category != "All":

                filtered_results = [
                    result
                    for result in results
                    if isinstance(
                        result,
                        dict,
                    )
                    and result.get(
                        "category",
                        "unknown",
                    )
                    == selected_category
                ]

            question_options = [
                result.get(
                    "id"
                )
                for result in filtered_results
            ]

            if not question_options:

                st.info(
                    "No questions available "
                    "for this category."
                )

            else:

                selected_question_id = st.selectbox(
                    "Question",
                    question_options,
                )

                selected_result = next(
                    (
                        result
                        for result in filtered_results
                        if result.get(
                            "id"
                        )
                        == selected_question_id
                    ),
                    None,
                )

                if selected_result:

                    # ------------------------------------------------
                    # Question
                    # ------------------------------------------------

                    st.subheader(
                        "Question"
                    )

                    question_text = (
                        selected_result.get(
                            "question"
                        )
                        or selected_result.get(
                            "prompt"
                        )
                        or "Not recorded."
                    )

                    st.write(
                        question_text
                    )

                    # ------------------------------------------------
                    # Expected
                    # ------------------------------------------------

                    st.subheader(
                        "Expected Answer"
                    )

                    st.write(
                        selected_result.get(
                            "expected",
                            "Not recorded.",
                        )
                    )

                    # ------------------------------------------------
                    # Response
                    # ------------------------------------------------

                    st.subheader(
                        "Model Response"
                    )

                    st.write(
                        selected_result.get(
                            "response",
                            "No response recorded.",
                        )
                    )

                    # ------------------------------------------------
                    # Evaluation
                    # ------------------------------------------------

                    evaluation = selected_result.get(
                        "evaluation",
                        {},
                    )

                    if not isinstance(
                        evaluation,
                        dict,
                    ):
                        evaluation = {}

                    semantic = selected_result.get(
                        "semantic_evaluation",
                        {},
                    )

                    if not isinstance(
                        semantic,
                        dict,
                    ):
                        semantic = {}

                    judge = selected_result.get(
                        "llm_judge",
                        {},
                    )

                    if not isinstance(
                        judge,
                        dict,
                    ):
                        judge = {}

                    st.subheader(
                        "Evaluation"
                    )

                    eval_col1, eval_col2, eval_col3, eval_col4 = (
                        st.columns(4)
                    )

                    eval_col1.metric(
                        "Exact Match",
                        format_percentage(
                            evaluation.get(
                                "exact_match"
                            )
                        ),
                    )

                    eval_col2.metric(
                        "Fuzzy Match",
                        format_percentage(
                            evaluation.get(
                                "fuzzy_match"
                            )
                        ),
                    )

                    eval_col3.metric(
                        "Keyword Match",
                        format_percentage(
                            evaluation.get(
                                "keyword_match"
                            )
                        ),
                    )

                    eval_col4.metric(
                        "Semantic Similarity",
                        f"{safe_float(
                            semantic.get(
                                "semantic_similarity"
                            )
                        ):.4f}",
                    )

                    # ------------------------------------------------
                    # Performance
                    # ------------------------------------------------

                    st.subheader(
                        "Performance"
                    )

                    performance_col1, performance_col2, performance_col3 = (
                        st.columns(3)
                    )

                    performance_col1.metric(
                        "Latency",
                        format_seconds(
                            selected_result.get(
                                "latency"
                            )
                        ),
                    )

                    (
                        input_tokens,
                        output_tokens,
                        total_tokens,
                    ) = extract_tokens(
                        selected_result
                    )

                    performance_col2.metric(
                        "Output Tokens",
                        output_tokens,
                    )

                    performance_col3.metric(
                        "Total Tokens",
                        total_tokens,
                    )

                    # ------------------------------------------------
                    # LLM Judge
                    # ------------------------------------------------

                    if judge:

                        st.subheader(
                            "LLM-as-a-Judge"
                        )

                        judge_col1, judge_col2, judge_col3, judge_col4 = (
                            st.columns(4)
                        )

                        judge_col1.metric(
                            "Correctness",
                            format_percentage(
                                judge.get(
                                    "correctness"
                                )
                            ),
                        )

                        judge_col2.metric(
                            "Relevance",
                            format_percentage(
                                judge.get(
                                    "relevance"
                                )
                            ),
                        )

                        judge_col3.metric(
                            "Reasoning Quality",
                            format_percentage(
                                judge.get(
                                    "reasoning_quality"
                                )
                            ),
                        )

                        judge_col4.metric(
                            "Confidence",
                            format_percentage(
                                judge.get(
                                    "confidence"
                                )
                            ),
                        )

                    # ------------------------------------------------
                    # Retry / Fallback
                    # ------------------------------------------------

                    retry = selected_result.get(
                        "retry",
                        {},
                    )

                    if not isinstance(
                        retry,
                        dict,
                    ):
                        retry = {}

                    if retry:

                        st.subheader(
                            "🔄 Retry / Fallback"
                        )

                        retry_col1, retry_col2, retry_col3 = (
                            st.columns(3)
                        )

                        retry_col1.metric(
                            "Attempts",
                            safe_int(
                                retry.get(
                                    "attempts"
                                )
                            ),
                        )

                        retry_col2.metric(
                            "Retries",
                            safe_int(
                                retry.get(
                                    "retries"
                                )
                            ),
                        )

                        retry_col3.metric(
                            "Fallback Used",
                            str(
                                retry.get(
                                    "fallback_used",
                                    False,
                                )
                            ),
                        )

                    # ------------------------------------------------
                    # Error
                    # ------------------------------------------------

                    if selected_result.get(
                        "error"
                    ):

                        st.error(
                            selected_result.get(
                                "error"
                            )
                        )

                    # ------------------------------------------------
                    # Raw result
                    # ------------------------------------------------

                    with st.expander(
                        "🔍 Raw Result"
                    ):

                        st.json(
                            selected_result
                        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "LLM Benchmarking Platform • "
    "Experiment-driven evaluation • "
    "Historical results preserved • "
    "No fabricated benchmark results"
)