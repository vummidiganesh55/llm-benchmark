from backend.benchmark.failure_analyzer import FailureAnalyzer


def test_failure_types_exist():
    expected_types = {
        "provider_error",
        "timeout",
        "rate_limit",
        "evaluation_failure",
        "factual_error",
        "reasoning_error",
        "coding_error",
        "formatting_error",
        "instruction_following_error",
        "unknown",
    }

    assert FailureAnalyzer.FAILURE_TYPES == expected_types


def test_analyze_provider_error():
    result = FailureAnalyzer.analyze(
        {
            "status": "error",
            "category": "general",
            "error_type": "ClientError",
            "error_message": "Internal provider server error",
        }
    )

    assert result["is_failure"] is True
    assert result["failure_type"] == "provider_error"
    assert result["category"] == "general"
    assert result["error_type"] == "ClientError"
    assert result["analysis_source"] == "provider"


def test_analyze_timeout_from_error_message():
    result = FailureAnalyzer.analyze(
        {
            "status": "error",
            "category": "general",
            "error_type": "ClientError",
            "error_message": "Request timeout after 120 seconds",
        }
    )

    assert result["is_failure"] is True
    assert result["failure_type"] == "timeout"
    assert result["analysis_source"] == "provider"


def test_analyze_timeout_from_error_type():
    result = FailureAnalyzer.analyze(
        {
            "status": "error",
            "category": "general",
            "error_type": "TimeoutError",
            "error_message": "Request failed",
        }
    )

    assert result["is_failure"] is True
    assert result["failure_type"] == "timeout"


def test_analyze_rate_limit_429():
    result = FailureAnalyzer.analyze(
        {
            "status": "error",
            "category": "math",
            "error_type": "ClientError",
            "error_message": "429 Too Many Requests",
        }
    )

    assert result["is_failure"] is True
    assert result["failure_type"] == "rate_limit"
    assert result["category"] == "math"


def test_analyze_rate_limit_resource_exhausted():
    result = FailureAnalyzer.analyze(
        {
            "status": "error",
            "category": "math",
            "error_type": "ClientError",
            "error_message": "RESOURCE_EXHAUSTED quota exceeded",
        }
    )

    assert result["is_failure"] is True
    assert result["failure_type"] == "rate_limit"


def test_analyze_rate_limit_quota_exceeded():
    result = FailureAnalyzer.analyze(
        {
            "status": "error",
            "category": "math",
            "error_type": "ClientError",
            "error_message": "You exceeded your quota",
        }
    )

    assert result["is_failure"] is True
    assert result["failure_type"] == "rate_limit"


def test_analyze_successful_correct_answer():
    result = FailureAnalyzer.analyze(
        {
            "status": "success",
            "category": "knowledge",
            "response": "Paris",
            "expected": "Paris",
            "evaluation": {
                "exact_match": 1.0,
                "fuzzy_match": 1.0,
                "keyword_match": 1.0,
                "semantic_match": 1.0,
            },
        }
    )

    assert result["is_failure"] is False
    assert result["failure_type"] is None
    assert result["category"] == "knowledge"
    assert result["analysis_source"] == "evaluation"


def test_analyze_factual_error():
    result = FailureAnalyzer.analyze(
        {
            "status": "success",
            "category": "knowledge",
            "response": "Berlin",
            "expected": "Paris",
            "evaluation": {
                "exact_match": 0.0,
                "fuzzy_match": 0.0,
                "keyword_match": 0.0,
                "semantic_match": 0.0,
            },
        }
    )

    assert result["is_failure"] is True
    assert result["failure_type"] == "factual_error"
    assert result["category"] == "knowledge"
    assert result["analysis_source"] == "evaluation"


def test_analyze_reasoning_error():
    result = FailureAnalyzer.analyze(
        {
            "status": "success",
            "category": "reasoning",
            "response": "20",
            "expected": "42",
            "evaluation": {
                "exact_match": 0.0,
                "fuzzy_match": 0.0,
                "keyword_match": 0.0,
                "semantic_match": 0.0,
            },
        }
    )

    assert result["is_failure"] is True
    assert result["failure_type"] == "reasoning_error"
    assert result["category"] == "reasoning"


def test_analyze_coding_error():
    result = FailureAnalyzer.analyze(
        {
            "status": "success",
            "category": "coding",
            "response": "def add(a, b): return a - b",
            "expected": "def add(a, b): return a + b",
            "evaluation": {
                "exact_match": 0.0,
                "fuzzy_match": 0.0,
                "keyword_match": 0.0,
                "semantic_match": 0.0,
            },
        }
    )

    assert result["is_failure"] is True
    assert result["failure_type"] == "coding_error"
    assert result["category"] == "coding"


def test_analyze_formatting_error():
    result = FailureAnalyzer.analyze(
        {
            "status": "success",
            "category": "general",
            "response": "Paris.",
            "expected": "Paris",
            "evaluation": {
                "exact_match": 0.0,
                "fuzzy_match": 1.0,
                "keyword_match": 1.0,
                "semantic_match": 1.0,
            },
        }
    )

    assert result["is_failure"] is True
    assert result["failure_type"] == "formatting_error"
    assert result["category"] == "general"


def test_analyze_keyword_semantic_factual_error():
    result = FailureAnalyzer.analyze(
        {
            "status": "success",
            "category": "knowledge",
            "response": "London",
            "expected": "Paris",
            "evaluation": {
                "exact_match": 0.0,
                "fuzzy_match": 0.0,
                "keyword_match": 0.5,
                "semantic_match": 0.0,
            },
        }
    )

    assert result["is_failure"] is True
    assert result["failure_type"] == "factual_error"


def test_analyze_semantic_match_factual_error():
    result = FailureAnalyzer.analyze(
        {
            "status": "success",
            "category": "knowledge",
            "response": "The capital is Paris",
            "expected": "Paris",
            "evaluation": {
                "exact_match": 0.0,
                "fuzzy_match": 0.0,
                "keyword_match": 0.0,
                "semantic_match": 1.0,
            },
        }
    )

    assert result["is_failure"] is True
    assert result["failure_type"] == "factual_error"


def test_analyze_unknown_status():
    result = FailureAnalyzer.analyze(
        {
            "status": "unknown",
            "category": "general",
            "error_type": "UnknownError",
            "error_message": "Something unexpected happened",
        }
    )

    assert result["is_failure"] is True
    assert result["failure_type"] == "unknown"
    assert result["category"] == "general"
    assert result["error_type"] == "UnknownError"
    assert result["error_message"] == "Something unexpected happened"
    assert result["analysis_source"] == "unknown"


def test_analyze_results_empty():
    result = FailureAnalyzer.analyze_results([])

    assert result["total_results"] == 0
    assert result["total_failures"] == 0
    assert result["failure_rate"] == 0.0
    assert result["failure_counts"] == {}
    assert result["category_failures"] == {}
    assert result["results"] == []


def test_analyze_results_mixed_failures():
    results = [
        {
            "status": "error",
            "category": "math",
            "error_type": "ClientError",
            "error_message": "429 Too Many Requests",
        },
        {
            "status": "error",
            "category": "general",
            "error_type": "TimeoutError",
            "error_message": "Request timed out",
        },
        {
            "status": "success",
            "category": "knowledge",
            "response": "Berlin",
            "expected": "Paris",
            "evaluation": {
                "exact_match": 0.0,
                "fuzzy_match": 0.0,
                "keyword_match": 0.0,
                "semantic_match": 0.0,
            },
        },
        {
            "status": "success",
            "category": "reasoning",
            "response": "20",
            "expected": "42",
            "evaluation": {
                "exact_match": 0.0,
                "fuzzy_match": 0.0,
                "keyword_match": 0.0,
                "semantic_match": 0.0,
            },
        },
        {
            "status": "success",
            "category": "coding",
            "response": "wrong code",
            "expected": "correct code",
            "evaluation": {
                "exact_match": 0.0,
                "fuzzy_match": 0.0,
                "keyword_match": 0.0,
                "semantic_match": 0.0,
            },
        },
        {
            "status": "success",
            "category": "general",
            "response": "Paris.",
            "expected": "Paris",
            "evaluation": {
                "exact_match": 0.0,
                "fuzzy_match": 1.0,
                "keyword_match": 1.0,
                "semantic_match": 1.0,
            },
        },
    ]

    result = FailureAnalyzer.analyze_results(results)

    assert result["total_results"] == 6
    assert result["total_failures"] == 6
    assert result["failure_rate"] == 1.0

    assert result["failure_counts"]["rate_limit"] == 1
    assert result["failure_counts"]["timeout"] == 1
    assert result["failure_counts"]["factual_error"] == 1
    assert result["failure_counts"]["reasoning_error"] == 1
    assert result["failure_counts"]["coding_error"] == 1
    assert result["failure_counts"]["formatting_error"] == 1


def test_analyze_results_failure_rate():
    results = [
        {
            "status": "success",
            "category": "knowledge",
            "response": "Paris",
            "expected": "Paris",
            "evaluation": {
                "exact_match": 1.0,
                "fuzzy_match": 1.0,
                "keyword_match": 1.0,
                "semantic_match": 1.0,
            },
        },
        {
            "status": "error",
            "category": "general",
            "error_type": "TimeoutError",
            "error_message": "Request timeout",
        },
        {
            "status": "success",
            "category": "reasoning",
            "response": "wrong",
            "expected": "correct",
            "evaluation": {
                "exact_match": 0.0,
                "fuzzy_match": 0.0,
                "keyword_match": 0.0,
                "semantic_match": 0.0,
            },
        },
        {
            "status": "success",
            "category": "coding",
            "response": "correct",
            "expected": "correct",
            "evaluation": {
                "exact_match": 1.0,
                "fuzzy_match": 1.0,
                "keyword_match": 1.0,
                "semantic_match": 1.0,
            },
        },
    ]

    result = FailureAnalyzer.analyze_results(results)

    assert result["total_results"] == 4
    assert result["total_failures"] == 2
    assert result["failure_rate"] == 0.5


def test_analyze_results_category_failures():
    results = [
        {
            "status": "error",
            "category": "math",
            "error_type": "ClientError",
            "error_message": "429 Too Many Requests",
        },
        {
            "status": "error",
            "category": "math",
            "error_type": "TimeoutError",
            "error_message": "Request timeout",
        },
        {
            "status": "success",
            "category": "knowledge",
            "response": "Berlin",
            "expected": "Paris",
            "evaluation": {
                "exact_match": 0.0,
                "fuzzy_match": 0.0,
                "keyword_match": 0.0,
                "semantic_match": 0.0,
            },
        },
    ]

    result = FailureAnalyzer.analyze_results(results)

    assert result["category_failures"]["math"]["rate_limit"] == 1
    assert result["category_failures"]["math"]["timeout"] == 1
    assert result["category_failures"]["knowledge"]["factual_error"] == 1


def test_analyze_results_preserves_original_result():
    original = {
        "status": "error",
        "category": "math",
        "error_type": "TimeoutError",
        "error_message": "Request timeout",
        "model": "test-model",
    }

    result = FailureAnalyzer.analyze_results([original])

    analyzed = result["results"][0]

    assert analyzed["status"] == "error"
    assert analyzed["category"] == "math"
    assert analyzed["model"] == "test-model"

    assert "failure_analysis" in analyzed
    assert analyzed["failure_analysis"]["failure_type"] == "timeout"