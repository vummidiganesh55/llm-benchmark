import pytest

from backend.benchmark.evaluator import Evaluator


# ============================================================
# NORMALIZATION
# ============================================================

def test_normalize_lowercase():
    result = Evaluator.normalize(
        "PARIS"
    )

    assert result == "paris"


def test_normalize_whitespace():
    result = Evaluator.normalize(
        "  Paris    is   beautiful  "
    )

    assert result == "paris is beautiful"


def test_normalize_punctuation():
    result = Evaluator.normalize(
        "Paris!"
    )

    assert result == "paris"


def test_normalize_number_words():
    result = Evaluator.normalize(
        "one two three four five"
    )

    assert result == "1 2 3 4 5"


# ============================================================
# EXACT MATCH
# ============================================================

def test_exact_match_correct_answer():
    result = Evaluator.exact_match(
        "Paris",
        "Paris",
    )

    assert result == 1.0


def test_exact_match_case_insensitive():
    result = Evaluator.exact_match(
        "PARIS",
        "paris",
    )

    assert result == 1.0


def test_exact_match_expected_inside_response():
    result = Evaluator.exact_match(
        "The capital of France is Paris.",
        "Paris",
    )

    assert result == 1.0


def test_exact_match_incorrect_answer():
    result = Evaluator.exact_match(
        "London",
        "Paris",
    )

    assert result == 0.0


def test_exact_match_empty_expected():
    result = Evaluator.exact_match(
        "Paris",
        "",
    )

    assert result == 0.0


# ============================================================
# FUZZY SIMILARITY
# ============================================================

def test_fuzzy_similarity_identical():
    result = Evaluator.fuzzy_similarity(
        "Paris",
        "Paris",
    )

    assert result == 1.0


def test_fuzzy_similarity_similar_words():
    result = Evaluator.fuzzy_similarity(
        "paris",
        "pari",
    )

    assert result > 0.80


def test_fuzzy_similarity_different_answers():
    result = Evaluator.fuzzy_similarity(
        "London",
        "Paris",
    )

    assert result < 0.80


def test_fuzzy_similarity_expected_phrase_inside_response():
    result = Evaluator.fuzzy_similarity(
        "The capital city of France is Paris.",
        "Paris",
    )

    assert result == 1.0


def test_fuzzy_similarity_empty_response():
    result = Evaluator.fuzzy_similarity(
        "",
        "Paris",
    )

    assert result == 0.0


def test_fuzzy_similarity_empty_expected():
    result = Evaluator.fuzzy_similarity(
        "Paris",
        "",
    )

    assert result == 0.0


# ============================================================
# FUZZY MATCH
# ============================================================

def test_fuzzy_match_above_threshold():
    result = Evaluator.fuzzy_match(
        "Paris",
        "Paris",
    )

    assert result == 1.0


def test_fuzzy_match_below_threshold():
    result = Evaluator.fuzzy_match(
        "London",
        "Paris",
    )

    assert result == 0.0


def test_fuzzy_match_custom_threshold():
    result = Evaluator.fuzzy_match(
        "paris",
        "pari",
        threshold=0.90,
    )

    assert result == 1.0


# ============================================================
# KEYWORD MATCH
# ============================================================

def test_keyword_match_all_keywords():
    result = Evaluator.keyword_match(
        "Paris is the capital of France",
        "Paris France",
    )

    assert result == 1.0


def test_keyword_match_partial_keywords():
    result = Evaluator.keyword_match(
        "Paris is beautiful",
        "Paris France",
    )

    assert result == 0.5


def test_keyword_match_no_keywords():
    result = Evaluator.keyword_match(
        "London",
        "Paris",
    )

    assert result == 0.0


def test_keyword_match_empty_expected():
    result = Evaluator.keyword_match(
        "Paris",
        "",
    )

    assert result == 0.0


# ============================================================
# FULL EVALUATION
# ============================================================

def test_evaluate_returns_all_metrics():

    result = Evaluator.evaluate(
        "Paris is the capital of France.",
        "Paris",
    )

    assert isinstance(
        result,
        dict,
    )

    assert "exact_match" in result
    assert "fuzzy_match" in result
    assert "fuzzy_similarity" in result
    assert "keyword_match" in result


def test_evaluate_correct_answer():

    result = Evaluator.evaluate(
        "Paris",
        "Paris",
    )

    assert result["exact_match"] == 1.0
    assert result["fuzzy_match"] == 1.0
    assert result["fuzzy_similarity"] == 1.0
    assert result["keyword_match"] == 1.0


def test_evaluate_incorrect_answer():

    result = Evaluator.evaluate(
        "London",
        "Paris",
    )

    assert result["exact_match"] == 0.0
    assert result["fuzzy_match"] == 0.0
    assert result["keyword_match"] == 0.0


# ============================================================
# NUMBER NORMALIZATION
# ============================================================

@pytest.mark.parametrize(
    "response,expected",
    [
        ("four", "4"),
        ("five", "5"),
        ("six", "6"),
        ("seven", "7"),
        ("eight", "8"),
        ("nine", "9"),
        ("ten", "10"),
    ],
)
def test_number_word_matches_digit(
    response,
    expected,
):

    result = Evaluator.exact_match(
        response,
        expected,
    )

    assert result == 1.0