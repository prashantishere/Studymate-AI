"""Unit tests for deterministic mastery, grading, and tie-breaking."""

import pytest
from studymate.mastery import (
    choose_weakest_concept,
    grade_answers,
    grade_retest_submission,
)
from studymate.schemas import GradingError


@pytest.fixture
def sample_questions():
    return [
        {
            "id": "q1",
            "concept": "Bayes Theorem",
            "question": "Q1 text",
            "options": ["A", "B", "C", "D"],
            "correct_index": 0,
            "explanation": "Exp 1",
            "source_pages": [1],
        },
        {
            "id": "q2",
            "concept": "Bayes Theorem",
            "question": "Q2 text",
            "options": ["A", "B", "C", "D"],
            "correct_index": 1,
            "explanation": "Exp 2",
            "source_pages": [1],
        },
        {
            "id": "q3",
            "concept": "Conditional Probability",
            "question": "Q3 text",
            "options": ["A", "B", "C", "D"],
            "correct_index": 2,
            "explanation": "Exp 3",
            "source_pages": [2],
        },
        {
            "id": "q4",
            "concept": "Conditional Probability",
            "question": "Q4 text",
            "options": ["A", "B", "C", "D"],
            "correct_index": 3,
            "explanation": "Exp 4",
            "source_pages": [2],
        },
        {
            "id": "q5",
            "concept": "Probability Distributions",
            "question": "Q5 text",
            "options": ["A", "B", "C", "D"],
            "correct_index": 0,
            "explanation": "Exp 5",
            "source_pages": [3],
        },
    ]


def test_correct_answer_grading(sample_questions):
    """Verify that all correct answers yield 100% accuracy and None for weakest concept."""
    answers = {"q1": 0, "q2": 1, "q3": 2, "q4": 3, "q5": 0}
    res = grade_answers(sample_questions, answers)

    assert res["correct_answers"] == 5
    assert res["attempted_answers"] == 5
    assert res["overall_accuracy"] == 100.0
    assert res["missed_concepts"] == []
    assert res["weakest_concept"] is None
    assert res["mastery"]["Bayes Theorem"]["accuracy"] == 100.0


def test_incorrect_answer_grading(sample_questions):
    """Verify grading with incorrect answers calculates concept accuracy accurately."""
    # q1 wrong (got 1, expected 0), q2 correct (1), q3 wrong (0, expected 2), q4 wrong (0, exp 3), q5 correct (0)
    answers = {"q1": 1, "q2": 1, "q3": 0, "q4": 0, "q5": 0}
    res = grade_answers(sample_questions, answers)

    assert res["correct_answers"] == 2
    assert res["attempted_answers"] == 5
    assert res["overall_accuracy"] == 40.0
    assert res["mastery"]["Bayes Theorem"]["accuracy"] == 50.0  # 1 of 2
    assert res["mastery"]["Conditional Probability"]["accuracy"] == 0.0  # 0 of 2
    assert res["mastery"]["Probability Distributions"]["accuracy"] == 100.0  # 1 of 1
    assert "Conditional Probability" in res["missed_concepts"]
    assert "Bayes Theorem" in res["missed_concepts"]
    assert res["weakest_concept"] == "Conditional Probability"


def test_unanswered_questions(sample_questions):
    """Verify unanswered questions are omitted from attempted calculations."""
    # Only answer q1 and q2, leave q3, q4, q5 unanswered
    answers = {"q1": 0, "q2": 1}
    res = grade_answers(sample_questions, answers)

    assert res["attempted_answers"] == 2
    assert res["correct_answers"] == 2
    assert res["overall_accuracy"] == 100.0
    assert res["mastery"]["Bayes Theorem"]["attempted"] == 2
    assert res["mastery"]["Conditional Probability"]["attempted"] == 0
    assert res["mastery"]["Conditional Probability"]["accuracy"] is None  # Untested


def test_concepts_with_zero_attempts(sample_questions):
    """Concepts with zero attempts must be marked untested, not failed or weakest."""
    answers = {"q1": 0, "q2": 0}  # 1 correct, 1 wrong in Bayes Theorem
    res = grade_answers(sample_questions, answers)

    # Bayes Theorem had 50%, other concepts had 0 attempts
    assert res["weakest_concept"] == "Bayes Theorem"
    assert res["mastery"]["Probability Distributions"]["accuracy"] is None


def test_tied_concept_scores_tie_breaker():
    """Tied concepts should use deterministic tie-breaking (more attempts first, then alphabetical)."""
    # Case 1: Same accuracy (0%), but different attempt counts
    mastery_diff_attempts = {
        "Topic Alpha": {"correct": 0, "attempted": 1, "accuracy": 0.0},
        "Topic Beta": {"correct": 0, "attempted": 3, "accuracy": 0.0},
    }
    # Topic Beta should be selected because it has 3 attempts vs 1 attempt
    assert choose_weakest_concept(mastery_diff_attempts) == "Topic Beta"

    # Case 2: Same accuracy (0%) and same attempts (2), tie-break alphabetically
    mastery_same_attempts = {
        "Zeta Topic": {"correct": 0, "attempted": 2, "accuracy": 0.0},
        "Alpha Topic": {"correct": 0, "attempted": 2, "accuracy": 0.0},
    }
    assert choose_weakest_concept(mastery_same_attempts) == "Alpha Topic"


def test_invalid_answer_indexes(sample_questions):
    """Out of bound answer index should raise GradingError."""
    answers = {"q1": 99}
    with pytest.raises(GradingError, match="Invalid answer index"):
        grade_answers(sample_questions, answers)


def test_retest_score_update():
    """Retest updates mastery accurately while preserving original diagnostic scores."""
    prev_mastery = {
        "Conditional Probability": {
            "correct": 0,
            "attempted": 2,
            "total": 2,
            "accuracy": 0.0,
        },
        "Bayes Theorem": {
            "correct": 2,
            "attempted": 2,
            "total": 2,
            "accuracy": 100.0,
        },
    }

    retest_q = {
        "id": "retest_q1",
        "concept": "Conditional Probability",
        "question": "What is P(A|B)?",
        "options": ["Option 0", "Option 1", "Option 2", "Option 3"],
        "correct_index": 1,
        "explanation": "Because...",
        "source_pages": [2],
    }

    # Student answers correctly
    res = grade_retest_submission(retest_q, answer_index=1, previous_mastery=prev_mastery)

    assert res["is_correct"] is True
    # Previous mastery must be preserved intact
    assert res["previous_mastery"]["Conditional Probability"]["accuracy"] == 0.0
    # Updated mastery should reflect 1 correct out of 3 attempts (33.3%)
    assert res["updated_mastery"]["Conditional Probability"]["correct"] == 1
    assert res["updated_mastery"]["Conditional Probability"]["attempted"] == 3
    assert res["updated_mastery"]["Conditional Probability"]["accuracy"] == 33.3
