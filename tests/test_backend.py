"""Unit tests for the StudyMate backend API and validation rules."""

import pytest
from studymate.ai_client import GemmaAIClient
from studymate.schemas import (
    ConfigurationError,
    DocumentContentError,
    InvalidModelResponseError,
    validate_question_dict,
    validate_quiz_list,
)
from backend import (
    choose_next_topic,
    evaluate_quiz,
    evaluate_retest,
    generate_quiz,
    generate_retest,
    set_ai_client,
    teach_concept,
)


@pytest.fixture(autouse=True)
def setup_mock_client():
    """Ensure mock AI client is active for backend testing."""
    mock_client = GemmaAIClient(mock_mode=True)
    set_ai_client(mock_client)
    yield


@pytest.fixture
def valid_doc_pages():
    return [
        {
            "page": 1,
            "text": "Conditional probability P(A|B) is the probability of event A given that B has occurred. Formula: P(A and B) / P(B).",
        },
        {
            "page": 2,
            "text": "Bayes' theorem updates probability as new evidence arrives. Discrete probability distributions must sum to exactly 1.0.",
        },
    ]


# =====================================================================
# Schema & Validation Tests
# =====================================================================

def test_question_schema_requires_four_options():
    """Questions with fewer or more than 4 options must be rejected."""
    bad_q = {
        "id": "q1",
        "concept": "Math",
        "question": "What is 2+2?",
        "options": ["3", "4", "5"],  # only 3 options
        "correct_index": 1,
        "explanation": "Because 2+2=4",
        "source_pages": [1],
    }
    with pytest.raises(InvalidModelResponseError, match="must have exactly 4 options"):
        validate_question_dict(bad_q)


def test_question_schema_rejects_empty_options():
    """Questions with blank/empty string options must be rejected."""
    bad_q = {
        "id": "q1",
        "concept": "Math",
        "question": "What is 2+2?",
        "options": ["3", "", "5", "6"],
        "correct_index": 0,
        "explanation": "Test",
        "source_pages": [1],
    }
    with pytest.raises(InvalidModelResponseError, match="empty or not a valid string"):
        validate_question_dict(bad_q)


def test_quiz_rejects_duplicate_ids():
    """A quiz with duplicate question IDs must raise InvalidModelResponseError."""
    dup_questions = [
        {
            "id": "q1",
            "concept": "C1",
            "question": "Q1 text?",
            "options": ["A", "B", "C", "D"],
            "correct_index": 0,
            "explanation": "Exp",
            "source_pages": [1],
        },
        {
            "id": "q1",  # duplicate ID
            "concept": "C2",
            "question": "Q2 text?",
            "options": ["A", "B", "C", "D"],
            "correct_index": 1,
            "explanation": "Exp",
            "source_pages": [1],
        },
    ]
    with pytest.raises(InvalidModelResponseError, match="Duplicate question ID found"):
        validate_quiz_list(dup_questions, expected_count=2)


def test_quiz_rejects_invalid_page_references():
    """Question referencing a page not in the document must be rejected."""
    q = {
        "id": "q1",
        "concept": "C1",
        "question": "Sample question text?",
        "options": ["A", "B", "C", "D"],
        "correct_index": 0,
        "explanation": "Exp",
        "source_pages": [99],  # Page 99 doesn't exist
    }
    with pytest.raises(InvalidModelResponseError, match="not in the document pages"):
        validate_question_dict(q, valid_page_numbers={1, 2})


def test_empty_or_unusable_document_content():
    """Generating a quiz from an empty document must raise DocumentContentError."""
    with pytest.raises(DocumentContentError):
        generate_quiz([])

    with pytest.raises(DocumentContentError):
        generate_quiz([{"page": 1, "text": "   "}])

    # Watermark-only scanned document
    scanned_pages = [{"page": i, "text": "arnav notes"} for i in range(1, 17)]
    with pytest.raises(DocumentContentError, match="does not contain sufficient readable text"):
        generate_quiz(scanned_pages)


def test_missing_model_configuration():
    """Client with no API key and mock_mode disabled must raise ConfigurationError."""
    import os
    orig_key = os.environ.pop("GEMMA_API_KEY", None)
    orig_gemini = os.environ.pop("GEMINI_API_KEY", None)
    try:
        client = GemmaAIClient(api_key="", mock_mode=False)
        with pytest.raises(ConfigurationError, match="Missing GEMMA_API_KEY"):
            client.generate_text("Test prompt")
    finally:
        if orig_key:
            os.environ["GEMMA_API_KEY"] = orig_key
        if orig_gemini:
            os.environ["GEMINI_API_KEY"] = orig_gemini


# =====================================================================
# End-to-End Learning Loop Flow Test
# =====================================================================

def test_full_learning_loop_flow(valid_doc_pages):
    """Test the complete learning loop:
    PDF pages -> diagnostic quiz -> submission -> evaluation -> weak topic ->
    teaching -> retest question -> retest evaluation -> updated mastery.
    """
    # 1. Generate diagnostic quiz
    quiz = generate_quiz(valid_doc_pages)
    assert len(quiz) == 5
    assert quiz[0]["id"] == "q1"

    # 2. Student submits answers (make q1 and q2 wrong in Conditional Probability)
    # q1 correct is 0 -> submit 1 (wrong)
    # q2 correct is 0 -> submit 2 (wrong)
    # q3 correct is 2 -> submit 2 (correct)
    # q4 correct is 0 -> submit 0 (correct)
    # q5 correct is 1 -> submit 1 (correct)
    answers = {"q1": 1, "q2": 2, "q3": 2, "q4": 0, "q5": 1}

    # 3. Deterministic evaluation
    eval_result = evaluate_quiz(quiz, answers)
    assert eval_result["correct_answers"] == 3
    assert eval_result["attempted_answers"] == 5
    assert eval_result["overall_accuracy"] == 60.0

    # 4. Weak topic selection
    weak_topic = choose_next_topic(eval_result["mastery"])
    assert weak_topic == "Conditional probability"

    # 5. Evidence-grounded teaching
    teach_result = teach_concept(weak_topic, valid_doc_pages)
    assert teach_result["concept"] == weak_topic
    assert teach_result["evidence_found"] is True
    assert len(teach_result["evidence"]) > 0
    assert 1 in teach_result["source_pages"]

    # 6. Generate targeted retest question
    retest_q = generate_retest(weak_topic, valid_doc_pages, previous_questions=quiz)
    assert retest_q["concept"] == weak_topic
    assert len(retest_q["options"]) == 4

    # 7. Evaluate retest question (student answers correctly)
    retest_eval = evaluate_retest(
        question=retest_q,
        answer_index=retest_q["correct_index"],
        previous_mastery=eval_result["mastery"]
    )
    assert retest_eval["is_correct"] is True
    # Initial mastery for Conditional Probability was 0% (0/2)
    assert retest_eval["previous_mastery"]["Conditional probability"]["accuracy"] == 0.0
    # Updated mastery should now be 1/3 = 33.3%
    assert retest_eval["updated_mastery"]["Conditional probability"]["correct"] == 1
    assert retest_eval["updated_mastery"]["Conditional probability"]["attempted"] == 3
    assert retest_eval["updated_mastery"]["Conditional probability"]["accuracy"] == 33.3


def test_conversational_tutor_actions(valid_doc_pages):
    """Verify conversational tutor supports specific actions (simplify, example, steps)."""
    concept = "Conditional probability"

    # 1. Simplify
    resp_simplify = teach_concept(concept, valid_doc_pages, action="simplify")
    assert resp_simplify["action"] == "simplify"
    assert resp_simplify["evidence_found"] is True
    assert 1 in resp_simplify["source_pages"]

    # 2. Example
    resp_example = teach_concept(concept, valid_doc_pages, action="example")
    assert resp_example["action"] == "example"
    assert resp_example["evidence_found"] is True

    # 3. Steps
    resp_steps = teach_concept(concept, valid_doc_pages, action="steps")
    assert resp_steps["action"] == "steps"
    assert resp_steps["evidence_found"] is True


def test_tutor_missing_evidence_honesty(valid_doc_pages):
    """If a concept has no matching text in the PDF, evidence_found must be False without inventing citations."""
    resp = teach_concept("Quantum entanglement superposition", valid_doc_pages)
    assert resp["evidence_found"] is False
    assert resp["evidence"] == ""
    assert resp["source_pages"] == []
    assert "could not find a direct, explicit passage" in resp["explanation"]
