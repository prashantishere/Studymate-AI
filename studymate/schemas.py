"""Shared data schemas and validation helpers for StudyMate AI.

This module defines standard error classes and deterministic validators for
quiz questions, teaching explanations, and mastery data.
"""

from typing import Any
from pydantic import BaseModel, Field, field_validator


# =====================================================================
# Error Hierarchy
# =====================================================================

class StudyMateError(Exception):
    """Base exception for all StudyMate errors."""
    pass


class ConfigurationError(StudyMateError):
    """Raised when configuration or environment variables are missing/invalid."""
    pass


class ModelRequestError(StudyMateError):
    """Raised when an inference call to the model provider fails or times out."""
    pass


class InvalidModelResponseError(StudyMateError):
    """Raised when the model output is malformed or violates the required schema."""
    pass


class DocumentContentError(StudyMateError):
    """Raised when the provided document text is empty, invalid, or insufficient."""
    pass


class GradingError(StudyMateError):
    """Raised when submitted answers or grading arguments are invalid."""
    pass


# =====================================================================
# Question & Response Schemas
# =====================================================================

class QuestionSchema(BaseModel):
    """Schema for a single multiple-choice question."""
    id: str = Field(..., description="Unique question identifier, e.g. 'q1'")
    concept: str = Field(..., min_length=1, description="Specific concept tested")
    question: str = Field(..., min_length=5, description="Question text")
    options: list[str] = Field(..., description="Exactly four non-empty answer options")
    correct_index: int = Field(..., ge=0, le=3, description="Index of the correct answer (0-3)")
    explanation: str = Field(..., min_length=1, description="Explanation for the correct answer")
    source_pages: list[int] = Field(default_factory=list, description="Original 1-indexed source pages")

    @field_validator("options")
    @classmethod
    def validate_options(cls, v: list[str]) -> list[str]:
        if len(v) != 4:
            raise ValueError(f"Question must have exactly 4 options, got {len(v)}")
        for idx, opt in enumerate(v):
            if not isinstance(opt, str) or not opt.strip():
                raise ValueError(f"Option {idx} is empty or not a valid string")
        return [opt.strip() for opt in v]

    @field_validator("source_pages")
    @classmethod
    def validate_source_pages(cls, v: list[int]) -> list[int]:
        for page in v:
            if not isinstance(page, int) or page < 1:
                raise ValueError(f"Invalid source page number: {page}. Pages must be positive integers.")
        return sorted(list(set(v)))


class QuickCheckQuestion(BaseModel):
    """Optional formative check question posed by the tutor."""
    question: str = Field(..., min_length=5)
    options: list[str] = Field(..., min_length=2)
    correct_index: int = Field(..., ge=0)
    explanation: str = Field(..., min_length=1)


class TeachingResponseSchema(BaseModel):
    """Schema for a conversational, evidence-grounded concept explanation."""
    concept: str = Field(..., min_length=1)
    explanation: str = Field(..., min_length=1)
    evidence: str = Field(default="")
    source_pages: list[int] = Field(default_factory=list)
    evidence_found: bool = Field(default=True)
    action: str = Field(default="initial_explanation")
    intuitive_intro: str = Field(default="")
    example: str = Field(default="")
    formal_connection: str = Field(default="")
    check_question: QuickCheckQuestion | None = Field(default=None)

    @field_validator("source_pages")
    @classmethod
    def validate_source_pages(cls, v: list[int]) -> list[int]:
        for page in v:
            if not isinstance(page, int) or page < 1:
                raise ValueError(f"Invalid source page number: {page}. Pages must be positive integers.")
        return sorted(list(set(v)))


def validate_teaching_response(
    resp_dict: dict[str, Any],
    valid_page_numbers: set[int] | None = None
) -> dict[str, Any]:
    """Validate a teaching response against TeachingResponseSchema.

    Args:
        resp_dict: Dictionary representing tutor output.
        valid_page_numbers: Optional set of valid 1-indexed page numbers.

    Returns:
        Validated dictionary.

    Raises:
        InvalidModelResponseError: If schema validation fails or invalid pages are cited.
    """
    if not isinstance(resp_dict, dict):
        raise InvalidModelResponseError(f"Expected dict, got {type(resp_dict).__name__}")

    try:
        validated = TeachingResponseSchema(**resp_dict)
    except Exception as exc:
        raise InvalidModelResponseError(f"Invalid teaching response: {exc}") from exc

    if valid_page_numbers is not None and validated.source_pages:
        for page in validated.source_pages:
            if page not in valid_page_numbers:
                raise InvalidModelResponseError(
                    f"Teaching response cited page {page}, which is not in valid pages {sorted(valid_page_numbers)}"
                )

    return validated.model_dump()


# =====================================================================
# Validation Functions
# =====================================================================

def validate_question_dict(q_dict: dict[str, Any], valid_page_numbers: set[int] | None = None) -> dict[str, Any]:
    """Validate a single question dictionary against QuestionSchema.

    Args:
        q_dict: Dictionary representing a question.
        valid_page_numbers: Optional set of allowable 1-indexed page numbers.

    Returns:
        Validated question dictionary.

    Raises:
        InvalidModelResponseError: If validation fails.
    """
    if not isinstance(q_dict, dict):
        raise InvalidModelResponseError(f"Expected question dictionary, got {type(q_dict).__name__}")

    try:
        validated = QuestionSchema(**q_dict)
    except Exception as exc:
        raise InvalidModelResponseError(f"Invalid question schema: {exc}") from exc

    # Check that source pages exist in document
    if valid_page_numbers is not None and validated.source_pages:
        for page in validated.source_pages:
            if page not in valid_page_numbers:
                raise InvalidModelResponseError(
                    f"Question '{validated.id}' references page {page}, which is not in the document pages {sorted(valid_page_numbers)}"
                )

    return validated.model_dump()


def validate_quiz_list(
    questions: list[dict[str, Any]],
    valid_page_numbers: set[int] | None = None,
    expected_count: int = 5,
) -> list[dict[str, Any]]:
    """Validate a list of questions for a diagnostic quiz.

    Args:
        questions: List of raw question dictionaries.
        valid_page_numbers: Set of valid page numbers from the PDF.
        expected_count: Number of questions expected (default 5).

    Returns:
        List of validated question dictionaries.

    Raises:
        InvalidModelResponseError: If question count is incorrect, IDs are duplicated,
                                   or any question violates the schema.
    """
    if not isinstance(questions, list):
        raise InvalidModelResponseError(f"Expected list of questions, got {type(questions).__name__}")

    if len(questions) != expected_count:
        raise InvalidModelResponseError(
            f"Expected exactly {expected_count} questions, got {len(questions)}"
        )

    validated_questions: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for idx, q_raw in enumerate(questions):
        validated = validate_question_dict(q_raw, valid_page_numbers=valid_page_numbers)
        qid = validated["id"]
        if qid in seen_ids:
            raise InvalidModelResponseError(f"Duplicate question ID found: '{qid}' at index {idx}")
        seen_ids.add(qid)
        validated_questions.append(validated)

    return validated_questions


def sanitize_for_student(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip correct answers and explanations so the client cannot see the answer key.

    Args:
        questions: List of full question dictionaries.

    Returns:
        List of safe question dictionaries with id, concept, question, options, and source_pages.
    """
    safe_questions: list[dict[str, Any]] = []
    for q in questions:
        safe_q = {
            "id": q["id"],
            "concept": q["concept"],
            "question": q["question"],
            "options": list(q["options"]),
            "source_pages": list(q.get("source_pages", [])),
        }
        safe_questions.append(safe_q)
    return safe_questions
