"""Unit tests for passage retrieval and source preservation."""

import pytest
from studymate.retrieval import (
    format_passages_for_prompt,
    retrieve_passages,
    split_into_passages,
)
from studymate.schemas import DocumentContentError


@pytest.fixture
def sample_pages():
    return [
        {
            "page": 1,
            "text": (
                "Introduction to Machine Learning. This chapter discusses linear regression "
                "and cost functions. Optimization is performed using gradient descent."
            ),
        },
        {
            "page": 2,
            "text": (
                "Supervised Learning and Classification. Decision trees partition feature space. "
                "Random forests combine multiple decision trees via bagging to reduce variance."
            ),
        },
        {
            "page": 3,
            "text": (
                "Neural Networks. Backpropagation computes gradients of the loss function "
                "with respect to each weight using the chain rule of calculus."
            ),
        },
    ]


def test_retrieval_returns_original_page_numbers(sample_pages):
    """Retrieved passages must retain their exact original one-indexed page numbers."""
    passages = retrieve_passages(sample_pages, "random forests bagging", limit=2)
    assert len(passages) >= 1
    # Page must be page 2
    assert passages[0]["page"] == 2
    assert "Random forests" in passages[0]["text"]


def test_retrieval_multi_page_concept(sample_pages):
    """Concept appearing across pages should rank by relevance and keep page IDs."""
    passages = retrieve_passages(sample_pages, "gradient", limit=2)
    assert len(passages) >= 2
    pages_found = {p["page"] for p in passages}
    assert 1 in pages_found  # page 1 has "gradient descent"
    assert 3 in pages_found  # page 3 has "Backpropagation computes gradients"


def test_retrieval_empty_document():
    """Empty or None page lists must raise DocumentContentError."""
    with pytest.raises(DocumentContentError):
        retrieve_passages([], "any concept")

    with pytest.raises(DocumentContentError):
        retrieve_passages(None, "any concept")


def test_retrieval_no_match(sample_pages):
    """A concept completely absent from the text should return an empty list."""
    passages = retrieve_passages(sample_pages, "quantum teleportation entanglement")
    assert passages == []


def test_format_passages_for_prompt(sample_pages):
    """Check that passages are formatted cleanly with page tags."""
    passages = [
        {"page": 2, "text": "Sample text from page two."},
        {"page": 5, "text": "Sample text from page five."},
    ]
    formatted = format_passages_for_prompt(passages)
    assert "[Source 1 - Page 2]:" in formatted
    assert "[Source 2 - Page 5]:" in formatted
    assert "Sample text from page two." in formatted
