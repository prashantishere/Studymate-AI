"""Lightweight, deterministic passage retrieval for StudyMate AI.

Retrieves relevant textual passages from extracted page dictionaries while
preserving exact text and original page numbers. No heavy vector database needed.
"""

import re
from typing import Any
from studymate.schemas import DocumentContentError


STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "about", "into", "over", "after",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "it", "its", "that", "this", "these", "those",
    "what", "which", "who", "whom", "how", "why", "when", "where",
}


def tokenize(text: str) -> list[str]:
    """Extract lowercase alphanumeric tokens, filtering out trivial single characters."""
    return [word for word in re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower()) if len(word) > 1]


def split_into_passages(page_dict: dict[str, Any], min_length: int = 30) -> list[dict[str, Any]]:
    """Split a page's text into readable paragraphs or sentence blocks.

    Args:
        page_dict: Dictionary with 'page' (int) and 'text' (str).
        min_length: Minimum character length for a passage to be considered meaningful.

    Returns:
        List of passage dictionaries: [{"page": int, "text": str}]
    """
    page_num = page_dict.get("page", 1)
    raw_text = page_dict.get("text", "")
    if not isinstance(raw_text, str) or not raw_text.strip():
        return []

    # Split by double newline first, then single newline if blocks are too few
    raw_blocks = re.split(r"\n\s*\n", raw_text)
    passages: list[dict[str, Any]] = []

    for block in raw_blocks:
        clean_block = " ".join(block.split()).strip()
        if len(clean_block) >= min_length:
            passages.append({"page": page_num, "text": clean_block})
        elif clean_block:
            # If the whole page text is shorter than min_length, keep it anyway
            if len(raw_blocks) == 1:
                passages.append({"page": page_num, "text": clean_block})

    # Fallback: if no blocks met min_length but text is present, return the whole text
    if not passages and raw_text.strip():
        passages.append({"page": page_num, "text": " ".join(raw_text.split()).strip()})

    return passages


def retrieve_passages(
    pages: list[dict[str, Any]],
    concept: str,
    limit: int = 4
) -> list[dict[str, Any]]:
    """Retrieve the most relevant passages for a given concept across all pages.

    Args:
        pages: List of extracted page dictionaries [{"page": 1, "text": "..."}].
        concept: The concept string to find evidence for.
        limit: Maximum number of passages to return (default 4).

    Returns:
        List of passage dictionaries [{"page": int, "text": str}], sorted by relevance.

    Raises:
        DocumentContentError: If pages list is empty or invalid.
    """
    if not pages or not isinstance(pages, list):
        raise DocumentContentError("Cannot retrieve passages from an empty or invalid document.")

    clean_concept = concept.strip().lower()
    if not clean_concept:
        return []

    concept_tokens = [t for t in tokenize(clean_concept) if t not in STOP_WORDS]
    if not concept_tokens:
        concept_tokens = tokenize(clean_concept)

    scored_passages: list[tuple[float, dict[str, Any]]] = []

    for p in pages:
        page_passages = split_into_passages(p)
        for passage in page_passages:
            text = passage["text"]
            text_lower = text.lower()
            tokens = set(tokenize(text))

            score = 0.0

            # 1. Exact phrase match bonus
            if clean_concept in text_lower:
                score += 15.0

            # 2. Token overlap score
            matched_tokens = set(concept_tokens).intersection(tokens)
            score += len(matched_tokens) * 3.0

            # 3. Frequency bonus
            for token in matched_tokens:
                score += min(text_lower.count(token), 3) * 0.5

            if score > 0.0:
                scored_passages.append((score, passage))

    # Sort descending by score, tie-break by page number ascending
    scored_passages.sort(key=lambda item: (-item[0], item[1]["page"]))

    # Return top unique passages
    results: list[dict[str, Any]] = []
    seen_texts: set[str] = set()

    for _, passage in scored_passages:
        if passage["text"] not in seen_texts:
            seen_texts.add(passage["text"])
            results.append(passage)
            if len(results) >= limit:
                break

    return results


def format_passages_for_prompt(passages: list[dict[str, Any]]) -> str:
    """Format retrieved passages into a structured text block for the LLM prompt."""
    if not passages:
        return "No direct evidence passages found."

    formatted_blocks = []
    for idx, p in enumerate(passages, start=1):
        formatted_blocks.append(f"[Source {idx} - Page {p['page']}]:\n\"{p['text']}\"")
    return "\n\n".join(formatted_blocks)
