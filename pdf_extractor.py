"""PDF Text Extraction Module for StudyMate AI.

Implements the contract agreed upon with the PDF teammate:
    extract_pdf_text(pdf_bytes: bytes) -> list[dict]

Extracts text page by page, preserving original one-indexed page numbers.
Empty pages do not shift page indices. Invalid or unreadable PDFs raise clear errors.
"""

import io
from typing import Any
import pypdf

from studymate.schemas import DocumentContentError


def extract_pdf_text(pdf_bytes: bytes) -> list[dict[str, Any]]:
    """Extract page-aware text from PDF raw bytes.

    Args:
        pdf_bytes: Raw bytes of the uploaded PDF file.

    Returns:
        List of dictionaries containing one-indexed page numbers and extracted text:
        [
            {"page": 1, "text": "Text from page one..."},
            {"page": 2, "text": "Text from page two..."}
        ]

    Raises:
        DocumentContentError: If pdf_bytes is empty, corrupt, password-protected without
                              access, or yields zero extractable text across all pages.
    """
    if not pdf_bytes or not isinstance(pdf_bytes, bytes):
        raise DocumentContentError("No PDF file content was provided or file is empty.")

    try:
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    except Exception as exc:
        raise DocumentContentError(f"Failed to parse PDF document. File may be corrupted: {exc}") from exc

    if reader.is_encrypted:
        try:
            # Try empty password default
            reader.decrypt("")
        except Exception as exc:
            raise DocumentContentError(
                "PDF is encrypted or password-protected and cannot be read."
            ) from exc

    total_pages = len(reader.pages)
    if total_pages == 0:
        raise DocumentContentError("The provided PDF contains no pages.")

    extracted_pages: list[dict[str, Any]] = []
    has_any_text = False

    for idx, page in enumerate(reader.pages):
        page_num = idx + 1  # 1-indexed
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""

        clean_text = page_text.strip()
        if clean_text:
            has_any_text = True

        extracted_pages.append({
            "page": page_num,
            "text": clean_text
        })

    if not has_any_text:
        raise DocumentContentError(
            "The PDF contains no readable text. It might be a scanned image or empty. "
            "Please provide a document containing selectable text."
        )

    return extracted_pages
