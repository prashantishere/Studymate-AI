"""Unit tests for pdf_extractor module."""

import io
import pytest
import pypdf
from pdf_extractor import extract_pdf_text
from studymate.schemas import DocumentContentError


def create_sample_pdf_bytes(page_texts: list[str]) -> bytes:
    """Helper to generate an in-memory PDF with specified page texts."""
    writer = pypdf.PdfWriter()
    for text in page_texts:
        # Create a blank page and add annotations or text using pypdf
        page = writer.add_blank_page(width=612, height=792)
    stream = io.BytesIO()
    writer.write(stream)
    return stream.getvalue()


def test_extract_empty_bytes():
    """Empty bytes input must raise DocumentContentError."""
    with pytest.raises(DocumentContentError, match="No PDF file content"):
        extract_pdf_text(b"")


def test_extract_invalid_pdf():
    """Non-PDF corrupted bytes must raise DocumentContentError."""
    with pytest.raises(DocumentContentError, match="Failed to parse PDF"):
        extract_pdf_text(b"not a valid pdf header")


def test_extract_blank_pages_raises_no_text():
    """PDF with zero extractable text must raise DocumentContentError."""
    blank_pdf = create_sample_pdf_bytes(["", ""])
    with pytest.raises(DocumentContentError, match="contains no readable text"):
        extract_pdf_text(blank_pdf)
