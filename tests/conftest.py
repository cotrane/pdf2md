"""Common test fixtures for the PDF to Markdown conversion tests."""

import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@pytest.fixture
def temp_pdf_file() -> Generator[Path, None, None]:
    """Create a temporary PDF file for testing.

    Yields:
        Path: Path to the temporary PDF file.
    """
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
        # Create a minimal PDF file
        temp_file.write(
            b"%PDF-1.4\n1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n"
            b"2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n"
            b"3 0 obj\n<</Type /Page /Parent 2 0 R /Resources <<>> "
            b"/MediaBox [0 0 612 792]>>\nendobj\n"
            b"xref\n0 4\n0000000000 65535 f\n0000000010 00000 n\n0000000056 00000 n\n"
            b"0000000102 00000 n\ntrailer\n<</Size 4/Root 1 0 R>>\nstartxref\n149\n%%EOF"
        )
        temp_file.flush()
        yield Path(temp_file.name)
        os.unlink(temp_file.name)


@pytest.fixture
def temp_output_dir() -> Generator[Path, None, None]:
    """Create a temporary output directory for testing.

    Yields:
        Path: Path to the temporary output directory.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_markdown() -> str:
    """Return a sample markdown text for testing.

    Returns:
        str: Sample markdown text.
    """
    return """# Test Document

## Section 1

This is a test paragraph.

### Subsection 1.1

- Item 1
- Item 2
- Item 3

## Section 2

| Header 1 | Header 2 |
|----------|----------|
| Value 1  | Value 2  |
"""
