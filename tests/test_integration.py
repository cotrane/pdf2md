"""Integration tests for PDF to Markdown conversion."""

import os
from pathlib import Path
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import pytest

from run import main


@pytest.mark.integration
@patch("parsers.googleai.genai.Client")
def test_end_to_end_conversion(
    mock_client_class: MagicMock,
    temp_pdf_file: Path,
    temp_output_dir: Path,
) -> None:
    """Test end-to-end PDF to Markdown conversion using Google AI.

    Args:
        mock_client_class: Mocked Google AI Client class.
        temp_pdf_file: Temporary PDF file fixture.
        temp_output_dir: Temporary output directory fixture.
    """
    # Create a proper string response
    mock_client = mock_client_class.return_value

    # Set up the response.text property to return a string
    response = Mock()
    type(response).text = PropertyMock(return_value="# Test Document\n\nHello World")

    # Configure the mock to return our response
    mock_client.models.generate_content.return_value = response

    # Mock the file upload method
    mock_client.files.upload.return_value = Mock()

    # Create a minimal PDF file with some text
    with open(temp_pdf_file, "wb") as f:
        f.write(
            b"%PDF-1.4\n1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n"
            b"2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n"
            b"3 0 obj\n<</Type /Page /Parent 2 0 R /Resources <<>> "
            b"/MediaBox [0 0 612 792] /Contents 4 0 R>>\nendobj\n"
            b"4 0 obj\n<</Length 44>>\n"
            b"stream\nBT\n/F1 12 Tf\n100 700 Td\n(Hello World) Tj\nET\nendstream\nendobj\n"
            b"xref\n0 5\n0000000000 65535 f\n0000000010 00000 n\n0000000056 00000 n\n"
            b"0000000102 00000 n\n0000000166 00000 n\n"
            b"trailer\n<</Size 5/Root 1 0 R>>\nstartxref\n226\n%%EOF"
        )

    # Mock the environment variable for the API key
    with patch.dict("os.environ", {"GOOGLE_API_KEY": "dummy_key"}):
        # Run the conversion
        with patch(
            "sys.argv",
            ["script.py", "-i", str(temp_pdf_file), "-o", str(temp_output_dir), "-p", "googleai"],
        ):
            main()

    # Verify output
    output_files = list(temp_output_dir.glob("*.md"))
    assert len(output_files) == 1
    output_file = output_files[0]
    assert output_file.exists()
    content = output_file.read_text()
    assert "Hello World" in content


@pytest.mark.integration
@patch("parsers.googleai.genai.Client")
def test_multiple_pages_conversion(
    mock_client_class: MagicMock,
    temp_pdf_file: Path,
    temp_output_dir: Path,
) -> None:
    """Test conversion of a multi-page PDF using Google AI.

    Args:
        mock_client_class: Mocked Google AI Client class.
        temp_pdf_file: Temporary PDF file fixture.
        temp_output_dir: Temporary output directory fixture.
    """
    # Create a proper mock client with proper string responses
    mock_client = mock_client_class.return_value

    # Create response objects with text properties
    response1 = Mock()
    type(response1).text = PropertyMock(return_value="# Page 1\n\nContent from page 1")

    response2 = Mock()
    type(response2).text = PropertyMock(return_value="# Page 2\n\nContent from page 2")

    # Set up the mock to return our responses in sequence
    mock_client.models.generate_content.side_effect = [response1, response2]

    # Mock the file upload method
    mock_client.files.upload.return_value = Mock()

    # Create a multi-page PDF file
    with open(temp_pdf_file, "wb") as f:
        f.write(
            b"%PDF-1.4\n1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n"
            b"2 0 obj\n<</Type /Pages /Kids [3 0 R 4 0 R] /Count 2>>\nendobj\n"
            b"3 0 obj\n<</Type /Page /Parent 2 0 R /Resources <<>> "
            b"/MediaBox [0 0 612 792] /Contents 5 0 R>>\nendobj\n"
            b"4 0 obj\n<</Type /Page /Parent 2 0 R /Resources <<>> "
            b"/MediaBox [0 0 612 792] /Contents 6 0 R>>\nendobj\n"
            b"5 0 obj\n<</Length 44>>\n"
            b"stream\nBT\n/F1 12 Tf\n100 700 Td\n(Page 1) Tj\nET\nendstream\nendobj\n"
            b"6 0 obj\n<</Length 44>>\n"
            b"stream\nBT\n/F1 12 Tf\n100 700 Td\n(Page 2) Tj\nET\nendstream\nendobj\n"
            b"xref\n0 7\n0000000000 65535 f\n0000000010 00000 n\n0000000056 00000 n\n"
            b"0000000102 00000 n\n0000000166 00000 n\n0000000220 00000 n\n"
            b"0000000274 00000 n\ntrailer\n<</Size 7/Root 1 0 R>>\nstartxref\n318\n%%EOF"
        )

    # Run the conversion with page splitting enabled
    with patch.dict("os.environ", {"GOOGLE_API_KEY": "dummy_key"}):
        with patch(
            "sys.argv",
            [
                "script.py",
                "-i",
                str(temp_pdf_file),
                "-o",
                str(temp_output_dir),
                "-p",
                "googleai",
                "--split-pages",
            ],
        ):
            main()

    # Verify output
    output_files = list(temp_output_dir.glob("*.md"))
    assert len(output_files) == 1
    output_file = output_files[0]
    assert output_file.exists()
    content = output_file.read_text()
    assert "Page 1" in content
    assert "Page 2" in content


@pytest.mark.integration
@patch("parsers.googleai.GoogleAIParser.generate_response")
@patch("parsers.googleai.GoogleAIParser.__init__")
def test_large_pdf_conversion(
    mock_init: MagicMock,
    mock_generate_response: MagicMock,
    temp_output_dir: Path,
) -> None:
    """Test conversion of a large PDF file using Google AI.

    Args:
        mock_init: Mocked GoogleAIParser initialization.
        mock_generate_response: Mocked generate_response method.
        temp_output_dir: Temporary output directory fixture.
    """
    # Setup mocks
    mock_init.return_value = None
    mock_generate_response.return_value = "# Large Document\n\nThis is a large document."

    large_pdf = Path("tests/data/large.pdf")
    if not large_pdf.exists():
        pytest.skip("Large PDF file not found")

    # Run the conversion
    with patch(
        "sys.argv",
        ["script.py", "-i", str(large_pdf), "-o", str(temp_output_dir), "-p", "googleai"],
    ):
        main()

    # Verify output
    output_files = list(temp_output_dir.glob("*.md"))
    assert len(output_files) == 1
    output_file = output_files[0]
    assert output_file.exists()
    content = output_file.read_text()
    assert len(content) > 0
