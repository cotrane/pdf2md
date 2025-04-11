"""Tests for the main script functionality."""

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from run import generate_output_path, get_parser, get_parser_help_text, main, setup_cli_args


@pytest.mark.unit
def test_get_parser() -> None:
    """Test getting parser instance."""
    parser = get_parser("textract")
    assert parser.__class__.__name__ == "TextractParser"


@pytest.mark.unit
def test_get_parser_invalid() -> None:
    """Test getting parser with invalid name."""
    with pytest.raises(ValueError):
        get_parser("invalid_parser")


@pytest.mark.unit
def test_generate_output_path() -> None:
    """Test generating output path."""
    output_path = generate_output_path("test.pdf", "output", "textract", "test-model")
    assert output_path == Path("output/test_textract_test-model.md")


@pytest.mark.unit
def test_generate_output_path_no_model() -> None:
    """Test generating output path without model."""
    output_path = generate_output_path("test.pdf", "output", "textract")
    assert output_path == Path("output/test_textract.md")


@pytest.mark.unit
def test_get_parser_help_text() -> None:
    """Test getting parser help text."""
    help_text = get_parser_help_text()
    assert "Available parsers and their models:" in help_text
    assert "textract:" in help_text


@pytest.mark.unit
@patch("argparse.ArgumentParser.parse_args")
def test_setup_cli_args(mock_parse_args: MagicMock) -> None:
    """Test setting up CLI arguments.

    Args:
        mock_parse_args: Mocked parse_args method.
    """
    mock_parse_args.return_value = argparse.Namespace(
        input_pdf_path="test.pdf",
        output_dir="output",
        parser="textract",
        model=None,
        log_level="INFO",
        split_pages=False,
    )
    args = setup_cli_args()
    assert args.input_pdf_path == "test.pdf"
    assert args.output_dir == "output"
    assert args.parser == "textract"
    assert args.model == ""
    assert args.log_level == "INFO"
    assert args.split_pages is False


@pytest.mark.unit
@patch("run.setup_cli_args")
@patch("run.get_parser")
@patch("run.generate_output_path")
def test_main(
    mock_generate_output_path: MagicMock,
    mock_get_parser: MagicMock,
    mock_setup_cli_args: MagicMock,
    temp_pdf_file: Path,
    temp_output_dir: Path,
) -> None:
    """Test main function.

    Args:
        mock_generate_output_path: Mocked generate_output_path function.
        mock_get_parser: Mocked get_parser function.
        mock_setup_cli_args: Mocked setup_cli_args function.
        temp_pdf_file: Temporary PDF file fixture.
        temp_output_dir: Temporary output directory fixture.
    """
    # Setup mocks
    mock_setup_cli_args.return_value = argparse.Namespace(
        input_pdf_path=str(temp_pdf_file),
        output_dir=str(temp_output_dir),
        parser="anthropic",
        model=None,
        log_level="INFO",
        split_pages=False,
    )

    mock_parser = MagicMock()
    mock_parser.convert_pdf_to_markdown.return_value = "Test markdown"
    mock_get_parser.return_value = mock_parser

    output_path = temp_output_dir / "test.md"
    mock_generate_output_path.return_value = output_path

    # Run main
    main()

    # Verify calls
    mock_setup_cli_args.assert_called_once()
    mock_get_parser.assert_called_once_with("anthropic", None)
    mock_parser.convert_pdf_to_markdown.assert_called_once_with(
        str(temp_pdf_file), split_pages=False
    )
    mock_parser.save_markdown.assert_called_once_with("Test markdown", str(output_path))
