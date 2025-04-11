"""Tests for the base parser functionality."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from parsers.base import BaseParser


class TestBaseParser(BaseParser):
    """Test implementation of BaseParser for testing."""

    AVAILABLE_MODELS = ["test-model"]
    DEFAULT_MODEL = "test-model"

    def convert_pdf_to_markdown(self, pdf_path: str) -> str:
        """Test implementation of convert_pdf_to_markdown."""
        return "Test markdown content"


@pytest.mark.unit
def test_validate_pdf_path(temp_pdf_file: Path) -> None:
    """Test PDF path validation with valid PDF file.

    Args:
        temp_pdf_file: Temporary PDF file fixture.
    """
    parser = TestBaseParser("test-model")
    parser.validate_pdf_path(str(temp_pdf_file))


@pytest.mark.unit
def test_validate_pdf_path_nonexistent() -> None:
    """Test PDF path validation with nonexistent file."""
    parser = TestBaseParser("test-model")
    with pytest.raises(FileNotFoundError):
        parser.validate_pdf_path("nonexistent.pdf")


@pytest.mark.unit
def test_validate_pdf_path_not_pdf(temp_output_dir: Path) -> None:
    """Test PDF path validation with non-PDF file.

    Args:
        temp_output_dir: Temporary output directory fixture.
    """
    parser = TestBaseParser("test-model")
    not_pdf = temp_output_dir / "not_pdf.txt"
    not_pdf.write_text("Not a PDF")
    with pytest.raises(ValueError):
        parser.validate_pdf_path(str(not_pdf))


@pytest.mark.unit
def test_read_pdf_as_base64(temp_pdf_file: Path) -> None:
    """Test reading PDF as base64.

    Args:
        temp_pdf_file: Temporary PDF file fixture.
    """
    parser = TestBaseParser("test-model")
    base64_content = parser.read_pdf_as_base64(str(temp_pdf_file))
    assert isinstance(base64_content, str)
    assert base64_content.startswith("JVBERi0x")


@pytest.mark.unit
def test_save_markdown(temp_output_dir: Path, sample_markdown: str) -> None:
    """Test saving markdown to file.

    Args:
        temp_output_dir: Temporary output directory fixture.
        sample_markdown: Sample markdown text fixture.
    """
    parser = TestBaseParser("test-model")
    output_path = temp_output_dir / "test.md"
    parser.save_markdown(sample_markdown, str(output_path))
    assert output_path.exists()
    assert output_path.read_text() == sample_markdown


@pytest.mark.unit
def test_split_pdf_into_pages(temp_pdf_file: Path) -> None:
    """Test splitting PDF into pages.

    Args:
        temp_pdf_file: Temporary PDF file fixture.
    """
    parser = TestBaseParser("test-model")
    page_files = parser.split_pdf_into_pages(str(temp_pdf_file))
    assert len(page_files) == 1
    assert all(Path(f).exists() for f in page_files)
    # Clean up
    for f in page_files:
        os.unlink(f)


@pytest.mark.unit
def test_get_pdf_page_count(temp_pdf_file: Path) -> None:
    """Test getting PDF page count.

    Args:
        temp_pdf_file: Temporary PDF file fixture.
    """
    parser = TestBaseParser("test-model")
    page_count = parser.get_pdf_page_count(str(temp_pdf_file))
    assert page_count == 1


@pytest.mark.unit
@patch("parsers.base.fitz")
def test_get_pdf_page_count_error(mock_fitz: MagicMock) -> None:
    """Test error handling in get_pdf_page_count.

    Args:
        mock_fitz: Mocked fitz module.
    """
    mock_fitz.open.side_effect = Exception("Test error")
    parser = TestBaseParser("test-model")
    with pytest.raises(Exception):
        parser.get_pdf_page_count("test.pdf")


@pytest.mark.unit
def test_resize_image() -> None:
    """Test image resizing functionality.

    Tests that:
    1. Images wider than MAX_IMAGE_WIDTH are resized while maintaining aspect ratio
    2. Images narrower than MAX_IMAGE_WIDTH are returned unchanged
    """
    parser = TestBaseParser("test-model")

    # Create a test image wider than MAX_IMAGE_WIDTH
    wide_image = Image.new("RGB", (parser.MAX_IMAGE_WIDTH + 100, 500))
    resized_image = parser.resize_image(wide_image)
    assert resized_image.width == parser.MAX_IMAGE_WIDTH
    assert resized_image.height == int(
        500 * (parser.MAX_IMAGE_WIDTH / (parser.MAX_IMAGE_WIDTH + 100))
    )

    # Create a test image narrower than MAX_IMAGE_WIDTH
    narrow_image = Image.new("RGB", (parser.MAX_IMAGE_WIDTH - 100, 500))
    unchanged_image = parser.resize_image(narrow_image)
    assert unchanged_image.width == parser.MAX_IMAGE_WIDTH - 100
    assert unchanged_image.height == 500


@pytest.mark.unit
@patch("parsers.base.convert_from_path")
def test_read_pdf_as_base64_img(mock_convert: MagicMock) -> None:
    """Test converting PDF pages to base64-encoded images.

    Args:
        mock_convert: Mocked convert_from_path function.
    """
    # Create test images
    test_images = [Image.new("RGB", (100, 100)), Image.new("RGB", (100, 100))]
    mock_convert.return_value = test_images

    parser = TestBaseParser("test-model")
    base64_images = list(parser.read_pdf_as_base64_img("test.pdf"))

    # Verify results
    assert len(base64_images) == 2
    for img_str in base64_images:
        assert isinstance(img_str, str)
        assert img_str.startswith("iVBORw0KGgo")  # PNG base64 header


@pytest.mark.unit
@patch("parsers.base.convert_from_path")
def test_read_pdf_as_base64_img_error(mock_convert: MagicMock) -> None:
    """Test error handling in read_pdf_as_base64_img.

    Args:
        mock_convert: Mocked convert_from_path function.
    """
    mock_convert.side_effect = Exception("Test error")
    parser = TestBaseParser("test-model")
    with pytest.raises(Exception):
        list(parser.read_pdf_as_base64_img("test.pdf"))
