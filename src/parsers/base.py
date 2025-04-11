"""Base parser class for PDF to Markdown conversion."""

import base64
import io
import logging
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator

import fitz  # PyMuPDF
from pdf2image import convert_from_path
from PIL import Image


class BaseParser(ABC):
    """Base class for PDF to Markdown parsers."""

    MAX_IMAGE_WIDTH = 1920
    SYSTEM_PROMPT = """
    You are a specialized PDF to Markdown converter. Your task is to convert PDF content to clean, properly formatted markdown text. Follow these strict rules:

    1. Output ONLY the markdown text without any code block markers or other decorators

    2. Format tables with precise attention to structure:
       - Preserve multi-row headers exactly as they appear in the PDF
       - Keep column headers that span multiple rows in their original format
       - Left-align text columns (`:---`)
       - Right-align number and currency columns (`---:`)
       - Maintain exact column widths and alignments from the PDF
       - For financial tables:
         * Currency symbols (€) should be in the header row only
         * Numbers should be right-aligned with consistent decimal places
         * Negative numbers should be in parentheses
         * Preserve any underlines or separators between sections

    3. Use proper heading hierarchy:
       - # for document title
       - ## for major sections
       - ### for subsections
       - #### for minor headings
       - Preserve any title continuations (e.g., "continued") in the same format

    4. Preserve all formatting:
       - **bold text** stays bold
       - *italic text* stays italic
       - Lists with proper indentation and markers (-, *, numbers)
       - Keep exact spacing between sections as in the PDF

    5. Handle special elements:
       - Footnotes using standard markdown footnote syntax [^1]
       - Block quotes with proper > markers
       - Mathematical formulas using LaTeX syntax
       - Preserve any special characters or symbols exactly as they appear

    6. Maintain document structure:
       - Keep all content in its original order
       - Preserve page numbers as level-6 headers (###### Page X)
       - Maintain all paragraph breaks and section spacing
       - Keep any header/footer information in its original location

    Remember: Your output should be pure markdown text that perfectly mirrors the PDF's layout and formatting.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.1,
        top_p: float = 0.1,
        top_k: int = 40,
        max_tokens: int = 8192,
    ) -> None:
        """Initialize the base parser.

        Args:
            model: The name of the model to use.
            temperature: The temperature parameter for text generation.
            top_p: The top_p parameter for text generation.
            top_k: The top_k parameter for text generation.
            max_tokens: The maximum number of tokens to generate.
        """
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def convert_pdf_to_markdown(self, pdf_path: str) -> str:
        """Convert a PDF file to markdown.

        Args:
            pdf_path: The path to the PDF file to convert.

        Returns:
            The converted markdown text.

        Raises:
            FileNotFoundError: If the PDF file doesn't exist.
            IOError: If there's an error reading the PDF file.
        """

    def validate_pdf_path(self, pdf_path: str) -> None:
        """Validate that the PDF file exists and is readable.

        Args:
            pdf_path: The path to the PDF file to validate.

        Raises:
            FileNotFoundError: If the PDF file doesn't exist.
            IOError: If there's an error reading the PDF file.
            ValueError: If the file is not a PDF.
        """
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        if not pdf_file.is_file():
            raise IOError(f"Path is not a file: {pdf_path}")
        if pdf_file.suffix.lower() != ".pdf":
            raise ValueError(f"File is not a PDF: {pdf_path}")

    def read_pdf_as_base64(self, pdf_path: str) -> str:
        """Read a PDF file and convert it to base64.

        Args:
            pdf_path: The path to the PDF file to read.

        Returns:
            The base64-encoded PDF content.

        Raises:
            FileNotFoundError: If the PDF file doesn't exist.
            IOError: If there's an error reading the PDF file.
        """
        self.validate_pdf_path(pdf_path)
        pdf_file = Path(pdf_path)
        with open(pdf_file, "rb") as f:
            binary_data = f.read()
            base64_encoded_data = base64.standard_b64encode(binary_data)
            return base64_encoded_data.decode("utf-8")

    def resize_image(self, image: Image.Image) -> Image.Image:
        """Resize image to maintain aspect ratio with max width.

        Args:
            image: PIL Image to resize.

        Returns:
            Resized PIL Image.
        """
        if image.width > self.MAX_IMAGE_WIDTH:
            ratio = self.MAX_IMAGE_WIDTH / image.width
            new_height = int(image.height * ratio)
            return image.resize((self.MAX_IMAGE_WIDTH, new_height), Image.Resampling.LANCZOS)
        return image

    def read_pdf_as_base64_img(self, pdf_path: str) -> Generator[str, None, None]:
        """Convert PDF pages to base64-encoded images.

        Args:
            pdf_path: Path to the PDF file.

        Raises:
            FileNotFoundError: If the PDF file doesn't exist.
            IOError: If there's an error reading the PDF file.

        Returns:
            List[str]: List of base64-encoded image strings.
        """
        try:
            # Convert PDF to images
            images = convert_from_path(pdf_path)

            # Convert each image to base64
            for image in images:
                # Resize image if needed
                image = self.resize_image(image)

                img_byte_io = io.BytesIO()
                image.save(img_byte_io, format="PNG")
                img_byte_arr = img_byte_io.getvalue()

                yield base64.b64encode(img_byte_arr).decode("utf-8")
        except Exception as e:
            self.logger.error(f"Error converting PDF to images: {str(e)}", exc_info=True)
            raise

    def save_markdown(self, markdown_text: str, output_path: str) -> None:
        """Save the markdown text to a file.

        Args:
            markdown_text: The markdown text to save.
            output_path: The path to save the markdown file to.

        Raises:
            IOError: If there's an error writing the file.
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(markdown_text)
            self.logger.debug(f"Successfully saved markdown to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving markdown file: {str(e)}")
            raise

    def split_pdf_into_pages(self, pdf_path: str) -> list[str]:
        """Split a multi-page PDF into separate single-page PDFs.

        Args:
            pdf_path (str): The path to the PDF file to split.

        Returns:
            list[str]: A list of paths to the created single-page PDF files.

        Raises:
            FileNotFoundError: If the PDF file doesn't exist.
            IOError: If there's an error reading or writing the PDF files.
            ValueError: If the file is not a PDF.
        """
        self.validate_pdf_path(pdf_path)

        pdf_file = Path(pdf_path)
        # Always use system temp directory
        output_path = Path(tempfile.gettempdir()) / "pdf2md_pages"
        output_path.mkdir(parents=True, exist_ok=True)

        output_files: list[str] = []
        try:
            # Open the PDF
            doc = fitz.open(pdf_path)

            # Split each page into a separate PDF
            for page_num in range(len(doc)):
                # Create a new PDF for this page
                new_doc = fitz.open()
                new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)

                # Generate output filename with a unique identifier
                output_file = (
                    output_path
                    / f"{pdf_file.stem}_page_{page_num + 1}_{tempfile.gettempprefix()}.pdf"
                )

                # Save the single-page PDF
                new_doc.save(str(output_file))
                new_doc.close()

                output_files.append(str(output_file))

            doc.close()
            self.logger.debug(
                f"Successfully split PDF into {len(output_files)} pages in {output_path}"
            )
            return output_files

        except Exception as e:
            self.logger.error(f"Error splitting PDF: {str(e)}")
            raise

    def get_pdf_page_count(self, pdf_path: str) -> int:
        """Get the number of pages in a PDF file.

        Args:
            pdf_path (str): The path to the PDF file.

        Returns:
            int: The number of pages in the PDF.

        Raises:
            FileNotFoundError: If the PDF file doesn't exist.
            IOError: If there's an error reading the PDF file.
            ValueError: If the file is not a PDF.
        """
        self.validate_pdf_path(pdf_path)

        try:
            doc = fitz.open(pdf_path)
            page_count = len(doc)
            doc.close()
            return page_count
        except Exception as e:
            self.logger.error(f"Error getting PDF page count: {str(e)}")
            raise
