"""Gemini parser for PDF to Markdown conversion."""

import os
from pathlib import Path

import openai
from markitdown import MarkItDown

from .base import BaseParser


class MarkitdownParser(BaseParser):
    """A class to handle PDF to Markdown conversion using Microsoft's Markitdown."""

    AVAILABLE_MODELS = [
        "pdfminer-six",
        "gpt-4o",
    ]
    DEFAULT_MODEL = "pdfminer-six"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
    ) -> None:
        """Initialize the Markitdown parser.

        Args:
            model: The name of the Markitdown model to use. By default will only use
                pdfminer-six to extract text from the PDF. If a model is given, it will
                use this model to extract image descriptions.

        Raises:
            ValueError: If the OPENAI_API_KEY environment variable is not set.
            ValueError: If the model is not supported.
        """
        if model not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Unsupported model: {model}. Available models: {', '.join(self.AVAILABLE_MODELS)}"
            )

        super().__init__(model)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        if model != "pdfminer-six":
            openai.api_key = api_key
            client = openai.OpenAI()
            self.client = MarkItDown(llm_client=client, llm_model=model)
        else:
            self.client = MarkItDown(enable_plugins=False)
        self.logger.debug("Markitdown parser initialization complete")

    def generate_response(self, pdf_path: str) -> str:
        """Generate a response from the Markitdown API.

        Args:
            pdf_path (str): The path to the PDF file to generate a response for.

        Returns:
            str: The generated response.

        Raises:
            Exception: If there's an error generating the response.
        """
        response = self.client.convert(pdf_path)
        return response.text_content  # type: ignore

    def convert_pdf_to_markdown(self, pdf_path: str, *, split_pages: bool = False) -> str:
        """Convert a PDF file to markdown using Markitdown.

        Args:
            pdf_path: The path to the PDF file to convert.
            split_pages: Whether to split the PDF into pages. Default is False.
                Not used in this parser.

        Returns:
            The converted markdown text.

        Raises:
            FileNotFoundError: If the PDF file doesn't exist.
            IOError: If there's an error reading the PDF file.
        """
        self.logger.debug(f"Starting PDF to Markdown conversion for file: {pdf_path}")
        self.validate_pdf_path(pdf_path)

        try:
            markdown_text = self.generate_response(pdf_path)
            self.logger.debug("Successfully received response from Markitdown API")
            return markdown_text

        except Exception as e:
            self.logger.error(f"Error during PDF conversion: {str(e)}", exc_info=True)
            raise
