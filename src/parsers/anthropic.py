"""Anthropic parser for PDF to Markdown conversion."""

import base64
import os
from pathlib import Path

import anthropic

from .base import BaseParser


class AnthropicParser(BaseParser):
    """A class to handle PDF to Markdown conversion using Anthropic's models."""

    AVAILABLE_MODELS = [
        "claude-3-7-sonnet-20250219",
        "claude-3-5-sonnet-20241022",
    ]
    DEFAULT_MODEL = "claude-3-7-sonnet-20250219"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.1,
        top_p: float = 0.1,
        top_k: int = 40,
        max_tokens: int = 8192,
    ) -> None:
        """Initialize the Anthropic parser.

        Args:
            model: The name of the Anthropic model to use.
            temperature: The temperature parameter for text generation.
            top_p: The top_p parameter for text generation.
            top_k: The top_k parameter for text generation.
            max_tokens: The maximum number of tokens to generate.

        Raises:
            ValueError: If the ANTHROPIC_API_KEY environment variable is not set.
            ValueError: If the model is not supported.
        """
        if model not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Unsupported model: {model}. Available models: {', '.join(self.AVAILABLE_MODELS)}"
            )

        super().__init__(model, temperature, top_p, top_k, max_tokens)

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.logger.debug("Anthropic parser initialization complete")

    def convert_pdf_to_markdown(self, pdf_path: str, *, split_pages: bool = True) -> str:
        """Convert a PDF file to markdown using Anthropic.

        Args:
            pdf_path: The path to the PDF file to convert.
            split_pages: Whether to split the PDF into pages. Default is True.
        Returns:
            The converted markdown text.

        Raises:
            FileNotFoundError: If the PDF file doesn't exist.
            IOError: If there's an error reading the PDF file.
        """
        self.logger.debug(f"Starting PDF to Markdown conversion for file: {pdf_path}")
        self.validate_pdf_path(pdf_path)

        try:
            page_count = self.get_pdf_page_count(pdf_path)

            if page_count > 1 and split_pages:
                pdf_pages = self.split_pdf_into_pages(pdf_path)
            else:
                pdf_pages = [pdf_path]

            markdown_text = ""
            for pdf_page in pdf_pages:
                # Read the PDF file
                self.logger.debug(f"Reading PDF file: {pdf_page}")
                base64_string = self.read_pdf_as_base64(pdf_page)

                prompt = "Convert the attached pdf to markdown format for me please."

                ocr_response = self.client.messages.create(  # type: ignore
                    model=self.model,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                    system=self.SYSTEM_PROMPT,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {  # type: ignore
                                    "type": "document",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "application/pdf",
                                        "data": base64_string,
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": prompt,
                                },
                            ],
                        }
                    ],
                )
                markdown_text += ocr_response.content[0].text + "\n\n"  # type: ignore

            self.logger.info("Successfully received response from Anthropic API")
            return markdown_text

        except Exception as e:
            self.logger.error(f"Error during PDF conversion: {str(e)}", exc_info=True)
            raise
