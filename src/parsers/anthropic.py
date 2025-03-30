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

    def convert_pdf_to_markdown(self, pdf_path: str) -> str:
        """Convert a PDF file to markdown using Anthropic.

        Args:
            pdf_path: The path to the PDF file to convert.

        Returns:
            The converted markdown text.

        Raises:
            FileNotFoundError: If the PDF file doesn't exist.
            IOError: If there's an error reading the PDF file.
        """
        self.logger.debug(f"Starting PDF to Markdown conversion for file: {pdf_path}")
        self.validate_pdf_path(pdf_path)

        try:
            # Read the PDF file
            self.logger.debug(f"Reading PDF file: {pdf_path}")
            pdf_file = Path(pdf_path)
            with open(pdf_file, "rb") as f:
                binary_data = f.read()
                base64_encoded_data = base64.standard_b64encode(binary_data)
                base64_string = base64_encoded_data.decode("utf-8")

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

            self.logger.info("Successfully received response from Anthropic API")
            return ocr_response.content[0].text  # type: ignore

        except Exception as e:
            self.logger.error(f"Error during PDF conversion: {str(e)}", exc_info=True)
            raise
