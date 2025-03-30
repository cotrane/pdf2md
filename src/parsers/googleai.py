"""Gemini parser for PDF to Markdown conversion."""

import os
from pathlib import Path

from google import genai

from .base import BaseParser


class GoogleAIParser(BaseParser):
    """A class to handle PDF to Markdown conversion using Google's Gemini models."""

    AVAILABLE_MODELS = [
        "gemini-1.5-flash",
        "gemini-2.0-flash",
        "gemini-2.0-flash-thinking-exp-01-21",
        "gemini-2.5-pro-exp-03-25",
    ]
    DEFAULT_MODEL = "gemini-2.0-flash"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.1,
        top_p: float = 0.1,
        top_k: int = 40,
        max_tokens: int = 8192,
    ) -> None:
        """Initialize the Gemini parser.

        Args:
            model: The name of the Gemini model to use.
            temperature: The temperature parameter for text generation.
            top_p: The top_p parameter for text generation.
            top_k: The top_k parameter for text generation.
            max_tokens: The maximum number of tokens to generate.

        Raises:
            ValueError: If the GOOGLE_API_KEY environment variable is not set.
            ValueError: If the model is not supported.
        """
        if model not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Unsupported model: {model}. Available models: {', '.join(self.AVAILABLE_MODELS)}"
            )

        super().__init__(model, temperature, top_p, top_k, max_tokens)

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")

        self.client = genai.Client(api_key=api_key)
        self.logger.debug("Gemini parser initialization complete")

    def convert_pdf_to_markdown(self, pdf_path: str) -> str:
        """Convert a PDF file to markdown using Gemini.

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
            pdf_file_id = self.client.files.upload(file=pdf_file)

            prompt = "Convert the attached pdf to markdown format for me please."

            response = self.client.models.generate_content(
                model=self.model,
                config=genai.types.GenerateContentConfig(  # type: ignore
                    system_instruction=self.SYSTEM_PROMPT,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    max_output_tokens=self.max_tokens,
                ),
                contents=[
                    prompt,
                    pdf_file_id,
                ],
            )

            self.logger.debug("Successfully received response from Gemini API")
            return response.text

        except Exception as e:
            self.logger.error(f"Error during PDF conversion: {str(e)}", exc_info=True)
            raise
