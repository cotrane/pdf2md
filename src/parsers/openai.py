"""OpenAI parser for PDF to Markdown conversion."""

import os
from pathlib import Path

import openai

from .base import BaseParser


class OpenAIParser(BaseParser):
    """A class to handle PDF to Markdown conversion using OpenAI's models."""

    AVAILABLE_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4.5-preview", "o1"]
    DEFAULT_MODEL = "gpt-4o"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.1,
        top_p: float = 0.1,
        top_k: int = 40,
        max_tokens: int = 8192,
    ) -> None:
        """Initialize the OpenAI parser.

        Args:
            model: The name of the OpenAI model to use.
            temperature: The temperature parameter for text generation.
            top_p: The top_p parameter for text generation.
            top_k: The top_k parameter for text generation.
            max_tokens: The maximum number of tokens to generate.

        Raises:
            ValueError: If the OPENAI_API_KEY environment variable is not set.
            ValueError: If the model is not supported.
        """
        if model not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Unsupported model: {model}. Available models: {', '.join(self.AVAILABLE_MODELS)}"
            )

        super().__init__(model, temperature, top_p, top_k, max_tokens)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        openai.api_key = api_key
        self.logger.debug("OpenAI parser initialization complete")

    def convert_pdf_to_markdown(self, pdf_path: str) -> str:
        """Convert a PDF file to markdown using OpenAI.

        Args:
            pdf_path: The path to the PDF file to convert.

        Returns:
            The converted markdown text.

        Raises:
            FileNotFoundError: If the PDF file doesn't exist.
            IOError: If there's an error reading the PDF file.
            ValueError: If the API response is empty or invalid.
        """
        self.logger.debug(f"Starting PDF to Markdown conversion for file: {pdf_path}")
        self.validate_pdf_path(pdf_path)

        try:
            # Read the PDF file
            self.logger.debug(f"Reading PDF file: {pdf_path}")
            pdf_file = Path(pdf_path)
            base64_string = self.read_pdf_as_base64(pdf_path)

            prompt = "Convert the attached pdf to markdown format for me please."

            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self.SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": [
                            {  # type: ignore
                                "type": "file",
                                "file": {
                                    "filename": pdf_file.name,
                                    "file_data": f"data:application/pdf;base64,{base64_string}",
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    },
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                max_completion_tokens=self.max_tokens,
            )

            self.logger.debug("Successfully received response from OpenAI API")

            # Check if we got a valid response
            if not response.choices:
                raise ValueError("No response choices received from OpenAI API")

            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response content received from OpenAI API")

            return content

        except Exception as e:
            self.logger.error(f"Error during PDF conversion: {str(e)}", exc_info=True)
            raise
