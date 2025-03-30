"""Mistral parser for PDF to Markdown conversion."""

import os
from pathlib import Path

from mistralai import Mistral, OCRResponse

from .base import BaseParser


class MistralParser(BaseParser):
    """A class to handle PDF to Markdown conversion using Mistral's models."""

    AVAILABLE_MODELS = [
        "mistral-ocr-latest",
    ]
    DEFAULT_MODEL = "mistral-ocr-latest"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.1,
        top_p: float = 0.1,
        top_k: int = 40,
        max_tokens: int = 8192,
    ) -> None:
        """Initialize the Mistral parser.

        Args:
            model: The name of the Mistral model to use.
            temperature: The temperature parameter for text generation.
            top_p: The top_p parameter for text generation.
            top_k: The top_k parameter for text generation.
            max_tokens: The maximum number of tokens to generate.

        Raises:
            ValueError: If the MISTRAL_API_KEY environment variable is not set.
            ValueError: If the model is not supported.
        """
        if model not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Unsupported model: {model}. Available models: {', '.join(self.AVAILABLE_MODELS)}"
            )

        super().__init__(model, temperature, top_p, top_k, max_tokens)

        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY environment variable not set")

        self.client = Mistral(api_key=api_key)
        self.logger.debug("Mistral parser initialization complete")

    def convert_pdf_to_markdown(self, pdf_path: str) -> str:
        """Convert a PDF file to markdown using Mistral.

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

            # Upload the PDF file
            uploaded_pdf = self.client.files.upload(
                file={
                    "file_name": pdf_file.name,
                    "content": pdf_file.read_bytes(),
                },
                purpose="ocr",
            )

            # Get signed URL for the uploaded file
            signed_url = self.client.files.get_signed_url(file_id=uploaded_pdf.id)

            # Process the PDF with OCR
            ocr_response = self.client.ocr.process(
                model=self.model,
                document={
                    "type": "document_url",
                    "document_url": signed_url.url,
                },
                include_image_base64=True,
            )

            # Combine markdown from all pages
            markdown_text = "\n\n".join(page.markdown for page in ocr_response.pages)
            self.logger.debug("Successfully received response from Mistral API")
            return markdown_text

        except Exception as e:
            self.logger.error(f"Error during PDF conversion: {str(e)}", exc_info=True)
            raise
