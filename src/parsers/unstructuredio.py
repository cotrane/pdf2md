"""UnstructuredIO parser for PDF to Markdown conversion."""

import os

import markdownify
import unstructured_client
from unstructured_client.models import operations, shared

from .base import BaseParser


class UnstructuredIOParser(BaseParser):
    """A class to handle PDF to Markdown conversion using UnstructuredIO."""

    AVAILABLE_MODELS = [
        "gpt-4o",
        "gpt-3.5-turbo",
    ]
    DEFAULT_MODEL = "gpt-4o"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.1,
        top_p: float = 0.1,
        top_k: int = 40,
        max_tokens: int = 8192,
    ) -> None:
        """Initialize the UnstructuredIO parser.

        Args:
            model: The name of the VLM model to use. Defaults to "gpt-4o".
            temperature: The temperature parameter for text generation (not used).
            top_p: The top_p parameter for text generation (not used).
            top_k: The top_k parameter for text generation (not used).
            max_tokens: The maximum number of tokens to generate (not used).

        Raises:
            ValueError: If the UNSTRUCTURED_API_KEY environment variable is not set.
            ValueError: If the model is not supported.
        """
        if model not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Unsupported model: {model}. Available models: {', '.join(self.AVAILABLE_MODELS)}"
            )

        super().__init__(model, temperature, top_p, top_k, max_tokens)

        api_key = os.getenv("UNSTRUCTURED_API_KEY")
        if not api_key:
            raise ValueError("UNSTRUCTURED_API_KEY environment variable not set")

        self.client = unstructured_client.UnstructuredClient(api_key_auth=api_key)
        self.logger.debug("UnstructuredIO parser initialization complete")

    def convert_pdf_to_markdown(self, pdf_path: str) -> str:
        """Convert a PDF file to markdown using UnstructuredIO.

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
            # Prepare the request
            req = operations.PartitionRequest(
                partition_parameters=shared.PartitionParameters(
                    files=shared.Files(
                        content=open(pdf_path, "rb"),
                        file_name=pdf_path,
                    ),
                    strategy=shared.Strategy.VLM,
                    vlm_model=self.model,  # type: ignore
                    vlm_model_provider="openai",  # type: ignore
                    languages=["eng"],
                    split_pdf_page=True,
                    split_pdf_allow_failed=True,
                    split_pdf_concurrency_level=15,
                ),
            )

            # Make the API request
            res = self.client.general.partition(request=req)

            if res.elements is None:
                self.logger.warning("No elements found in the result")
                return ""

            # Convert elements to HTML
            html_string = """
                <!DOCTYPE html><html>
                <head><style>table, th, td { border: 1px solid; }</style></head>
                <body>
            """

            for element in res.elements:
                if "text_as_html" in element["metadata"]:
                    html_string += f"{element['metadata']['text_as_html']}\n"
            html_string += "</body></html>"

            # Convert HTML to Markdown
            markdown_text = markdownify.markdownify(html_string)
            self.logger.debug("Successfully converted PDF to markdown using UnstructuredIO")
            return markdown_text

        except Exception as e:
            self.logger.error(f"Error during PDF conversion: {str(e)}", exc_info=True)
            raise
