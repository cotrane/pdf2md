"""AWS Textract parser for PDF to Markdown conversion."""

import os

from textractcaller.t_call import Textract_Features
from textractor import Textractor
from textractor.data.text_linearization_config import TextLinearizationConfig

from .base import BaseParser


class TextractParser(BaseParser):
    """A class to handle PDF to Markdown conversion using AWS Textract."""

    AVAILABLE_MODELS: list[str] = []
    DEFAULT_MODEL: str = ""

    def __init__(
        self,
        region_name: str = os.getenv("AWS_REGION", "eu-west-1"),
        max_retries: int = 3,
    ) -> None:
        """Initialize the Textract parser.

        Args:
            region_name: AWS region name for Textract service.
                Defaults to "eu-west-1".
            max_retries: Maximum number of retries for AWS API calls.
                Defaults to 3.

        Raises:
            ClientError: If there's an error initializing the AWS client.
        """
        super().__init__("textract", 0.0, 0.0, 0, 0)
        self.region_name = region_name
        self.max_retries = max_retries
        self.client = Textractor(region_name=region_name)
        self.config = TextLinearizationConfig(
            table_linearization_format="markdown",
            table_remove_column_headers=True,
            table_column_header_threshold=0.7,
            table_duplicate_text_in_merged_cells=False,
            title_prefix="# ",
            section_header_prefix="## ",
            hide_figure_layout=True,  # Hide pandas column numbering
        )
        self.logger.debug("Textract parser initialization complete")

    def convert_pdf_to_markdown(self, pdf_path: str, *, split_pages: bool = True) -> str:
        """Convert a PDF file to markdown using AWS Textract.

        Args:
            pdf_path: The path to the PDF file to convert.
            split_pages: Whether to split the PDF into pages. Default is True.
        Raises:
            FileNotFoundError: If the PDF file doesn't exist.
            IOError: If there's an error reading the PDF file.
            ClientError: If there's an error calling Textract API.

        Returns:
            str: The converted markdown text.
        """
        self.logger.debug(f"Starting PDF to Markdown conversion for file: {pdf_path}")

        try:
            page_count = self.get_pdf_page_count(pdf_path)

            if page_count > 1 and split_pages:
                pdf_pages = self.split_pdf_into_pages(pdf_path)
            else:
                pdf_pages = [pdf_path]

            markdown_text = ""
            for pdf_page in pdf_pages:
                # Analyze PDF with Textract
                document = self.client.analyze_document(
                    file_source=pdf_page,
                    features=[Textract_Features.LAYOUT, Textract_Features.TABLES],
                )
                markdown_text += document.get_text(self.config) + "\n\n"

            return markdown_text

        except Exception as e:
            self.logger.error(f"Error during PDF conversion: {str(e)}", exc_info=True)
            raise
