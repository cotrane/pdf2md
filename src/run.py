"""Main entry point for PDF to Markdown conversion."""

import argparse
import logging
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from parsers.anthropic import AnthropicParser
from parsers.googleai import GoogleAIParser
from parsers.mistral import MistralParser
from parsers.ollama import OllamaParser
from parsers.openai import OpenAIParser
from parsers.textract import TextractParser
from parsers.unstructuredio import UnstructuredIOParser
from utils.logging import setup_logging

# Load environment variables
load_dotenv()

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

# Define available parsers
PARSERS = {
    "anthropic": AnthropicParser,
    "googleai": GoogleAIParser,
    "mistral": MistralParser,
    "ollama": OllamaParser,
    "openai": OpenAIParser,
    "textract": TextractParser,
    "unstructuredio": UnstructuredIOParser,
}


def get_parser(parser_name: str, model: str | None = None) -> Any:
    """Get a parser instance by name.

    Args:
        parser_name: The name of the parser to use.
        model: The model to use with the parser.

    Returns:
        A parser instance.

    Raises:
        ValueError: If the parser name is not supported.
    """
    if parser_name not in PARSERS:
        raise ValueError(
            f"Unsupported parser: {parser_name}. Available parsers: {', '.join(PARSERS.keys())}"
        )

    parser_class = PARSERS[parser_name]
    return parser_class(model=model) if model else parser_class()


def generate_output_path(
    input_path: str, output_dir: str, parser_name: str, model: str | None = None
) -> Path:
    """Generate the output path for the markdown file.

    Args:
        input_path: The path to the input PDF file.
        output_dir: The directory to save the output file.
        parser_name: The name of the parser being used.
        model: The model being used (optional).

    Returns:
        The full output path for the markdown file.
    """
    input_filename = Path(input_path).stem
    model_suffix = f"_{model}" if model else ""
    output_filename = f"{input_filename}_{parser_name}{model_suffix}.md"
    return Path(output_dir) / output_filename


def get_parser_help_text() -> str:
    """Get help text for available parsers and their models.

    Returns:
        A formatted string containing parser and model information.
    """
    help_text = "Available parsers and their models:\n"
    for parser_name, parser_class in PARSERS.items():
        help_text += f"\n{parser_name}:\n"
        help_text += f"  Models: {', '.join(parser_class.AVAILABLE_MODELS)}\n"
        help_text += f"  Default: {parser_class.DEFAULT_MODEL}\n"
    return help_text


def setup_cli_args() -> argparse.Namespace:
    """Set up and parse command line arguments.

    Returns:
        The parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            f"Convert PDF files to Markdown using various AI models.\n\n{get_parser_help_text()}"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input_pdf_path",
        type=str,
        required=True,
        help="The path to the PDF file to convert.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="./output",
        help="The directory to save the output Markdown file. Defaults to ./output.",
    )
    parser.add_argument(
        "-p",
        "--parser",
        type=str,
        required=True,
        choices=list(PARSERS.keys()),
        help="The parser to use for conversion.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="The model to use with the parser. If not specified, uses parser default.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="The logging level to use. Defaults to INFO.",
    )

    args = parser.parse_args()

    # If model is specified, validate it against the parser's available models
    parser_class = PARSERS[args.parser]
    if args.model:
        if args.model not in parser_class.AVAILABLE_MODELS:
            parser.error(
                f"Invalid model '{args.model}' for parser '{args.parser}'. "
                f"Available models: {', '.join(parser_class.AVAILABLE_MODELS)}"
            )
    else:
        args.model = parser_class.DEFAULT_MODEL

    return args


def main() -> None:
    """Main entry point for the PDF to Markdown conversion script."""
    args = setup_cli_args()

    try:
        # Set up logging with specified level
        setup_logging(args.log_level)
        logger.info(f"Starting PDF to Markdown conversion with {args.parser} parser")

        # Get the parser instance
        pdf_parser = get_parser(args.parser, args.model)

        # Generate output path
        output_path = generate_output_path(
            args.input_pdf_path, args.output_dir, args.parser, args.model
        )

        # Convert PDF to markdown
        markdown_text = pdf_parser.convert_pdf_to_markdown(args.input_pdf_path)

        # Save the markdown
        pdf_parser.save_markdown(markdown_text, str(output_path))

        logger.info(f"Successfully converted {args.input_pdf_path} to {output_path}")

    except Exception as e:
        logger.error(f"Error during conversion: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
