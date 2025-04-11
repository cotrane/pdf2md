"""Hugging Face parser for PDF to Markdown conversion."""

import base64
import io
import os
from pathlib import Path
from typing import Generator

import tenacity
from huggingface_hub import InferenceClient, InferenceTimeoutError
from pdf2image import convert_from_path
from PIL import Image

from .base import BaseParser


class HuggingFaceParser(BaseParser):
    """A class to handle PDF to Markdown conversion using Hugging Face's inference endpoints."""

    AVAILABLE_MODELS = [
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        "Qwen/Qwen2.5-VL-7B-Instruct",
    ]
    DEFAULT_MODEL = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    PROVIDER_TO_ENV_VAR = {
        "hf-inference": "HUGGINGFACE_API_KEY",
    }
    MAX_IMAGE_WIDTH = 1920

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        provider: str = "hf-inference",
        temperature: float = 0.1,
        top_p: float = 0.1,
        top_k: int = 40,
        max_tokens: int = 8192,
    ) -> None:
        """Initialize the Hugging Face parser.

        Args:
            model: The name of the Hugging Face model to use.
            provider: The provider to use.
            temperature: The temperature parameter for text generation.
            top_p: The top_p parameter for text generation.
            top_k: The top_k parameter for text generation.
            max_tokens: The maximum number of tokens to generate.

        Raises:
            ValueError: If the HUGGINGFACE_API_KEY environment variable is not set.
            ValueError: If the model is not supported.
        """
        if model not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Unsupported model: {model}. Available models: {', '.join(self.AVAILABLE_MODELS)}"
            )

        super().__init__(model, temperature, top_p, top_k, max_tokens)

        api_key = os.getenv(self.PROVIDER_TO_ENV_VAR[provider])
        if not api_key:
            raise ValueError(f"{self.PROVIDER_TO_ENV_VAR[provider]} environment variable not set")

        self.api_key = api_key
        self.client = InferenceClient(
            provider=provider,
            api_key=api_key,
        )
        self.logger.debug("Hugging Face parser initialization complete")

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

                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format="PNG")
                img_byte_arr = img_byte_arr.getvalue()

                yield base64.b64encode(img_byte_arr).decode("utf-8")
        except Exception as e:
            self.logger.error(f"Error converting PDF to images: {str(e)}", exc_info=True)
            raise

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=15),
        retry=tenacity.retry_if_exception_type(InferenceTimeoutError),
    )
    def generate_response(self, prompt: str, base64_img_str: str) -> str:
        """Generate a response from the Hugging Face API.

        Args:
            prompt (str): The prompt to generate a response for.
            base64_image (str): The base64 encoded image string.

        Returns:
            str: The generated response.

        Raises:
            Exception: If there's an error generating the response.
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": prompt,
                        "images": ["data:image/png;base64," + base64_img_str],
                    },
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                stream=True,
            )
            output = ""
            for chunk in completion:
                output += chunk.choices[0].delta.content or ""
            return output
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}", exc_info=True)
            raise

    def convert_pdf_to_markdown(self, pdf_path: str, split_pages: bool = False) -> str:
        """Convert a PDF file to markdown using Gemini.

        Args:
            pdf_path: The path to the PDF file to convert.
            split_pages: Whether to split the PDF into pages. Default is False.

        Returns:
            The converted markdown text.

        Raises:
            FileNotFoundError: If the PDF file doesn't exist.
            IOError: If there's an error reading the PDF file.
        """
        self.logger.debug(f"Starting PDF to Markdown conversion for file: {pdf_path}")

        try:
            # In order to work around the free tier output token limit, we split the PDF into pages
            # and process each page separately.
            markdown_text = ""
            for base64_image in self.read_pdf_as_base64_img(pdf_path):
                prompt = "Convert this image to markdown format for me please."

                response_text = self.generate_response(prompt, base64_image)

                markdown_text += response_text + "\n\n"

            self.logger.debug("Successfully received response from Gemini API")

            return markdown_text

        except Exception as e:
            self.logger.error(f"Error during PDF conversion: {str(e)}", exc_info=True)
            raise
